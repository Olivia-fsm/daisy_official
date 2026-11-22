"""FineWeb-Edu scores of the pool's documents (daisy/targets/edu_scores.py) and the pool's keep filter
(--daisy_pool_keep, daisy/data/pool.py).

  * edu_scores: a synthetic pool of two dumps (uuid ids, plus a non-uuid id) and synthetic FineWeb-Edu files per dump:
    every shard's .u8 holds exactly each document's int_score in its own dump's files (0 when absent; an id present
    only in the other dump's files is not matched; a duplicated edu id takes its largest score); the report counts add
    up; a rerun keeps the files; the same scores from the FineWeb-Edu files in the public dataset's layout (flat
    columns under DIR/data/<dump>) as from the layout with a metadata struct;
  * the stream with keep_dir / keep_min draws only documents with score >= keep_min that are not held out, and with a
    filter that keeps every document it is identical to the unfiltered stream.

    cd tests && python3 -m unittest test_edu_scores
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from test_pool import make_pool  # noqa: E402
from daisy.data.pool import DsPool, ShardDocStream  # noqa: E402

TOOL = os.path.join(ROOT, "daisy", "targets", "edu_scores.py")
NS = 8


def edu_md(scores, ids):
    return pa.array([{"dump": "d", "file_path": "f", "int_score": int(s), "language": "en", "language_score": 0.9,
                      "pii_count": 0, "score": float(s) + 0.2, "token_count": 10, "url": f"u{i}"}
                     for s, i in zip(scores, ids)])


def edu_flat(scores, ids, dump):
    """The same rows with the columns of the public dataset."""
    n = len(ids)
    return pa.table({"text": ["x"] * n, "id": ids, "dump": [dump] * n, "url": [f"u{i}" for i in ids],
                     "file_path": ["f"] * n, "language": ["en"] * n, "language_score": [0.9] * n,
                     "token_count": [10] * n, "score": [float(s) + 0.2 for s in scores],
                     "int_score": [int(s) for s in scores]})


class EduScoresTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        rng = np.random.default_rng(0)
        self.rng = rng
        self.pool = os.path.join(self.root, "pool")
        self.truth = make_pool(self.pool, rng, n_shards=NS)
        files = [f"CC-MAIN-T{1 + (s >= NS // 2)}/000_{s:05d}.parquet" for s in range(NS)]
        with open(os.path.join(self.pool, "manifest.json"), "w") as f:
            json.dump({"files": files, "pool": "x"}, f)
        self.dump = {s: os.path.dirname(files[s]) for s in range(NS)}
        # uuid ids (one shard-0 document keeps a non-uuid id), the sidecar rewritten with them
        self.ids, self.want = {}, {}
        edu = {"CC-MAIN-T1": ([], []), "CC-MAIN-T2": ([], [])}
        other = {"CC-MAIN-T1": "CC-MAIN-T2", "CC-MAIN-T2": "CC-MAIN-T1"}
        for s in range(NS):
            mp = os.path.join(self.pool, "sidecar", f"{s:05d}.meta.parquet")
            t = pq.read_table(mp)
            n = t.num_rows
            ids = [f"<urn:uuid:{uuid.UUID(int=int(rng.integers(0, 2**62)) << 64 | int(rng.integers(0, 2**62)))}>"
                   for _ in range(n)]
            if s == 0:
                ids[3] = "not-a-uuid-3"
            want = np.zeros(n, np.uint8)
            for i in range(n):
                u = rng.random()
                if u < 0.6:                         # in this dump's FineWeb-Edu files
                    sc = int(rng.integers(2, 6))
                    edu[self.dump[s]][0].append(ids[i])
                    edu[self.dump[s]][1].append(sc)
                    want[i] = sc
                elif u < 0.7:                       # only in the OTHER dump's files: not matched
                    edu[other[self.dump[s]]][0].append(ids[i])
                    edu[other[self.dump[s]]][1].append(5)
            if n > 5 and want[5] > 0:               # a duplicated edu id: the larger score counts
                edu[self.dump[s]][0].append(ids[5])
                edu[self.dump[s]][1].append(max(2, int(want[5]) - 1))
            pq.write_table(t.drop(["id"]).append_column("id", pa.array(ids)), mp)
            self.ids[s], self.want[s] = ids, want
        self.edu = os.path.join(self.root, "edu")             # T1 here, T2 only in a second copy (DIR/data/<dump>)
        self.edu2 = os.path.join(self.root, "edu2")
        self.flat = os.path.join(self.root, "flat")           # both dumps in the public dataset's layout
        for d, (ids, sc) in edu.items():
            base = self.edu if d == "CC-MAIN-T1" else os.path.join(self.edu2, "data")
            os.makedirs(os.path.join(base, d))
            os.makedirs(os.path.join(self.flat, "data", d))
            ids = ids + [f"<urn:uuid:{uuid.uuid4()}>" for _ in range(20)]      # FineWeb-Edu documents not in the pool
            sc = sc + [4] * 20
            o = rng.permutation(len(ids))
            for k, part in enumerate(np.array_split(o, 3)):
                pq.write_table(pa.table({"text": ["x"] * len(part), "id": [ids[i] for i in part],
                                         "metadata": edu_md([sc[i] for i in part], [ids[i] for i in part])}),
                               os.path.join(base, d, f"000_0000{k}.parquet"))
                pq.write_table(edu_flat([sc[i] for i in part], [ids[i] for i in part], d),
                               os.path.join(self.flat, "data", d, f"000_0000{k}.parquet"))
        self.out = os.path.join(self.root, "scores")

    def tearDown(self):
        shutil.rmtree(self.root)

    def run_tool(self, edu_dir=None, out=None):
        out = out or self.out
        r = subprocess.run([sys.executable, TOOL, "--pool", self.pool, "--edu_dir", edu_dir or f"{self.edu},{self.edu2}",
                            "--out", out, "--workers", "2"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        with open(os.path.join(out, "report.json")) as f:
            return json.load(f)

    def test_flat_layout(self):
        """The public dataset's layout (flat columns, DIR/data/<dump>) gives the scores of the metadata-struct layout."""
        nested = self.run_tool()
        out = os.path.join(self.root, "scores_flat")
        rep = self.run_tool(self.flat, out)
        for s in range(NS):
            got = np.fromfile(os.path.join(out, f"{s:05d}.u8"), dtype=np.uint8)
            np.testing.assert_array_equal(got, self.want[s], err_msg=f"shard {s}")
            with open(os.path.join(self.out, f"{s:05d}.u8"), "rb") as f, open(os.path.join(out, f"{s:05d}.u8"), "rb") as g:
                self.assertEqual(f.read(), g.read())
        self.assertEqual(rep["total"], nested["total"])
        for d in ("CC-MAIN-T1", "CC-MAIN-T2"):
            self.assertEqual(rep["dumps"][d]["edu"]["dir"], self.flat)
            for k in ("rows", "duplicate_keys", "non_uuid_ids", "files"):
                self.assertEqual(rep["dumps"][d]["edu"][k], nested["dumps"][d]["edu"][k], (d, k))

    def test_scores_and_filter(self):
        rep = self.run_tool()
        for s in range(NS):
            got = np.fromfile(os.path.join(self.out, f"{s:05d}.u8"), dtype=np.uint8)
            np.testing.assert_array_equal(got, self.want[s], err_msg=f"shard {s}")
        allw = np.concatenate([self.want[s] for s in range(NS)])
        self.assertEqual(rep["total"]["docs_by_int_score"], np.bincount(allw, minlength=6).tolist())
        self.assertEqual(rep["dumps"]["CC-MAIN-T1"]["non_uuid_pool_ids"], 1)
        self.assertEqual((rep["dumps"]["CC-MAIN-T1"]["edu"]["dir"], rep["dumps"]["CC-MAIN-T2"]["edu"]["dir"]),
                         (self.edu, self.edu2))
        self.assertEqual(sum(v["edu"]["duplicate_keys"] for v in rep["dumps"].values()),
                         sum(1 for s in range(NS) if len(self.want[s]) > 5 and self.want[s][5] > 0))
        mtime = os.path.getmtime(os.path.join(self.out, "00000.u8"))
        self.run_tool()                                            # rerun: files kept
        self.assertEqual(os.path.getmtime(os.path.join(self.out, "00000.u8")), mtime)

        # the stream draws only kept, non-held documents
        keys = sorted(self.truth)
        held = keys[5:9]
        ht, hu = [self.truth[k][2] for k in held[:2]], [self.truth[k][3] for k in held[2:]]

        def stream(keep_dir="", keep_min=0, preload=0):
            return ShardDocStream(DsPool(self.pool), seq_len=31, max_doc_tokens=64, seed=7, n_open=3, held_text=ht,
                                  held_url=hu, preload=preload, keep_dir=keep_dir, keep_min=keep_min)

        for preload in (0, 2):
            st = stream(self.out, 3, preload)
            drawn = set()
            for _ in range(20):
                drawn |= {int(k) for k in st.next_microbatch(2)["rows"]}
            self.assertTrue(drawn)
            for k in drawn:
                self.assertGreaterEqual(int(self.want[k >> 32][k & 0xFFFFFFFF]), 3)
                self.assertNotIn(k, held)
            self.assertGreater(st.filtered, 0)
        # a filter that keeps everything = no filter
        allk = os.path.join(self.root, "all5")
        os.makedirs(allk)
        for s in range(NS):
            np.full(len(self.want[s]), 5, np.uint8).tofile(os.path.join(allk, f"{s:05d}.u8"))
        a, b = stream(), stream(allk, 3)
        for _ in range(15):
            ma, mb = a.next_microbatch(2), b.next_microbatch(2)
            np.testing.assert_array_equal(ma["rows"], mb["rows"])
            np.testing.assert_array_equal(ma["x"], mb["x"])
        # a score file of the wrong length is refused
        np.zeros(3, np.uint8).tofile(os.path.join(allk, "00000.u8"))
        with self.assertRaises(ValueError):
            st = stream(allk, 3)
            for _ in range(40):
                st.next_microbatch(2)


if __name__ == "__main__":
    unittest.main()
