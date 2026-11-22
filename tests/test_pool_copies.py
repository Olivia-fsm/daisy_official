"""Exact pool copies of the FineWeb-Edu target / held-out documents (daisy/targets/pool_copies.py): a synthetic pool
with planted copies (a held-out document's text twice, another's URL with other text, a used target document's text
once, an unused target document's URL), synthetic FineWeb-Edu files and a synthetic target directory: the per-group
counts, the copies' scores, and the clean held-out split (every held-out document without a copy, read back
identically, in order); the same report from the FineWeb-Edu files in the public dataset's layout (flat columns under
DIR/data/<dump>) as from the layout with a metadata struct.

    cd tests && python3 -m unittest test_pool_copies
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from test_pool import make_pool  # noqa: E402
from daisy.index.build_index import h64  # noqa: E402
from daisy.labels.targets import load_items  # noqa: E402
from daisy.targets.build_fwedu_target import write_split  # noqa: E402

TOOL = os.path.join(ROOT, "daisy", "targets", "pool_copies.py")
NS = 4


class PoolCopiesTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        rng = np.random.default_rng(0)
        self.pool = os.path.join(self.root, "pool")
        make_pool(self.pool, rng, n_shards=NS)
        files = [f"CC-MAIN-P{s % 2}/000_{s:05d}.parquet" for s in range(NS)]
        with open(os.path.join(self.pool, "manifest.json"), "w") as f:
            json.dump({"files": files, "pool": "x"}, f)
        # 6 held-out + 4 target documents
        n_h, n_t = 6, 4
        ids = [f"<urn:uuid:{k}>" for k in range(n_h + n_t)]
        texts = [f"document {k} text" for k in range(n_h + n_t)]
        urls = [f"https://site{k}.org/p" for k in range(n_h + n_t)]
        edu = os.path.join(self.root, "edu", "CC-MAIN-E")
        os.makedirs(edu)
        o = rng.permutation(n_h + n_t)
        e_ids, e_texts = [ids[i] for i in o] + ["<urn:uuid:other>"], [texts[i] for i in o] + ["not ours"]
        e_urls = [urls[i] for i in o] + ["https://other.org/p"]
        md = pa.array([{"dump": "CC-MAIN-E", "int_score": 4, "score": 4.1, "url": u} for u in e_urls])
        pq.write_table(pa.table({"id": e_ids, "text": e_texts, "metadata": md}), os.path.join(edu, "000_00000.parquet"))
        self.flat = os.path.join(self.root, "edu_flat")          # the same file in the public dataset's layout
        os.makedirs(os.path.join(self.flat, "data", "CC-MAIN-E"))
        pq.write_table(pa.table({"text": e_texts, "id": e_ids, "dump": ["CC-MAIN-E"] * len(e_ids), "url": e_urls,
                                 "score": [4.1] * len(e_ids), "int_score": [4] * len(e_ids)}),
                       os.path.join(self.flat, "data", "CC-MAIN-E", "000_00000.parquet"))
        toks = [np.append(rng.integers(0, 50000, int(rng.integers(5, 40))), 50256).astype(np.uint16)
                for _ in range(n_h + n_t)]
        self.tgt = os.path.join(self.root, "target")
        write_split(os.path.join(self.tgt, "fwedu_theldout"), toks[:n_h], "val")
        write_split(os.path.join(self.tgt, "fwedu_t00"), toks[n_h:], "train")
        roles = ["heldout"] * n_h + ["fwedu_t00"] * n_t
        pq.write_table(pa.table({"id": ids, "url": urls, "dump": ["CC-MAIN-E"] * (n_h + n_t),
                                 "file": ["000_00000.parquet"] * (n_h + n_t), "int_score": [4] * (n_h + n_t),
                                 "score": [4.1] * (n_h + n_t), "tokens": [len(t) for t in toks], "role": roles}),
                       os.path.join(self.tgt, "docs.parquet"))
        with open(os.path.join(self.tgt, "report.json"), "w") as f:
            json.dump({"args": {"edu_dir": os.path.join(self.root, "edu")},
                       "tasks": {"fwedu_t00": {"used_ids": [ids[n_h], ids[n_h + 1]]}}}, f)
        # planted copies in the pool's sidecars, and the pool documents' scores
        self.keep = os.path.join(self.root, "scores")
        os.makedirs(self.keep)
        plant = {(0, 1): ("text", 0), (1, 3): ("text", 0), (2, 0): ("url", 2), (3, 4): ("text", n_h),
                 (1, 5): ("url", n_h + 3)}
        score_at = {(0, 1): 2, (1, 3): 4, (2, 0): 3, (3, 4): 5, (1, 5): 1}
        for s in range(NS):
            mp_ = os.path.join(self.pool, "sidecar", f"{s:05d}.meta.parquet")
            t = pq.read_table(mp_).to_pydict()
            th, uh = list(t["text_hash"]), list(t["url_hash"])
            sc = np.zeros(len(th), np.uint8)
            for (ss, r), (kind, k) in plant.items():
                if ss == s:
                    if kind == "text":
                        th[r] = h64(texts[k])
                    else:
                        uh[r] = h64(urls[k])
                    sc[r] = score_at[(ss, r)]
            pq.write_table(pa.table({"id": t["id"], "text_hash": pa.array(th, pa.uint64()),
                                     "url_hash": pa.array(uh, pa.uint64())}), mp_)
            sc.tofile(os.path.join(self.keep, f"{s:05d}.u8"))
        self.toks, self.n_h = toks, n_h

    def tearDown(self):
        shutil.rmtree(self.root)

    def run_tool(self, out, clean, extra=()):
        r = subprocess.run([sys.executable, TOOL, "--pool", self.pool, "--target", self.tgt, "--keep_dir", self.keep,
                            "--out", out, "--clean_out", clean, "--workers", "2", *extra], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        with open(os.path.join(out, "report.json")) as f:
            return json.load(f)

    def test_flat_layout(self):
        """--edu_dir with the public dataset's layout (flat columns, DIR/data/<dump>): the same report and clean split."""
        out, clean = os.path.join(self.root, "out"), os.path.join(self.root, "clean")
        out2, clean2 = os.path.join(self.root, "out2"), os.path.join(self.root, "clean2")
        nested = self.run_tool(out, clean)
        rep = self.run_tool(out2, clean2, ["--edu_dir", self.flat])
        self.assertEqual(rep["edu_dir"], self.flat)
        self.assertEqual(rep["groups"], nested["groups"])
        self.assertEqual(rep["heldout_docs"], nested["heldout_docs"])
        self.assertEqual(rep["clean"]["docs"], 4)
        for name in ("val.bin", "val.len"):
            with open(os.path.join(clean, name), "rb") as f, open(os.path.join(clean2, name), "rb") as g:
                self.assertEqual(f.read(), g.read())

    def test_copies(self):
        out, clean = os.path.join(self.root, "out"), os.path.join(self.root, "clean")
        rep = self.run_tool(out, clean)
        g = rep["groups"]
        self.assertEqual((g["heldout"]["docs"], g["heldout"]["with_text_copy"], g["heldout"]["with_url_copy_only"],
                          g["heldout"]["with_any_copy"], g["heldout"]["text_copies"]), (6, 1, 1, 2, 2))
        self.assertEqual((g["heldout"]["with_text_copy_score_ge3"], g["heldout"]["with_any_copy_score_ge3"]), (1, 2))
        self.assertEqual((g["target_used"]["docs"], g["target_used"]["with_any_copy"]), (2, 1))
        self.assertEqual((g["target_written"]["docs"], g["target_written"]["with_any_copy"],
                          g["target_written"]["with_url_copy_only"]), (4, 2, 1))
        h0 = rep["heldout_docs"][0]
        self.assertEqual((h0["text_copies"], h0["max_score_text_copy"], h0["dumps"]), (2, 4, ["CC-MAIN-P0", "CC-MAIN-P1"]))
        self.assertEqual(rep["clean"]["docs"], 4)
        back = load_items(clean, "val")
        want = [self.toks[k] for k in range(self.n_h) if k not in (0, 2)]
        self.assertEqual(len(back), len(want))
        for x, y in zip(back, want):
            np.testing.assert_array_equal(x, y)


if __name__ == "__main__":
    unittest.main()
