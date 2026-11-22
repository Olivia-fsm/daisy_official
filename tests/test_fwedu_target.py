"""The FineWeb-Edu target (daisy/targets/build_fwedu_target.py).

A synthetic FineWeb-Edu dump (two parquet files with the columns the builder reads) with planted documents, each of
which must be removed at its stage: int_score below the threshold; text hash equal to an index document's; URL hash
equal to an index document's; a repeat of another candidate's text; an n-gram of a held-out benchmark item; a
near-duplicate of a held-out FineWeb document. Checks the stage counts, that no removed document is written, that the
held-out set and the tasks are disjoint and read back as the documents' tokens (+ EOS, cut as DocStore.doc), and that
the trainer runs with these targets ("targets" records carry one loss per task). The same dump in the public dataset's
layout (flat columns under DIR/data/<dump>) gives the same target as the layout with a metadata struct.

    cd tests && python3 -m unittest test_fwedu_target
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from helpers import cfg_for, make_benchmarks, make_index, records, run  # noqa: E402
from daisy.index.build_index import h64  # noqa: E402
from daisy.labels.targets import EOT, load_items  # noqa: E402

BUILD = os.path.join(ROOT, "daisy", "targets", "build_fwedu_target.py")
MAXDOC, VAL, NT, DPT = 64, 8, 2, 10
ITEM = "which gas do plants take in from the air to make food carbon dioxide"


def words(rng, n):
    return " ".join("".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), size=rng.integers(3, 9))) for _ in range(n))


def toks(text):
    t = np.append(np.frombuffer(text.encode("utf-8"), np.uint8).astype(np.uint16), EOT)
    return np.concatenate([t[: MAXDOC - 1], t[-1:]]) if len(t) > MAXDOC else t


class FwEduTargetTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        rng = np.random.default_rng(0)
        self.rng = rng
        self.idx = os.path.join(self.root, "idx")
        os.makedirs(self.idx)
        make_index(self.idx, rng)
        # screening benchmark: one task whose val item is ITEM (bytes)
        self.screen = os.path.join(self.root, "screen")
        os.makedirs(os.path.join(self.screen, "taska"))
        it = np.append(np.frombuffer(ITEM.encode(), np.uint8).astype(np.uint16), EOT)
        arr = np.full(128, EOT, np.uint16)
        arr[: len(it)] = it
        arr.tofile(os.path.join(self.screen, "taska", "val.bin"))
        with open(os.path.join(self.screen, "taska", "val.len"), "wb") as f:
            np.save(f, np.array([128], np.int64))
        # a held-out FineWeb document (topic 0, first in row order) made of bytes, to plant a near-duplicate of
        held = np.load(os.path.join(self.idx, "heldout_rows.npy"))
        topic = np.load(os.path.join(self.idx, "heldout_topic.npy"))
        r0 = int(np.sort(held[topic == 0])[0])
        offs = np.load(os.path.join(self.idx, "offsets.i64.npy"))
        L = int(offs[r0 + 1] - offs[r0])
        self.fw_text = "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz "), size=L - 1))
        tk = np.memmap(os.path.join(self.idx, "tokens.u16"), dtype=np.uint16, mode="r+")
        tk[offs[r0]: offs[r0 + 1]] = np.append(np.frombuffer(self.fw_text.encode(), np.uint8), EOT)
        tk.flush()
        del tk

    def tearDown(self):
        shutil.rmtree(self.root)

    def make_edu(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        rng = self.rng
        rows = []
        for i in range(90):
            rows.append({"id": f"d{i}", "text": words(rng, int(rng.integers(8, 60))), "url": f"https://x.org/{i}",
                         "int_score": int(rng.choice([4, 5, 4, 2, 3]))})
        plant = {
            "p_low": {"text": words(rng, 20), "int_score": 3},
            "p_text": {"text": words(rng, 20), "int_score": 4},
            "p_url": {"text": words(rng, 20), "int_score": 5},
            "p_rep_a": {"text": "repeated text " + words(rng, 15), "int_score": 4},
            "p_bench": {"text": words(rng, 6) + " " + ITEM + " " + words(rng, 6), "int_score": 4},
            "p_fw": {"text": self.fw_text, "int_score": 4},
        }
        plant["p_rep_b"] = dict(plant["p_rep_a"])
        for k, v in plant.items():
            rows.append({"id": k, "url": f"https://y.org/{k}", **v})
        # index hashes: p_text's text and p_url's URL become index documents' hashes
        th = np.load(os.path.join(self.idx, "text_hash.npy"))
        uh = np.load(os.path.join(self.idx, "url_hash.npy"))
        th[7] = h64(plant["p_text"]["text"])
        uh[9] = h64("https://y.org/p_url")
        np.save(os.path.join(self.idx, "text_hash.npy"), th)
        np.save(os.path.join(self.idx, "url_hash.npy"), uh)
        d = os.path.join(self.root, "edu", "CC-MAIN-2023-06")
        flat = os.path.join(self.root, "edu_flat", "data", "CC-MAIN-2023-06")     # the public dataset's layout
        os.makedirs(d)
        os.makedirs(flat)
        order = rng.permutation(len(rows))
        for fi, part in enumerate(np.array_split(order, 2)):
            sel = [rows[i] for i in part]
            md = pa.array([{"dump": "CC-MAIN-2023-06", "file_path": "f", "int_score": r["int_score"], "language": "en",
                            "language_score": 0.9, "pii_count": 0, "score": r["int_score"] + 0.1,
                            "token_count": len(r["text"]), "url": r["url"]} for r in sel])
            pq.write_table(pa.table({"text": [r["text"] for r in sel], "id": [r["id"] for r in sel], "metadata": md}),
                           os.path.join(d, f"000_0000{fi}.parquet"))
            pq.write_table(pa.table({
                "text": [r["text"] for r in sel], "id": [r["id"] for r in sel], "dump": ["CC-MAIN-2023-06"] * len(sel),
                "url": [r["url"] for r in sel], "file_path": ["f"] * len(sel), "language": ["en"] * len(sel),
                "language_score": [0.9] * len(sel), "token_count": [len(r["text"]) for r in sel],
                "score": [r["int_score"] + 0.1 for r in sel], "int_score": [r["int_score"] for r in sel]}),
                os.path.join(flat, f"000_0000{fi}.parquet"))
        return {r["id"]: r for r in rows}

    def build(self, out, edu="edu"):
        subprocess.run([sys.executable, BUILD, "--index", self.idx, "--bench", self.screen, "--out", out,
                        "--edu_dir", os.path.join(self.root, edu), "--files", "2", "--n_tasks", str(NT),
                        "--docs_per_task", str(DPT), "--val_docs", str(VAL), "--max_doc_tokens", str(MAXDOC),
                        "--val_docs_per_topic", "2", "--tasks", "taska", "--lmeval", "0", "--tokenizer", "bytes",
                        "--seq_len", "32", "--target_tokens", "100"], check=True, capture_output=True, text=True)
        with open(os.path.join(out, "report.json")) as f:
            return json.load(f)

    def test_flat_layout(self):
        """The public dataset's layout (flat columns, DIR/data/<dump>) gives the target of the metadata-struct layout."""
        import pyarrow.parquet as pq
        self.make_edu()
        out_n, out_f = os.path.join(self.root, "fwedu"), os.path.join(self.root, "fwedu_flat")
        rep_n, rep_f = self.build(out_n), self.build(out_f, "edu_flat")
        for k in ("stages", "screen_items", "heldout", "tasks", "int_score_counts", "targets_flag"):
            self.assertEqual(rep_f[k], rep_n[k], k)
        self.assertEqual(pq.read_table(os.path.join(out_f, "docs.parquet")).to_pydict(),
                         pq.read_table(os.path.join(out_n, "docs.parquet")).to_pydict())
        files = [("fwedu_theldout", "val")] + [(f"fwedu_t{k:02d}", "train") for k in range(NT)]
        for name, split in files:
            for ext in ("bin", "len"):
                with open(os.path.join(out_n, name, f"{split}.{ext}"), "rb") as f, \
                        open(os.path.join(out_f, name, f"{split}.{ext}"), "rb") as g:
                    self.assertEqual(f.read(), g.read(), (name, ext))

    def test_builder_and_trainer(self):
        import pyarrow.parquet as pq
        rows = self.make_edu()
        out = os.path.join(self.root, "fwedu")
        rep = self.build(out)
        st = rep["stages"]
        self.assertEqual(st["rows_read"], len(rows))
        self.assertEqual(st["int_score_ok"], sum(r["int_score"] >= 4 for r in rows.values()))
        self.assertEqual((st["dropped_text_in_index"], st["dropped_url_in_index"], st["dropped_repeats"]), (1, 1, 1))
        self.assertEqual(st["dropped_eval_overlap"]["val/taska"], 1)
        self.assertEqual(st["dropped_eval_overlap"]["fineweb_heldout"], 1)
        self.assertEqual(rep["targets_flag"], "fwedu_t00,fwedu_t01")
        dp = pq.read_table(os.path.join(out, "docs.parquet")).to_pydict()
        ids = dp["id"]
        self.assertEqual(len(ids), VAL + NT * DPT)
        self.assertEqual(len(set(ids)), len(ids))
        for k in ("p_low", "p_text", "p_url", "p_bench", "p_fw"):
            self.assertNotIn(k, ids)
        self.assertFalse({"p_rep_a", "p_rep_b"} <= set(ids))
        self.assertTrue(all(s >= 4 for s in dp["int_score"]))
        self.assertEqual(dp["role"], ["heldout"] * VAL + [f"fwedu_t{k:02d}" for k in range(NT) for _ in range(DPT)])
        held = load_items(os.path.join(out, "fwedu_theldout"), "val")
        self.assertEqual(len(held), VAL)
        for x, i in zip(held, ids[:VAL]):
            np.testing.assert_array_equal(x, toks(rows[i]["text"]))
        for k in range(NT):
            items = load_items(os.path.join(out, f"fwedu_t{k:02d}"), "train")
            mine = ids[VAL + k * DPT: VAL + (k + 1) * DPT]
            self.assertEqual(len(items), DPT)
            for x, i in zip(items, mine):
                np.testing.assert_array_equal(x, toks(rows[i]["text"]))
            self.assertTrue(set(rep["tasks"][f"fwedu_t{k:02d}"]["used_ids"]) <= set(mine))
        # the trainer reads these targets unchanged
        bench = os.path.join(self.root, "bench")
        make_benchmarks(bench, self.rng)
        exp = os.path.join(self.root, "run")
        extra = ["--daisy_select_mult", "2", "--daisy_targets", rep["targets_flag"], "--daisy_target_dir", out,
                 "--daisy_eval_dir", bench, "--daisy_eval_tasks", "taska,taskb"]
        run(cfg_for(self.idx, bench, 20, extra), exp)
        recs = records(exp)
        tg = [r for r in recs if r["kind"] == "targets"]
        self.assertTrue(tg)
        for r in tg:
            self.assertEqual(sorted(k for k in r if k.startswith("loss/")), ["loss/fwedu_t00", "loss/fwedu_t01"])


if __name__ == "__main__":
    unittest.main()
