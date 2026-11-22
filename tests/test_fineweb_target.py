"""The FineWeb topic target (daisy/targets/build_fineweb_target.py) and --daisy_eval_tasks.

  * the builder writes, per topic, exactly the held-out reserve (the held-out documents after the first
    val_docs_per_topic of the topic in row order, i.e. not the evaluated ones), minus documents whose text or URL hash
    equals an evaluated document's; every document reads back (load_items) as DocStore.doc(row, max_doc_tokens), the
    held-out FineWeb loss's cut; the report's used rows are the documents target_batches packs;
  * the trainer with these targets and --daisy_eval_tasks: "targets" records carry one loss per topic task,
    "bench_val" records the tasks named by --daisy_eval_tasks; without the flag the evaluation follows the targets.

    cd tests && python3 -m unittest test_fineweb_target
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from helpers import cfg_for, make_benchmarks, make_index, records, run  # noqa: E402
from daisy.data.docstore import DocStore  # noqa: E402
from daisy.eval.heldout import eval_tasks  # noqa: E402
from daisy.labels.targets import load_items, target_items  # noqa: E402

BUILD = os.path.join(ROOT, "daisy", "targets", "build_fineweb_target.py")
VAL_DOCS, MAX_DOC, T = 4, 64, 64


def build(idx, out, budget=260):
    subprocess.run([sys.executable, BUILD, "--index", idx, "--out", out, "--val_docs_per_topic", str(VAL_DOCS),
                    "--max_doc_tokens", str(MAX_DOC), "--seq_len", str(T), "--target_tokens", str(budget)],
                   check=True, capture_output=True, text=True)
    with open(os.path.join(out, "report.json")) as f:
        return json.load(f)


class FineWebTargetTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        rng = np.random.default_rng(0)
        self.idx, self.bench = os.path.join(self.root, "idx"), os.path.join(self.root, "bench")
        os.makedirs(self.idx)
        make_index(self.idx, rng)          # 3 topics x 10 held-out docs, lengths 5..149 (some cut at MAX_DOC)
        make_benchmarks(self.bench, rng)
        self.held = np.load(os.path.join(self.idx, "heldout_rows.npy"))
        self.topic = np.load(os.path.join(self.idx, "heldout_topic.npy"))
        self.ev = {t: np.sort(self.held[self.topic == t])[:VAL_DOCS] for t in range(3)}
        self.res = {t: np.sort(self.held[self.topic == t])[VAL_DOCS:] for t in range(3)}

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_builder(self):
        # plant duplicates: a topic-1 reserve doc with an evaluated topic-0 doc's text, a topic-2 one with its URL
        th = np.load(os.path.join(self.idx, "text_hash.npy"))
        uh = np.load(os.path.join(self.idx, "url_hash.npy"))
        dup_t, dup_u = self.res[1][2], self.res[2][0]
        th[dup_t] = th[self.ev[0][1]]
        uh[dup_u] = uh[self.ev[0][3]]
        np.save(os.path.join(self.idx, "text_hash.npy"), th)
        np.save(os.path.join(self.idx, "url_hash.npy"), uh)
        out = os.path.join(self.root, "fw")
        rep = build(self.idx, out)
        self.assertEqual(rep["targets_flag"], "fineweb_t00,fineweb_t01,fineweb_t02")
        store = DocStore(self.idx)
        evaluated = set(np.concatenate(list(self.ev.values())).tolist())
        for t in range(3):
            name = f"fineweb_t{t:02d}"
            path = os.path.join(out, name)
            rows = np.load(os.path.join(path, "train_rows.npy"))
            want = [r for r in self.res[t] if r not in (dup_t, dup_u)]
            self.assertEqual(rows.tolist(), want)
            self.assertFalse(set(rows.tolist()) & evaluated)
            self.assertEqual(rep["topics"][name]["dropped_duplicates_of_evaluated"], int(t in (1, 2)))
            items = load_items(path, "train")
            self.assertEqual(len(items), len(rows))
            for it, r in zip(items, rows):
                np.testing.assert_array_equal(it, store.doc(int(r), MAX_DOC))
            # the report's used rows are the documents target_batches packs (seed 0 order, the last one possibly cut)
            spans = target_items(path, T, 260)
            order = np.random.default_rng(0).permutation(len(items))[:len(spans)]
            for s, i in zip(spans, order):
                np.testing.assert_array_equal(s, items[i][:len(s)])
            self.assertEqual(sorted(rep["topics"][name]["used_rows"]), sorted(rows[order].tolist()))

    def test_eval_tasks_flag(self):
        ns = types.SimpleNamespace
        self.assertEqual(eval_tasks(ns(daisy_targets="a,b", daisy_eval_tasks="")), ["a", "b"])
        self.assertEqual(eval_tasks(ns(daisy_targets="a,b")), ["a", "b"])
        self.assertEqual(eval_tasks(ns(daisy_targets="fw0,fw1", daisy_eval_tasks="a,b")), ["a", "b"])

    def test_trainer_with_fineweb_targets(self):
        out = os.path.join(self.root, "fw")
        rep = build(self.idx, out)
        exp = os.path.join(self.root, "run")
        extra = ["--daisy_select_mult", "2", "--daisy_targets", rep["targets_flag"], "--daisy_target_dir", out,
                 "--daisy_eval_dir", self.bench, "--daisy_eval_tasks", "taska,taskb"]
        run(cfg_for(self.idx, self.bench, 20, extra), exp)
        recs = records(exp)
        tg = [r for r in recs if r["kind"] == "targets"]
        bv = [r for r in recs if r["kind"] == "bench_val"]
        self.assertTrue(tg and bv)
        for r in tg:
            self.assertEqual(sorted(k for k in r if k.startswith("loss/")),
                             ["loss/fineweb_t00", "loss/fineweb_t01", "loss/fineweb_t02"])
        for r in bv:
            self.assertEqual(sorted(k for k in r if k.startswith("loss/")), ["loss/mean", "loss/taska", "loss/taskb"])


if __name__ == "__main__":
    unittest.main()
