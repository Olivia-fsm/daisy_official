"""train.py end to end on CPU (tiny Llama; a synthetic index, tokenized pool and target tasks; a compressed schedule):
uniform sampling, DAiSY and the similarity baseline.

  * every run writes its outputs (refresh.jsonl, metrics.jsonl, checkpoints, finished.json); a DAiSY run also its
    label dumps and scorer records;
  * the label gates do not change training: a uniform run with the learner ends with exactly the model of the uniform
    run without it;
  * held-out documents and their pool copies are never trained on, with and without selection;
  * until the scorer starts, a selecting run trains on uniformly drawn candidates; from then on its batches follow the
    scorer (the similarity run trains on documents close to its target direction);
  * a run resumed from a permanent checkpoint, and a chained job that stops before its time limit and resumes,
    end exactly like the uninterrupted run (model, optimizer, ridge, labels).

    cd tests && python -m unittest test_training
"""

import glob
import json
import os
import resource
import shutil
import tempfile
import unittest

import numpy as np
import torch

from helpers import EOT, main_flags, make_run_data, records, run_main

DAISY = ["--daisy_learn", "--daisy_select_mult", "4"]


def docs_of(record, keys):
    """The pool keys of the documents of every recorded micro-batch (any token of a pool document names it)."""
    out = []
    for _, x, _, doc in record:
        ks = set()
        for j in np.unique(doc):
            t = x[doc == j]
            t = t[t != EOT]
            if len(t):
                ks.add(int(keys[int((t[0] - 1) // 5)]))
        out.append(ks)
    return out


def final_state(exp):
    m = torch.load(os.path.join(exp, "ckpts", "latest", "main.pt"), map_location="cpu", weights_only=False)
    d = torch.load(os.path.join(exp, "ckpts", "latest", "daisy_r0.pt"), map_location="cpu", weights_only=False)
    return m, d


def labels_of(exp):
    cols = {}
    for f in sorted(glob.glob(os.path.join(exp, "daisy", "labels", "labels_*.npz"))):
        with np.load(f) as z:
            for k in z.files:
                cols.setdefault(k, []).append(z[k])
    return {k: np.concatenate(v) for k, v in cols.items()}


def same_model(a, b):
    return a["itr"] == b["itr"] and all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])


class TrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)      # every pool run keeps its shard files open
        want = 4096 if hard == resource.RLIM_INFINITY else min(4096, hard)
        if soft != resource.RLIM_INFINITY and soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
        cls.root = tempfile.mkdtemp(prefix="daisy_test_")
        cls.paths, cls.keys = make_run_data(cls.root)
        cls.rec, cls.exp = {}, {}
        for name, extra in (("uniform", []), ("uniform_learn", ["--daisy_learn"]), ("daisy", DAISY),
                            ("similarity", ["--daisy_select_mult", "4", "--daisy_select_cosine",
                                            "--daisy_select_static", cls.paths["w_static"]])):
            cls.rec[name] = []
            run_main(main_flags(cls.paths, name, cls.root, extra=extra), record=cls.rec[name])
            cls.exp[name] = os.path.join(cls.root, "exps", name)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    def test_outputs(self):
        for name, exp in self.exp.items():
            with self.subTest(run=name):
                self.assertEqual(len(self.rec[name]), 32 * 2)                    # 32 steps of 2 micro-batches
                self.assertEqual(json.load(open(os.path.join(exp, "daisy", "finished.json")))["iter"], 32)
                self.assertEqual([r["step"] for r in records(exp, "refresh.jsonl")], list(range(0, 33, 4)))
                kinds = [r["kind"] for r in records(exp)]
                self.assertEqual(kinds.count("bench_val"), 9)                    # the held-out downstream loss
                self.assertEqual(sorted(os.listdir(os.path.join(exp, "ckpts"))), ["0", "20", "latest"])
                learner = name in ("uniform_learn", "daisy")
                self.assertEqual("targets" in kinds, learner)
                self.assertEqual(os.path.isdir(os.path.join(exp, "daisy", "labels")), learner)
        rec = [r for r in records(self.exp["daisy"]) if r["kind"] == "targets"]
        self.assertEqual([r["step"] for r in rec], [8, 12, 16, 20, 24, 28, 32])  # from --daisy_label_start on
        self.assertEqual({k for k in rec[0] if k.startswith("loss/")}, {"loss/taska", "loss/taskb"})
        self.assertTrue(any(r["kind"] == "scorer" and r.get("n", 0) > 0 for r in records(self.exp["daisy"])))

    def test_label_gates_do_not_change_training(self):
        a, b = self.rec["uniform"], self.rec["uniform_learn"]
        self.assertEqual([r[0] for r in a], [r[0] for r in b])                   # the same losses on the same batches
        self.assertTrue(same_model(final_state(self.exp["uniform"])[0], final_state(self.exp["uniform_learn"])[0]))
        lab = labels_of(self.exp["uniform_learn"])
        self.assertEqual(sorted(set(lab["step"].tolist())), list(range(8, 32, 2)))   # every 2nd step from 8 on
        self.assertTrue(np.isfinite(lab["y"]).all() and float(np.abs(lab["y"]).max()) > 0)
        trained = docs_of(a, self.keys)
        for step in sorted(set(lab["step"].tolist())):                           # the labelled documents = the trained ones
            self.assertEqual(set(lab["key"][lab["step"] == step].tolist()), trained[2 * step] | trained[2 * step + 1])

    def test_heldout_copies_are_never_trained(self):
        banned = {0, 1, 2}                                                       # pool copies of held-out documents
        for name in self.exp:
            seen = set().union(*docs_of(self.rec[name], self.keys))
            self.assertFalse(seen & banned, name)
            self.assertGreater(len(seen), 300)

    def test_selection_starts_at_the_scorer_start(self):
        w = np.load(self.paths["w_static"])[:-1]

        def cosine(k):
            phi = np.fromfile(os.path.join(self.paths["pool"], "sidecar", f"{k >> 32:05d}.phi.f16"),
                              dtype=np.float16).reshape(-1, 768)[k & 0xFFFFFFFF].astype(np.float32)
            return float(phi @ w / np.linalg.norm(phi))
        d = docs_of(self.rec["similarity"], self.keys)
        before = np.mean([cosine(k) for mb in d[:2 * 16] for k in mb])
        after = np.mean([cosine(k) for mb in d[2 * 17:] for k in mb])           # weights set after step 16
        self.assertLess(abs(before), 0.02)
        self.assertGreater(after, before + 0.02)
        # the two selecting runs draw the same windows and differ only in the scorer: equal batches until it starts
        dd = docs_of(self.rec["daisy"], self.keys)
        self.assertEqual(dd[:2 * 16], d[:2 * 16])
        self.assertNotEqual(dd[2 * 20:], d[2 * 20:])
        ridge = final_state(self.exp["daisy"])[1]["learner"]["scorer"]["ridge"]
        self.assertEqual(tuple(ridge["w"].shape), (769,))
        self.assertGreater(ridge["n"], 0)

    def test_resume_is_exact(self):
        name = "daisy_resumed"
        exp = os.path.join(self.root, "exps", name)
        shutil.copytree(os.path.join(self.exp["daisy"], "ckpts", "20"), os.path.join(exp, "ckpts", "20"))
        rec = []
        run_main(main_flags(self.paths, name, self.root, extra=DAISY + ["--resume_from", os.path.join(exp, "ckpts", "20")]),
                 record=rec)
        self.assertEqual([r[0] for r in rec], [r[0] for r in self.rec["daisy"][2 * 20:]])
        (m1, d1), (m2, d2) = final_state(exp), final_state(self.exp["daisy"])
        self.assertTrue(same_model(m1, m2))
        self.assertTrue(torch.equal(d1["learner"]["scorer"]["ridge"]["w"], d2["learner"]["scorer"]["ridge"]["w"]))

    def test_chained_jobs_are_exact(self):
        """DAISY_STOP_AT: the job stops at a refresh step with a checkpoint and without finished.json; the next job
        resumes from ckpts/latest and ends like the uninterrupted run."""
        name = "daisy_chain"
        exp = os.path.join(self.root, "exps", name)
        flags, rec = main_flags(self.paths, name, self.root, extra=DAISY), []
        run_main(flags, env={"DAISY_STOP_AT": "1", "DAISY_STOP_MIN_ITER": "28"}, record=rec)
        self.assertEqual(final_state(exp)[0]["itr"], 28)
        self.assertFalse(os.path.exists(os.path.join(exp, "daisy", "finished.json")))
        run_main(flags, record=rec)
        self.assertEqual([r[0] for r in rec], [r[0] for r in self.rec["daisy"]])
        (m1, d1), (m2, d2) = final_state(exp), final_state(self.exp["daisy"])
        self.assertTrue(same_model(m1, m2))
        self.assertTrue(torch.equal(d1["learner"]["scorer"]["ridge"]["A"], d2["learner"]["scorer"]["ridge"]["A"]))
        la, lb = labels_of(exp), labels_of(self.exp["daisy"])
        for k in lb:
            np.testing.assert_array_equal(la[k], lb[k])
        run_main(flags, record=rec)                                              # a job of a finished run does nothing
        self.assertEqual(len(rec), 64)


if __name__ == "__main__":
    unittest.main()
