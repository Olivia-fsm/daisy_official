"""Held-out losses of saved checkpoints (daisy/eval/heldout_ckpt.py): a short run with permanent checkpoints; the
post-hoc loss of each checkpoint on a benchmark's val split, with the run's --daisy_eval_tokens, equals the run's own
"bench_val" record of that step (same packing, model, dtype), and the whole-split evaluation covers the val split up to
its last whole row; a rerun skips what is done.

    cd tests && python3 -m unittest test_heldout_ckpt
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from helpers import T, cfg_for, make_benchmarks, make_index, run  # noqa: E402
from daisy.eval import heldout_ckpt  # noqa: E402
from daisy.labels.targets import load_items  # noqa: E402


def tool(*argv):
    """heldout_ckpt.main in-process; -> its stdout"""
    old, sys.argv = sys.argv, ["heldout_ckpt.py", *argv]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            heldout_ckpt.main()
    finally:
        sys.argv = old
    return buf.getvalue()


class HeldoutCkptTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        rng = np.random.default_rng(0)
        self.idx, self.bench = os.path.join(self.root, "idx"), os.path.join(self.root, "bench")
        os.makedirs(self.idx)
        make_index(self.idx, rng)
        make_benchmarks(self.bench, rng)

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_matches_bench_val_records(self):
        exps = os.path.join(self.root, "exps")
        exp = os.path.join(exps, "r_s0")
        run(cfg_for(self.idx, self.bench, 20, ["--permanent_ckpt_interval", "8", "--daisy_eval_dir", self.bench]), exp)
        ta = os.path.join(self.bench, "taska")
        out = os.path.join(self.root, "out")
        tool("--exps", exps, "--runs", "r_s0", "--tasks", f"whole={ta}", "--out", out, "--check_bench", f"taska={ta}")
        with open(os.path.join(out, "r_s0.json")) as f:
            res = json.load(f)
        self.assertEqual(sorted(res["whole"], key=int), ["0", "8", "16"])
        n_val = sum(len(t) - 1 for t in load_items(ta, "val"))   # every item's tokens are targets but its first
        for s in ("0", "8", "16"):
            c = res["check/taska"][s]
            self.assertIsNotNone(c["record"])
            self.assertAlmostEqual(c["loss"], c["record"], places=5)
            # whole split: every item but the tail that does not fill a last row of T + 1 tokens (whole rows only)
            self.assertGreaterEqual(res["whole"][s]["items"], len(load_items(ta, "val")) - 3)
            self.assertLessEqual(res["whole"][s]["tokens"], n_val)
            self.assertGreater(res["whole"][s]["tokens"], n_val - 2 * (T + 1))
        # a rerun skips what is done
        self.assertNotIn("@", tool("--exps", exps, "--runs", "r_s0", "--tasks", f"whole={ta}", "--out", out,
                                   "--check_bench", f"taska={ta}"))


class StepsOfTest(unittest.TestCase):
    """--min_age_s: checkpoints younger than it (or with no main.pt yet) are left for a later call."""

    def test_min_age(self):
        import time
        d = tempfile.mkdtemp()
        try:
            for s in (0, 16000, 32000, 48000):
                os.makedirs(os.path.join(d, "ckpts", str(s)))
            os.makedirs(os.path.join(d, "ckpts", "latest"))
            for s in (0, 16000, 32000):
                open(os.path.join(d, "ckpts", str(s), "main.pt"), "w").close()
            old = time.time() - 600
            for s in (0, 16000):
                os.utime(os.path.join(d, "ckpts", str(s), "main.pt"), (old, old))
            self.assertEqual(heldout_ckpt.steps_of(d, "all"), [0, 16000, 32000, 48000])
            self.assertEqual(heldout_ckpt.steps_of(d, "all", 300), [0, 16000])
            self.assertEqual(heldout_ckpt.steps_of(d, "16000,32000", 300), [16000])
        finally:
            shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main()
