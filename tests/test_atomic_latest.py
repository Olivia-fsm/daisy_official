"""DAISY_ATOMIC_LATEST=1 (daisy/train/checkpoint.py: swap_latest): a kill at any point of the checkpoint swap leaves a
whole ckpts/latest after scripts/restore_latest.sh, and a toy run with the variable set equals the run without it.

    cd tests && python -m unittest test_atomic_latest
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from helpers import ROOT, main_flags, make_run_data, run_main
from daisy.train.checkpoint import swap_latest

RESTORE = os.path.join(ROOT, "scripts", "restore_latest.sh")
DAISY = ["--daisy_learn", "--daisy_select_mult", "4"]


def make(d, text):
    d.mkdir(parents=True)
    (d / "main.pt").write_text(text)


class SwapTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.c = self.root / "ckpts"

    def tearDown(self):
        shutil.rmtree(self.root)

    def restored(self):
        subprocess.run(["bash", RESTORE, str(self.root)], check=True, capture_output=True)
        self.assertEqual(sorted(p.name for p in self.c.iterdir()), ["latest"])
        return (self.c / "latest" / "main.pt").read_text()

    def test_swap(self):
        make(self.c / "latest", "old")
        make(self.c / "latest.new", "new")
        swap_latest(self.c)
        self.assertEqual(self.restored(), "new")

    def test_first_checkpoint(self):
        make(self.c / "latest.new", "new")
        swap_latest(self.c)
        self.assertEqual(self.restored(), "new")

    def test_killed_during_the_write(self):
        make(self.c / "latest", "old")
        (self.c / "latest.new").mkdir()
        (self.c / "latest.new" / "main.pt").write_text("partial")
        self.assertEqual(self.restored(), "old")

    def test_killed_between_the_renames(self):
        make(self.c / "latest.old", "old")
        make(self.c / "latest.new", "new")
        self.assertEqual(self.restored(), "old")

    def test_killed_before_the_old_one_went(self):
        make(self.c / "latest.old", "old")
        make(self.c / "latest", "new")
        self.assertEqual(self.restored(), "new")

    def test_nothing_yet(self):
        subprocess.run(["bash", RESTORE, str(self.root)], check=True, capture_output=True)
        self.assertFalse(self.c.exists())


class RunTest(unittest.TestCase):
    def test_toy_run_equals_the_run_without_it(self):
        root = tempfile.mkdtemp(prefix="atomic_")
        try:
            paths, _ = make_run_data(root)
            rec, state = {}, {}
            for name, env in (("direct", None), ("atomic", {"DAISY_ATOMIC_LATEST": "1"})):
                rec[name] = []
                run_main(main_flags(paths, name, root, extra=DAISY), env=env, record=rec[name])
                ckpts = os.path.join(root, "exps", name, "ckpts")
                self.assertEqual(sorted(p for p in os.listdir(ckpts) if p.startswith("latest")), ["latest"])
                self.assertEqual(sorted(os.listdir(os.path.join(ckpts, "latest"))),
                                 ["daisy_r0.pt", "main.pt", "stream_r0.json", "worker_0.pt"])
                state[name] = torch.load(os.path.join(ckpts, "latest", "main.pt"), map_location="cpu",
                                         weights_only=False)
            a, b = rec["direct"], rec["atomic"]
            self.assertEqual(len(a), len(b))
            self.assertEqual([r[0] for r in a], [r[0] for r in b])            # the same losses ...
            for ra, rb in zip(a, b):                                          # ... on the same micro-batches
                for xa, xb in zip(ra[1:], rb[1:]):
                    np.testing.assert_array_equal(xa, xb)
            self.assertEqual(state["direct"]["itr"], state["atomic"]["itr"])
            for k, v in state["direct"]["model"].items():
                self.assertTrue(torch.equal(v, state["atomic"]["model"][k]), k)
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
