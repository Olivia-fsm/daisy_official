"""lm_eval_run.load_run picks the requested checkpoint (latest or a permanent step) with the run's own configuration.

    cd tests && python3 -m unittest test_lm_eval_run
"""

import json
import os
import shutil
import sys
import tempfile
import types
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    import lm_eval.api.model  # noqa: F401
except ImportError:  # lm_eval is an optional dependency; load_run does not use it
    for name in ("lm_eval", "lm_eval.api"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["lm_eval.api.model"] = types.SimpleNamespace(TemplateLM=object)
from daisy import config  # noqa: E402
from daisy.eval.lm_eval_run import load_run  # noqa: E402
from daisy.models.llama import Llama  # noqa: E402
from daisy.train.checkpoint import save_checkpoint  # noqa: E402


class LoadRunTest(unittest.TestCase):
    def test_permanent_checkpoint_vs_latest(self):
        root = tempfile.mkdtemp()
        try:
            cfg = config.parse_args([
                "--n_embd", "32", "--n_head", "2", "--n_layer", "2", "--sequence_length", "64", "--vocab_size", "50304",
                "--device", "cpu", "--dtype", "float32", "--experiment_name", "test", "--daisy_index_dir", "unused"])
            os.makedirs(os.path.join(root, "daisy"))
            with open(os.path.join(root, "daisy", "meta_train.json"), "w") as f:
                json.dump({"config": vars(cfg), "git": {"commit": "x"}}, f)
            nets = {}
            for itr, d in ((16, "16"), (32, "latest")):
                torch.manual_seed(itr)
                nets[d] = Llama(cfg)
                save_checkpoint(nets[d], torch.optim.AdamW(nets[d].parameters()), None, itr, os.path.join(root, "ckpts", d))
            for ck, itr in (("16", 16), ("latest", 32)):
                net, _, step, _ = load_run(root, "cpu", ck)
                self.assertEqual(step, itr)
                for (n, p), q in zip(net.named_parameters(), nets[ck].parameters()):
                    self.assertTrue(torch.equal(p, q), n)
            self.assertEqual(load_run(root, "cpu")[2], 32)               # default: ckpts/latest
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
