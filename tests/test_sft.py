"""Instruction tuning (daisy/sft): chat rendering and loss mask, packing, batches, and an end-to-end CPU run of
sft_train.py whose output loads like a pretraining run.

  * render: assistant content + its EOT are targets; role markers, system/user turns and the newline after EOT are not;
  * pack: every trainable conversation exactly once, whole, in order, rows <= seq_len + 1 tokens;
  * batch: y is the next token exactly where that token is an assistant token of the same conversation, else -1;
  * trainable: a conversation whose assistant tokens all fall beyond one row is dropped;
  * sft_train: the step loss is the mean over the step's target tokens (micro-step split does not change the update),
    and the checkpoint + meta load like those of a pretraining run (sft_train.load_base);
  * sft_train over several epochs: every epoch end saves ckpts/<step>; with --schedule wsd the checkpoint of an inner
    epoch k (decayed on a copy) equals a run stopped at k epochs, so the main run is restored after each copy; a run
    stopped with --stop_at and restarted ends with the same weights as an uninterrupted one.

    cd tests && python -m unittest test_sft
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from daisy.sft.data import EOT, ChatSet, render  # noqa: E402
from daisy.sft.sft_train import load_base, main as sft_main  # noqa: E402


def enc(text):
    """Toy encoder: one id per character (ids < 1000, never EOT)."""
    return [ord(c) % 997 for c in text]


def write_set(root, split, convs):
    toks, masks = zip(*[render(c, enc) for c in convs])
    lens = [len(t) for t in toks]
    np.concatenate(toks).astype(np.uint16).tofile(os.path.join(root, f"{split}.tok"))
    np.concatenate(masks).astype(np.uint8).tofile(os.path.join(root, f"{split}.mask"))
    np.concatenate([[0], np.cumsum(lens)]).astype(np.int64).tofile(os.path.join(root, f"{split}.idx"))
    return toks, masks


def random_convs(rng, n, long_prompt_every=0):
    convs = []
    for i in range(n):
        msgs = []
        if rng.random() < 0.3:
            msgs.append({"role": "system", "content": "be brief"})
        for _ in range(int(rng.integers(1, 3))):
            plen = 200 if long_prompt_every and i % long_prompt_every == 0 else int(rng.integers(1, 12))
            msgs.append({"role": "user", "content": "q" * plen})
            msgs.append({"role": "assistant", "content": "a" * int(rng.integers(1, 15))})
        convs.append(msgs)
    return convs


class RenderTest(unittest.TestCase):
    def test_mask(self):
        msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo<|endoftext|>"}, {"role": "user", "content": "x"},
                {"role": "assistant", "content": "ok"}]
        t, m = render(msgs, enc)
        text = "<|system|>\nS\n<|user|>\nhi\n<|assistant|>\n"
        p = len(text)
        self.assertEqual(list(t[:p]), enc(text))
        self.assertFalse(m[:p].any())
        a1 = enc("yo<endoftext>")                                   # the literal EOT string is rewritten
        self.assertEqual(list(t[p:p + len(a1)]), a1)
        self.assertTrue(m[p:p + len(a1)].all())
        self.assertEqual(int(t[p + len(a1)]), EOT)
        self.assertEqual(int(m[p + len(a1)]), 1)
        q = p + len(a1) + 1
        rest = "\n<|user|>\nx\n<|assistant|>\n"
        self.assertEqual(list(t[q:q + len(rest)]), enc(rest))
        self.assertFalse(m[q:q + len(rest)].any())
        self.assertEqual(list(t[-3:]), enc("ok") + [EOT])            # no newline after the last turn
        self.assertTrue(m[-3:].all())
        self.assertEqual(int(m.sum()), len(a1) + 1 + 3)
        with self.assertRaises(ValueError):
            render([{"role": "tool", "content": "x"}], enc)


class PackTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.T = 63
        self.convs = random_convs(np.random.default_rng(0), 120, long_prompt_every=17)
        self.toks, self.masks = write_set(self.root, "train", self.convs)
        self.ds = ChatSet(self.root, "train")

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_trainable_pack_batch(self):
        T, W = self.T, self.T + 1
        keep = self.ds.trainable(T)
        for c, (t, m) in enumerate(zip(self.toks, self.masks)):
            self.assertEqual(bool(keep[c]), bool(m[1:min(len(t), W)].any()))
        self.assertGreater(int((~keep).sum()), 0)                     # the long prompts lose every target
        order = np.random.default_rng(1).permutation(np.flatnonzero(keep))
        rows = self.ds.pack(T, order)
        self.assertEqual([c for r in rows for c, _ in r], list(order))  # each once, in order, never split
        for r in rows:
            self.assertLessEqual(sum(n for _, n in r), W)
            for c, n in r:
                self.assertEqual(n, min(len(self.toks[c]), W))
        x, y, d = self.ds.batch(rows, T)
        self.assertEqual(x.shape, (len(rows), T))
        for i, r in enumerate(rows):
            p, want_y, want_x = 0, np.full(W, -1), np.full(W, EOT)
            for c, n in r:
                t, m = self.toks[c][:n].astype(np.int64), self.masks[c][:n]
                want_x[p:p + n] = t
                tgt = np.where(m[1:] == 1, t[1:], -1)
                want_y[p:p + n - 1] = tgt
                p += n
            np.testing.assert_array_equal(x[i], want_x[:-1])
            np.testing.assert_array_equal(y[i], want_y[:-1])
            self.assertTrue((y[i] >= 0).any())
            self.assertEqual(int(d[i, -1]), len(r) if p < W - 1 else len(r) - 1)   # padding is its own document


def make_base(root):
    """A tiny Llama checkpoint (iter 77) and a toy chat set under root -> (base dir, data dir)."""
    from daisy import config
    from daisy.models.llama import Llama
    from daisy.train.checkpoint import save_checkpoint

    cfg = config.parse_args([
        "--experiment_name", "base", "--daisy_index_dir", "unused", "--n_embd", "32", "--n_head", "2", "--n_layer", "2",
        "--sequence_length", "63", "--vocab_size", "50304", "--device", "cpu", "--dtype", "float32"])
    base, data = os.path.join(root, "base"), os.path.join(root, "data")
    os.makedirs(os.path.join(base, "daisy"))
    os.makedirs(data)
    with open(os.path.join(base, "daisy", "meta_train.json"), "w") as f:
        json.dump({"config": vars(cfg)}, f)
    torch.manual_seed(0)
    net = Llama(cfg)
    save_checkpoint(net, torch.optim.AdamW(net.parameters()), None, 77, os.path.join(base, "ckpts", "latest"))
    rng = np.random.default_rng(0)
    write_set(data, "train", random_convs(rng, 200))
    write_set(data, "val", random_convs(rng, 20))
    return base, data


def weights(out, ckpt):
    return torch.load(os.path.join(out, "ckpts", str(ckpt), "main.pt"), weights_only=False)["model"]


class SftTrainTest(unittest.TestCase):
    def assert_same(self, a, b, tol=1e-6):
        for k in a:
            torch.testing.assert_close(a[k], b[k], rtol=tol, atol=tol, msg=k)

    def test_epochs_wsd_and_resume(self):
        root = tempfile.mkdtemp()
        try:
            base, data = make_base(root)
            common = ["--base", base, "--data", data, "--rows", "4", "--acc", "1", "--lr", "3e-3", "--log_every", "1",
                      "--eval_every", "1000", "--warmup_frac", "0.001", "--epochs", "3", "--cooldown_frac", "0.25"]

            def sft(name, *extra):
                out = os.path.join(root, name)
                sft_main(common + ["--out", out] + list(extra))
                return out

            full = sft("wsd", "--schedule", "wsd", "--save_epochs")
            meta = json.load(open(os.path.join(full, "daisy", "meta_train.json")))["sft"]
            E, total = meta["steps_per_epoch"], meta["steps"]
            self.assertEqual(total, 3 * E)
            self.assertGreaterEqual(E, 8)
            for k in (1, 2, 3):
                net, _, itr = load_base(full, k * E)            # loads like any run, at the epoch's step
                self.assertEqual(itr, k * E)
            self.assert_same(weights(full, 3 * E), weights(full, "latest"))
            # an inner epoch's checkpoint = a run that stops (and decays) there; k = 2 also needs the restore after k = 1
            for k in (1, 2):
                short = sft(f"wsd_stop{k}", "--schedule", "wsd", "--max_steps", str(k * E))
                self.assert_same(weights(full, k * E), weights(short, "latest"))
            # stopped after the first cooldown copy, restarted: the same checkpoints and final weights
            part = sft("wsd_resume", "--schedule", "wsd", "--save_epochs", "--save_every", "3", "--stop_at", str(E + 2))
            self.assertFalse(os.path.exists(os.path.join(part, "sft", "finished.json")))
            self.assertTrue(os.path.exists(os.path.join(part, "ckpts", "resume", "state.pt")))
            sft("wsd_resume", "--schedule", "wsd", "--save_epochs", "--save_every", "3")
            self.assertFalse(os.path.exists(os.path.join(part, "ckpts", "resume", "state.pt")))
            for c in (E, 2 * E, "latest"):
                self.assert_same(weights(full, c), weights(part, c))
            recs = [json.loads(line) for line in open(os.path.join(part, "sft", "metrics.jsonl"))]
            self.assertTrue(any(r.get("resumed") for r in recs))
            # linear: one decay over the 3 epochs, a checkpoint at every epoch end; epochs 2 and 3 revisit every row
            lin = sft("linear", "--save_epochs")
            for k in (1, 2, 3):
                self.assertTrue(os.path.exists(os.path.join(lin, "ckpts", str(k * E), "main.pt")))
            recs = [json.loads(line) for line in open(os.path.join(lin, "sft", "metrics.jsonl"))]
            lrs = [r["lr"] for r in recs if "lr" in r]
            self.assertTrue(all(x >= y for x, y in zip(lrs[1:], lrs[2:])))   # decreasing after the 1-step warmup
            self.assertTrue(all(np.isfinite(r["train_loss"]) for r in recs if "train_loss" in r))
        finally:
            shutil.rmtree(root)

    def test_end_to_end_and_token_mean(self):
        root = tempfile.mkdtemp()
        try:
            base, data = make_base(root)
            outs = {}
            for name, rows, acc in (("a", 8, 1), ("b", 4, 2)):        # same global step, split differently
                out = os.path.join(root, name)
                sft_main(["--base", base, "--data", data, "--out", out, "--rows", str(rows), "--acc", str(acc),
                          "--max_steps", "3", "--lr", "1e-3", "--log_every", "1", "--eval_every", "2"])
                outs[name] = out
            sd = {n: torch.load(os.path.join(outs[n], "ckpts", "latest", "main.pt"), weights_only=False)["model"]
                  for n in outs}
            # the two splits differ only by rounding. Adam turns rounding in near-zero gradients into steps of size lr,
            # so single weights can differ by ~lr; the update as a whole and the validation losses must agree
            sd0 = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in
                   torch.load(os.path.join(base, "ckpts", "latest", "main.pt"), weights_only=False)["model"].items()}
            diff = sum(float((sd["a"][k].float() - sd["b"][k].float()).pow(2).sum()) for k in sd["a"]) ** 0.5
            upd = sum(float((sd["a"][k].float() - sd0[k].float()).pow(2).sum()) for k in sd["a"]) ** 0.5
            self.assertGreater(upd, 0.0)
            self.assertLess(diff, 0.05 * upd, f"micro-step split changed the update: |a - b| {diff} vs |a - base| {upd}")
            va, vb = ([json.loads(line)["val_loss"] for line in open(os.path.join(outs[n], "sft", "metrics.jsonl"))
                       if "val_loss" in line] for n in ("a", "b"))
            np.testing.assert_allclose(va, vb, rtol=1e-6)
            recs = [json.loads(line) for line in open(os.path.join(outs["a"], "sft", "metrics.jsonl"))]
            self.assertEqual(recs[0]["step"], 0)
            self.assertIn("val_loss", recs[0])
            self.assertTrue(recs[-1]["final"])
            meta = json.load(open(os.path.join(outs["a"], "daisy", "meta_train.json")))
            self.assertEqual(meta["sft"]["base_itr"], 77)
            self.assertEqual(meta["sft"]["steps"], 3)
            # the output loads like any pretraining run (daisy/eval/lm_eval_run.py load_run reads it the same way)
            net2, _, itr = load_base(outs["a"], "latest")
            self.assertEqual(itr, 3)
            for (n, p) in net2.named_parameters():
                torch.testing.assert_close(p, sd["a"][n])
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
