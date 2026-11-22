"""The labels of the label gates (daisy/labels/gates.py: TargetLabels) against exact per-document autograd, on a tiny
packed Llama with document-masked attention and a real AdamW state.

Checks:
  1. the gates' labels, summed over layers, == <P ⊙ G_V, grad of each document's share of the training loss>, per
     document (fp32), with gradient-accumulation scaling; the documents' labels sum to the full-batch inner product;
  2. the gates never change training outputs or gradients (bit-identical with and without them);
  3. reference = Adam's preconditioner 1/(sqrt(v_hat)+eps) ⊙ G_V; validation_pass G_V and topic losses;
  4. under bf16 autocast the labels still track exact per-document autograd (correlation);
  5. torch.compile (aot_eager) gives the same labels as eager.

    cd tests && python -m unittest test_label_gates
"""

import contextlib
import os
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from daisy import config  # noqa: E402
from daisy.models.llama import Llama  # noqa: E402
from daisy.data.docstore import pack  # noqa: E402
from daisy.labels.gates import TargetLabels, tracked_linears, validation_pass  # noqa: E402

V, T = 128, 128


class ListStore:
    def __init__(self, docs):
        self.d = docs

    def doc(self, i, cap=None):
        return self.d[i]


def tiny_model():
    args = config.parse_args(["--n_embd", "64", "--n_head", "4", "--n_layer", "2", "--sequence_length", str(T),
                              "--vocab_size", str(V), "--dropout", "0.0", "--experiment_name", "test",
                              "--daisy_index_dir", ""])
    torch.manual_seed(0)
    return Llama(args)


def packed_batch(rng, n_docs=14, rows=3):
    while True:
        docs = [np.append(rng.integers(1, V, size=int(rng.integers(10, 60))), 0) for _ in range(n_docs)]
        p = pack(ListStore(docs), list(range(n_docs)), T)
        if p["n_rows"] >= rows:
            x, y, doc = (torch.from_numpy(p[k][:rows]).long() for k in ("x", "y", "doc"))
            used = int(doc.max()) + 1
            return x, y, doc, used


def trained_model_and_opt(steps=3):
    model = tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.8, 0.999), eps=1e-8, weight_decay=0.1)
    g = torch.Generator().manual_seed(1)
    for _ in range(steps):
        x = torch.randint(0, V, (2, T), generator=g)
        model(x, targets=x)["loss"].backward()
        opt.step()
        opt.zero_grad()
    return model, opt


def exact_labels(model, ref, x, y, doc, n_docs, acc=1, ctx=contextlib.nullcontext(), group=None):
    """Per-group <ref, grad of the group's share of the training loss>; groups default to documents."""
    layers = tracked_linears(model)
    W = [m.weight for m in layers.values()]
    with ctx:
        logits = model(x, targets=y, get_logits=True, doc_ids=doc)["logits"]
    tl = F.cross_entropy(logits.float().reshape(-1, V), y.reshape(-1), ignore_index=-1, reduction="none")
    N = (y >= 0).sum()
    ids = (doc if group is None else group).reshape(-1)
    out = torch.zeros(n_docs, dtype=torch.float64)
    for i in range(n_docs):
        gi = torch.autograd.grad(tl[ids == i].sum() / (N * acc), W, retain_graph=True, allow_unused=True)
        out[i] = sum((ref[n].double() * g.double()).sum() for n, g in zip(layers, gi) if g is not None)
    return out


def labeled_step(model, tl, x, y, doc, n_docs, acc=1, ctx=contextlib.nullcontext(), fwd=None):
    """One labelled backward; -> the first target's labels summed over the tracked layers (n_docs,)."""
    part = tl.make(x.shape, x.device)
    with ctx:
        loss = (fwd or model)(x, targets=y, doc_ids=doc, gates=part)["loss"]
    (loss / acc).backward()
    return tl.collect(part, doc, n_docs)[0].sum(0)


class LabelGateTest(unittest.TestCase):
    def setUp(self):
        self.model, self.opt = trained_model_and_opt()
        self.rng = np.random.default_rng(0)
        self.layers = tracked_linears(self.model)
        vx, vy, vdoc, _ = packed_batch(np.random.default_rng(5), rows=4)
        self.vbatches = [{"x": vx[:2], "y": vy[:2], "doc": vdoc[:2], "topic": (vdoc[:2] % 3)},
                         {"x": vx[2:], "y": vy[2:], "doc": vdoc[2:], "topic": (vdoc[2:] % 3)}]
        self.G_V, self.ell, self.cnt = validation_pass(self.model, self.layers, self.vbatches, 3,
                                                       contextlib.nullcontext())

    def gates(self):
        tl = TargetLabels(self.model, self.opt)
        tl.set_targets({"t": self.G_V})
        return tl, {n: tl.refs[n][0] for n in tl.names}

    def test_tracked_layers(self):
        self.assertEqual(len(self.layers), 2 * 5)  # per block: attn c_attn, c_proj; mlp w1, w2, c_proj
        self.assertTrue(all("lm_head" not in n for n in self.layers))

    def test_validation_pass(self):
        W = [m.weight for m in self.layers.values()]
        tl_all, top_all = [], []
        for b in self.vbatches:
            lg = self.model(b["x"], targets=b["y"], get_logits=True, doc_ids=b["doc"])["logits"]
            tl = F.cross_entropy(lg.reshape(-1, V), b["y"].reshape(-1), ignore_index=-1, reduction="none")
            keep = b["y"].reshape(-1) >= 0
            tl_all.append(tl[keep])
            top_all.append(b["topic"].reshape(-1)[keep])
        tl_all, top_all = torch.cat(tl_all), torch.cat(top_all)
        g = torch.autograd.grad(tl_all.mean(), W)
        for n, gi in zip(self.layers, g):
            torch.testing.assert_close(self.G_V[n], gi, atol=1e-6, rtol=1e-4)
        for t in range(3):
            torch.testing.assert_close(self.ell[t], tl_all[top_all == t].mean().detach(), atol=1e-5, rtol=1e-5)

    def test_reference_is_adam_preconditioned(self):
        _, refs = self.gates()
        for n, m in self.layers.items():
            st = self.opt.state[m.weight]
            step = int(st["step"])
            vhat = st["exp_avg_sq"] / (1 - 0.999 ** step)
            torch.testing.assert_close(refs[n], self.G_V[n] / (vhat.sqrt() + 1e-8), atol=0, rtol=1e-6)

    def test_labels_exact_and_grads_untouched(self):
        x, y, doc, n_docs = packed_batch(self.rng)
        self.model.zero_grad()
        out1 = self.model(x, targets=y, doc_ids=doc, get_logits=True)
        (out1["loss"] / 2).backward()
        g_plain = [p.grad.clone() for p in self.model.parameters()]
        tl, refs = self.gates()
        self.model.zero_grad()
        part = tl.make(x.shape, x.device)
        out2 = self.model(x, targets=y, doc_ids=doc, get_logits=True, gates=part)
        (out2["loss"] / 2).backward()                                     # acc_steps = 2
        self.assertTrue(torch.equal(out1["logits"], out2["logits"]))     # the gate is the identity in the forward
        for a, b in zip(g_plain, self.model.parameters()):
            self.assertTrue(torch.equal(a, b.grad))
        raw = tl.collect(part, doc, n_docs)[0].sum(0)
        ex = exact_labels(self.model, refs, x, y, doc, n_docs, acc=2)
        scale = ex.abs().max()
        err = (raw.double() - ex).abs().max() / scale
        self.assertLess(err, 1e-4)
        # the documents' labels sum to the full-batch inner product
        full = exact_labels(self.model, refs, x, y, doc, 1, acc=2, group=torch.zeros_like(doc))
        torch.testing.assert_close(raw.double().sum(), full[0], atol=1e-4 * scale.item(), rtol=1e-4)

    def test_bf16_autocast_tracks_exact(self):
        x, y, doc, n_docs = packed_batch(np.random.default_rng(3), n_docs=24, rows=4)
        tl, refs = self.gates()
        ctx = torch.autocast("cpu", dtype=torch.bfloat16)
        self.model.zero_grad()
        raw = labeled_step(self.model, tl, x, y, doc, n_docs, ctx=ctx)
        ex = exact_labels(self.model, refs, x, y, doc, n_docs, ctx=ctx)
        c = np.corrcoef(raw.double().numpy(), ex.numpy())[0, 1]
        self.assertGreater(c, 0.99)


class LabelGateCompileTest(unittest.TestCase):
    """The gates work under torch.compile (they are part of the graph, not Python hooks)."""

    def test_compiled_labels_match_eager(self):
        model, opt = trained_model_and_opt()
        layers = tracked_linears(model)
        x, y, doc, n_docs = packed_batch(np.random.default_rng(7))
        vb = [{"x": x[:1], "y": y[:1], "doc": doc[:1], "topic": torch.zeros_like(doc[:1])}]
        G_V, _, _ = validation_pass(model, layers, vb, 1, contextlib.nullcontext())
        tl = TargetLabels(model, opt)
        tl.set_targets({"t": G_V})
        model.zero_grad()
        eager = labeled_step(model, tl, x, y, doc, n_docs)
        g_eager = [p.grad.clone() for p in model.parameters()]
        torch._dynamo.reset()
        cmodel = torch.compile(model, backend="aot_eager")
        for _ in range(2):
            model.zero_grad()
            comp = labeled_step(model, tl, x, y, doc, n_docs, fwd=cmodel)
            self.assertGreater(comp.abs().max().item(), 0)
            torch.testing.assert_close(comp, eager, atol=1e-5 * eager.abs().max().item(), rtol=1e-4)
        for a, p in zip(g_eager, model.parameters()):
            torch.testing.assert_close(p.grad, a, atol=1e-6, rtol=1e-5)


class CombinedOnlyTest(unittest.TestCase):
    """--daisy_no_task_labels: refresh_combined(keep_tasks=False) sums the task gradients as they come
    and keeps only the combined reference; it must be bitwise the combined reference of keep_tasks=True (same task
    order, same additions), with the same per-task losses."""

    def test_streaming_combined_is_bitwise_equal(self):
        import json
        import tempfile
        from daisy.labels.targets import SelectionTargets
        model, opt = trained_model_and_opt()
        rng = np.random.default_rng(5)
        d = tempfile.mkdtemp()
        dirs = {}
        for k in range(3):
            td = os.path.join(d, f"t{k}")
            os.makedirs(td)
            items = [np.append(rng.integers(1, V, size=int(rng.integers(20, 90))), 0).astype(np.uint16) for _ in range(12)]
            np.concatenate(items).tofile(os.path.join(td, "train.bin"))
            with open(os.path.join(td, "train.len"), "wb") as f:
                np.save(f, np.array([len(x) for x in items], dtype=np.int64))
            with open(os.path.join(td, "meta.json"), "w") as f:
                json.dump({"dtype": "uint16", "eos_id": 0}, f)
            dirs[f"t{k}"] = td
        bt = SelectionTargets(model, opt, dirs, T, 600, contextlib.nullcontext(), 1, 0, log=lambda *a: None)
        la = bt.refresh_combined(keep_tasks=True)
        ia = bt.tl.targets.index("combined")
        self.assertEqual(bt.tl.targets, ["t0", "t1", "t2", "combined"])
        ref_a = {n: bt.tl.refs[n][ia].clone() for n in bt.tl.names}
        lb = bt.refresh_combined(keep_tasks=False)
        self.assertEqual(bt.tl.targets, ["combined"])
        self.assertEqual(la, lb)
        for n in bt.tl.names:
            self.assertTrue(torch.equal(ref_a[n], bt.tl.refs[n][0]), n)
        self.assertGreater(sum(float(v.abs().sum()) for v in ref_a.values()), 0)


if __name__ == "__main__":
    unittest.main(verbosity=1)
