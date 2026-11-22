"""Document-masked attention (daisy/models/doc_mask.py) and packing (daisy/data/docstore.py).

1. Doc masking is exact: packing two documents into one row gives the same per-token logits as running each
   document on its own from position 0 (RoPE is relative), and each document's loss gradient is unchanged.
2. pack(): rows, document ids, cross-document targets masked, nothing dropped except the row-boundary token.

    cd tests && python -m unittest test_doc_mask
"""

import os
import sys
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from daisy import config  # noqa: E402
from daisy.data.docstore import pack  # noqa: E402
from daisy.models.llama import Llama  # noqa: E402


def tiny_args(T=256):
    return config.parse_args(["--n_embd", "64", "--n_head", "4", "--n_layer", "2", "--sequence_length", str(T),
                              "--vocab_size", "128", "--dropout", "0.0", "--experiment_name", "test",
                              "--daisy_index_dir", "unused"])


class DocMaskTest(unittest.TestCase):
    def test_doc_mask_equals_separate_documents(self):
        args = tiny_args()
        torch.manual_seed(1)
        model = Llama(args)  # fp32: RoPE always computes q, k in fp32
        a = torch.randint(0, 128, (100,))
        b = torch.randint(0, 128, (156,))
        packed = torch.cat([a, b])[None]
        doc = torch.cat([torch.zeros(100, dtype=torch.long), torch.ones(156, dtype=torch.long)])[None]
        lp = model(packed, targets=packed, get_logits=True, doc_ids=doc)["logits"][0]
        la = model(a[None], targets=a[None], get_logits=True)["logits"][0]
        lb = model(b[None], targets=b[None], get_logits=True)["logits"][0]
        torch.testing.assert_close(lp[:100], la, atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(lp[100:], lb, atol=2e-5, rtol=1e-5)
        # without the mask, document b sees document a (the leakage the mask removes)
        lnomask = model(packed, targets=packed, get_logits=True)["logits"][0]
        self.assertGreater((lnomask[100:] - lb).abs().max().item(), 1e-3)
        # per-document loss gradients are unchanged by packing
        ya = torch.cat([a[1:], torch.tensor([-1])])
        yb = torch.cat([b[1:], torch.tensor([-1])])
        def grads(fn):
            model.zero_grad()
            fn().backward()
            return [p.grad.clone() for p in model.parameters()]
        g_sep = grads(lambda: torch.nn.functional.cross_entropy(
            model(b[None], targets=b[None], get_logits=True)["logits"][0], yb, ignore_index=-1, reduction="sum"))
        y_packed = torch.cat([torch.full((100,), -1), yb])  # only b's loss terms
        g_pack = grads(lambda: torch.nn.functional.cross_entropy(
            model(packed, targets=packed, get_logits=True, doc_ids=doc)["logits"][0], y_packed, ignore_index=-1,
            reduction="sum"))
        for g1, g2 in zip(g_sep, g_pack):
            torch.testing.assert_close(g1, g2, atol=1e-4, rtol=1e-4)

    def test_pack(self):
        class Store:
            def __init__(self, docs):
                self.d = docs
            def doc(self, i, cap=None):
                t = self.d[i]
                return np.concatenate([t[: cap - 1], t[-1:]]) if cap and len(t) > cap else t
        rng = np.random.default_rng(0)
        docs = [np.append(rng.integers(1, 100, size=n), 0).astype(np.uint16) for n in (5, 30, 3, 50, 12)]
        st = Store(docs)
        out = pack(st, [0, 1, 2, 3, 4], seq_len=15, max_doc_tokens=20)
        flat = np.concatenate([st.doc(i, 20) for i in range(5)])
        W = 16
        self.assertEqual(out["n_rows"], len(flat) // W)
        self.assertEqual(out["rest"], len(flat) - out["n_rows"] * W)
        np.testing.assert_array_equal(out["x"], flat[: out["n_rows"] * W].reshape(-1, W)[:, :-1])
        owner = np.repeat(np.arange(5), [len(st.doc(i, 20)) for i in range(5)])[: out["n_rows"] * W].reshape(-1, W)
        np.testing.assert_array_equal(out["doc"], owner[:, :-1])
        cross = owner[:, 1:] != owner[:, :-1]
        self.assertTrue((out["y"][cross] == -1).all())
        self.assertTrue((out["y"][~cross] == flat[: out["n_rows"] * W].reshape(-1, W)[:, 1:][~cross]).all())
        self.assertEqual(len(st.doc(3, 20)), 20)
        self.assertEqual(st.doc(3, 20)[-1], 0)


if __name__ == "__main__":
    unittest.main()
