"""Hindsight labels: each trained document's first-order utility <P_t ⊙ G, grad l(x)>, read off the TRAINING backward
pass (paper Section 3.2, Eq. 3). No candidate passes through the model, and no document is scored and then discarded.

For a tracked nn.Linear with per-token input a_tok and output gradient delta_tok in the training backward,

    y_layer(x) = < P_t ⊙ G , grad_W L_batch |_x >  =  sum_{tok in x} delta_tok^T (P_t ⊙ G) a_tok

and y(x) is the sum over the tracked layers. The per-token terms come out of the backward as the gradient of a
per-token gate (daisy/models/label_gate.py, the identity in the forward) and are summed by document here (TargetLabels).
  * tracked layers: every nn.Linear of the transformer blocks (the vocabulary readout, which is tied to the token
    embedding, and the normalization weights are not tracked);
  * P_t: Adam's coordinatewise preconditioner 1 / (sqrt(v_hat) + eps), applied to the target gradient;
  * G: the gradient of a target loss with respect to each tracked weight, from `validation_pass`
    (daisy/labels/targets.py, every R steps).

Per-document sums are exact only with document-masked attention (daisy/models/doc_mask.py), and only for the loss as the
trainer computes it: the backward carries the gradient of (loss / acc_steps), so the raw labels are
<P ⊙ G, grad (sum_{tok in x} l_tok) / (N_micro * acc_steps)>; the learner rescales them to per-token utilities.
"""

from collections import OrderedDict

import torch
import torch.nn as nn


def tracked_linears(model):
    """All nn.Linear layers except vocabulary-sized readouts."""
    out = OrderedDict()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "lm_head" not in name and mod.out_features <= 32768:
            out[name] = mod
    return out


def attach_gate_names(model):
    """Give every tracked Linear its `gate_name` (what gated_linear keys on). Idempotent; returns the layers."""
    layers = tracked_linears(model)
    for n, m in layers.items():
        m.gate_name = n
    return layers


def adam_preconditioner(optimizer, weight):
    """P_t of one weight as a tensor of its shape: 1 / (sqrt(v_hat) + eps) from AdamW's second moment with bias
    correction, or None before the optimizer's first step (identity geometry)."""
    state = optimizer.state.get(weight, {})
    if "exp_avg_sq" not in state:
        return None
    step = state.get("step", 1)
    if isinstance(step, torch.Tensor):
        step = int(step.item())
    beta2, eps = 0.999, 1e-8
    for group in optimizer.param_groups:
        if any(p is weight for p in group["params"]):
            beta2 = group.get("betas", (0.9, 0.999))[1]
            eps = group.get("eps", 1e-8)
            break
    v_hat = state["exp_avg_sq"] / (1.0 - beta2 ** step if step > 0 else 1.0)
    return (1.0 / (v_hat.sqrt() + eps)).reshape(weight.shape)


class TargetLabels:
    """Utility labels against several targets at once, one gate per (target, layer).

        tl = TargetLabels(model, opt); tl.set_targets({"combined": G})            # references P_t ⊙ G
        part = tl.make(x.shape, x.device)                                         # per micro-batch
        model(x, targets=y, doc_ids=doc, gates=part)["loss"].backward()
        labels = tl.collect(part, doc, n_docs)    # (n_targets, n_layers, n_docs) raw labels

    collect() returns the labels per layer, so any subset of the layers can be summed afterwards."""

    def __init__(self, model, optimizer):
        self.layers = attach_gate_names(model)
        self.names = list(self.layers)
        self.optimizer = optimizer
        self.targets = []
        self.refs = {}

    @torch.no_grad()
    def set_targets(self, grads):
        """grads: {target: {layer: gradient of the target's mean loss}} -> preconditioned references P_t ⊙ grad."""
        self.targets = list(grads)
        self.refs = {}
        for n, m in self.layers.items():
            P = adam_preconditioner(self.optimizer, m.weight)
            self.refs[n] = [(grads[t][n].float() if P is None else P.float() * grads[t][n].float())
                            for t in self.targets]

    def keep(self, targets):
        """Restrict the references to a subset of the targets (the combined target between refreshes)."""
        idx = [self.targets.index(t) for t in targets]
        self.refs = {n: [self.refs[n][i] for i in idx] for n in self.refs}
        self.targets = list(targets)

    def make(self, shape, device):
        """The gates of one micro-batch of token shape `shape`: the `gates` argument of the model."""
        z = {n: [torch.zeros(shape, device=device, requires_grad=True) for _ in self.targets] for n in self.names}
        return {"z": z, "mats": {n: self.refs[n] for n in self.names}}

    @torch.no_grad()
    def collect(self, part, doc, n_docs):
        """After the backward: the gates' gradients summed by document. doc (B, T): document index per position."""
        ids = doc.reshape(-1).long()
        out = torch.zeros(len(self.targets), len(self.names), n_docs, device=ids.device)
        for li, n in enumerate(self.names):
            for ti, z in enumerate(part["z"][n]):
                if z.grad is not None:
                    out[ti, li].index_add_(0, ids, z.grad.reshape(-1).float())
        return out


def validation_pass(model, layers, batches, n_topics, type_ctx, world=1, rank=0, compute_grad=True,
                    out_dtype=torch.float32):
    """One forward (and backward if compute_grad) over fixed rows. Returns (G, ell, counts); G is None when
    compute_grad=False (only the loss vector is needed: the held-out losses):
    G[name] = gradient of the MEAN token loss with respect to each tracked Linear weight (fp32) (a target gradient);
    ell[t]  = mean token loss of topic t (held-out web text: the loss vector of refresh.jsonl), accumulated in float64
              and returned as out_dtype.
    batches: list of dicts with x, y (-1 = ignored), topic (B, T) long (topic of each target position), and optionally
    doc (B, T) for document-masked attention.
    Uses autograd.grad on the unwrapped model, so the training .grad buffers are untouched. The rows are split across
    the ranks (batches[rank::world]) and reduced with all_reduce."""
    import torch.distributed as dist
    import torch.nn.functional as F

    weights = [m.weight for m in layers.values()]
    mine = batches[rank::world]
    dev = weights[0].device
    n_local = sum(int((b["y"] >= 0).sum()) for b in mine)
    n_total = torch.tensor(float(n_local), device=dev)
    if world > 1:
        dist.all_reduce(n_total)
    G = [torch.zeros_like(w, dtype=torch.float32) for w in weights] if compute_grad else None
    tsum = torch.zeros(n_topics, dtype=torch.float64, device=dev)
    tcnt = torch.zeros(n_topics, dtype=torch.float64, device=dev)
    was_training = model.training
    model.eval()
    for b in mine:
        x, y, topic = b["x"].to(dev), b["y"].to(dev), b["topic"].to(dev)
        doc = b["doc"].to(dev) if "doc" in b else None
        with type_ctx, torch.set_grad_enabled(compute_grad):
            logits = model(x, targets=y, get_logits=True, doc_ids=doc)["logits"]
            tl = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-1,
                                 reduction="none")
        keep = y.reshape(-1) >= 0
        tp = topic.reshape(-1)[keep]
        tsum.index_add_(0, tp, tl[keep].double().detach())
        tcnt.index_add_(0, tp, torch.ones_like(tp, dtype=torch.float64))
        if compute_grad:
            gs = torch.autograd.grad(tl[keep].sum() / n_total, weights)
            for acc, g in zip(G, gs):
                acc += g.float()
    if world > 1:
        if compute_grad:
            for g in G:
                dist.all_reduce(g)
        dist.all_reduce(tsum)
        dist.all_reduce(tcnt)
    model.train(was_training)
    ell = (tsum / tcnt.clamp_min(1)).to(out_dtype)
    return ({n: g for n, g in zip(layers, G)} if compute_grad else None), ell, tcnt
