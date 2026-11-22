"""The selection target: a set of target tasks, and the combined target gradient that the label gates
(daisy/labels/gates.py) score every labelled document against.

A target task is a directory with train.bin / train.len (and val.bin / val.len for the held-out evaluation): the
concatenated token ids of the task's items and the length of each item. For the benchmark target, an item is
"question correct-answer" + EOS from the benchmark's TRAINING split (daisy/targets/prepare_benchmarks.py; the val split
is the benchmark's test or validation split and is used for evaluation only); for the web-text targets, an item is a
document (daisy/targets/build_fineweb_target.py, build_fwedu_target.py). Items may be padded with EOS (the .len file
then holds the padded length): we strip the trailing EOS padding, keep one EOS, take a fixed seeded subset of the items
up to `max_tokens`, and pack them densely into document-masked rows.
The target gradient of a task is the gradient of the mean token loss over these rows (gates.validation_pass with one
"topic"); the combined target is the mean over the tasks, preconditioned like the training labels (P_t ⊙ G). The
held-out evaluation (daisy/eval/heldout.py) packs the val splits in the same way (target_batches(..., split="val")).
"""

import os

import numpy as np
import torch

from daisy.data.stream import pack_segments
from daisy.labels.gates import TargetLabels, validation_pass

EOT = 50256
TASKS = ("arc_easy", "arc_challenge", "piqa", "sciq", "hellaswag")


def task_format(task_dir):
    """(token dtype, EOS id) of a task directory: its meta.json when present (e.g. the multilingual targets: uint32,
    </s> = 2, not padded), else GPT-2 tokens (uint16, EOS 50256)."""
    mp = os.path.join(task_dir, "meta.json")
    if os.path.exists(mp):
        import json

        with open(mp) as f:
            m = json.load(f)
        return np.dtype(m.get("dtype", "uint16")), int(m.get("eos_id", EOT))
    return np.dtype(np.uint16), EOT


def load_items(task_dir, split="train"):
    """-> list of token arrays (the task's dtype), one per item: the item's tokens + one EOS (padding stripped).
    split: "train" (the label targets) or "val" (evaluation only). The .len file holds each item's (padded) length."""
    dt, eos = task_format(task_dir)
    toks = np.memmap(os.path.join(task_dir, f"{split}.bin"), dtype=dt, mode="r")
    lens = np.load(os.path.join(task_dir, f"{split}.len")).astype(np.int64)  # padded lengths
    offs = np.concatenate([[0], np.cumsum(lens)])
    if offs[-1] != len(toks):
        raise ValueError(f"{task_dir}: .len sums to {offs[-1]}, .bin has {len(toks)} tokens")
    items = []
    for s, e in zip(offs[:-1], offs[1:]):
        t = np.asarray(toks[s:e])
        nz = np.flatnonzero(t != eos)
        if len(nz) == 0:
            continue
        items.append(np.append(t[: nz[-1] + 1], eos).astype(dt))
    return items


def target_item_indices(task_dir, seq_len, max_tokens, seed=0, split="train"):
    """Indices (into load_items order) of the items packed into the target rows, in packing order (as
    _target_segments: the seeded permutation, up to the token budget, the tail cut to whole rows)."""
    items = load_items(task_dir, split)
    _, _, out = _target_segments(task_dir, seq_len, max_tokens, seed, split)
    order = np.random.default_rng(seed).permutation(len(items))
    return [int(order[r]) for r, _, _ in out]


class _ListStore:
    def __init__(self, docs):
        self.d = docs

    def doc(self, i, cap=None):
        return self.d[i]


def _target_segments(task_dir, seq_len, max_tokens, seed=0, split="train"):
    """The fixed seeded subset of items up to max_tokens (whole rows): (items, docs, segments (doc, start, end))."""
    items = load_items(task_dir, split)
    order = np.random.default_rng(seed).permutation(len(items))
    W = seq_len + 1
    budget = (max_tokens // W) * W
    segs, docs, have = [], [], 0
    for i in order:
        if have >= budget:
            break
        it = items[i]
        take = min(len(it), budget - have)
        docs.append(it)
        segs.append((len(docs) - 1, 0, take))
        have += take
    if have < W:
        raise ValueError(f"{task_dir}: not enough tokens for one row")
    keep = (have // W) * W
    # drop the tail so the segments fill whole rows
    out, acc = [], 0
    for r, s, e in segs:
        if acc >= keep:
            break
        e2 = min(e, s + keep - acc)
        out.append((r, s, e2))
        acc += e2 - s
    return items, docs, out


def target_items(task_dir, seq_len, max_tokens, seed=0, split="train"):
    """The token spans (EOS included where kept) of exactly the items packed into the target rows."""
    _, docs, out = _target_segments(task_dir, seq_len, max_tokens, seed, split)
    return [docs[r][s:e] for r, s, e in out]


def target_batches(task_dir, seq_len, max_tokens, rows_per_batch=4, seed=0, split="train"):
    """Fixed seeded subset of items up to max_tokens, packed into rows; returns (batches, info). Batches carry
    x, y (-1 across items), doc (item index per position) and topic = 0, ready for validation_pass(..., n_topics=1)."""
    items, docs, out = _target_segments(task_dir, seq_len, max_tokens, seed, split)
    x, y, seg = pack_segments(_ListStore(docs), out, seq_len, None)
    batches = []
    for i in range(0, len(x), rows_per_batch):
        sl = slice(i, i + rows_per_batch)
        batches.append({"x": torch.from_numpy(x[sl]), "y": torch.from_numpy(y[sl]),
                        "doc": torch.from_numpy(seg[sl]).long(), "topic": torch.zeros_like(torch.from_numpy(x[sl]))})
    info = {"items_total": len(items), "items_used": len(out), "rows": int(len(x)),
            "target_tokens": int((y >= 0).sum())}
    return batches, info


class SelectionTargets:
    """The label targets of a run (all ranks): every refresh, the gradient G_b of each target task's mean token loss and
    the COMBINED target mean_b G_b; their references P_t ⊙ G (TargetLabels) give the labels of every trained document
    on the label steps."""

    def __init__(self, raw_model, optimizer, target_dirs, seq_len, target_tokens, type_ctx, world, rank, log=print):
        self.raw, self.type_ctx, self.world, self.rank = raw_model, type_ctx, world, rank
        self.tl = TargetLabels(raw_model, optimizer)
        self.batches, self.info = {}, {}
        for task, d in target_dirs.items():
            self.batches[task], self.info[task] = target_batches(d, seq_len, target_tokens)
        log(f"[learner] label targets (train splits): {self.info}")

    def refresh_combined(self, keep_tasks=True):
        """G_b for every target task and the COMBINED target gradient mean_b G_b (the gradient of the average of the
        tasks' mean token losses; labels are linear in it, so its label is the mean of the per-task labels).
        References: the combined one, plus every task's on this step if keep_tasks (per-task labels on refresh steps;
        call keep_combined() after the step). Without keep_tasks the per-task gradients are summed as they come, so only
        one task gradient and the sum are held at a time (--daisy_no_task_labels: large models with many target tasks);
        the additions run in the same order, so the combined reference is bitwise the same.
        -> losses {task: mean token loss}."""
        grads, losses = {}, {}
        if not keep_tasks:
            acc, n_tasks = None, 0
            for task, b in self.batches.items():
                G, ell, _ = validation_pass(self.raw, self.tl.layers, b, 1, self.type_ctx, world=self.world,
                                            rank=self.rank, compute_grad=True)
                losses[task] = float(ell[0])
                acc = {n: 0 + g.float() for n, g in G.items()} if acc is None else {n: acc[n] + G[n].float() for n in acc}
                n_tasks += 1
                del G
            self.tl.set_targets({"combined": {n: v / n_tasks for n, v in acc.items()}})
            del acc
            return losses
        for task, b in self.batches.items():
            G, ell, _ = validation_pass(self.raw, self.tl.layers, b, 1, self.type_ctx, world=self.world, rank=self.rank,
                                        compute_grad=True)
            grads[task], losses[task] = G, float(ell[0])
        names = list(next(iter(grads.values())))
        comb = {n: sum(grads[t][n].float() for t in grads) / len(grads) for n in names}
        self.tl.set_targets({**grads, "combined": comb})
        del grads
        return losses

    def keep_combined(self):
        if "combined" in self.tl.targets:
            self.tl.keep(["combined"])

    def label_part(self, shape, device):
        return self.tl.make(shape, device)

    def collect(self, part, seg, n_seg):
        return self.tl.collect(part, seg, n_seg)  # (n_targets, n_tracked_layers, n_seg) raw labels

    @property
    def targets(self):
        return list(self.batches)
