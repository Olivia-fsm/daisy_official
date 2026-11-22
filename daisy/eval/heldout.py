"""Held-out downstream loss during training: the mean token loss on a fixed seeded subset of each evaluation task's
held-out split (val.bin; for the benchmarks: ARC / PIQA / SciQ test, HellaSwag validation), every R steps, written as
"bench_val" records of metrics.jsonl (by the learner, or by the trainer in runs without a learner).
"""

import torch

from daisy.labels.gates import validation_pass
from daisy.labels.targets import target_batches


def eval_tasks(cfg):
    """The tasks of the held-out downstream loss: --daisy_eval_tasks, or the targets when it is empty (the benchmark
    target, whose tasks have a held-out split of their own)."""
    names = str(getattr(cfg, "daisy_eval_tasks", "") or cfg.daisy_targets)
    return [t for t in names.split(",") if t.strip()]


class EvalSet:
    """Held-out evaluation (the val splits; never used for labels): mean token loss per task on a fixed seeded subset,
    forward only."""

    def __init__(self, raw_model, layers, eval_dirs, seq_len, max_tokens, type_ctx, world, rank, log=print):
        self.raw, self.layers, self.type_ctx, self.world, self.rank = raw_model, layers, type_ctx, world, rank
        self.batches, info = {}, {}
        for task, d in eval_dirs.items():
            self.batches[task], info[task] = target_batches(d, seq_len, max_tokens, split="val", seed=1)
        log(f"[eval] held-out evaluation (val splits): {info}")

    @torch.no_grad()
    def losses(self):
        out = {}
        for task, b in self.batches.items():
            _, ell, _ = validation_pass(self.raw, self.layers, b, 1, self.type_ctx, world=self.world, rank=self.rank,
                                        compute_grad=False)
            out[task] = float(ell[0])
        out["mean"] = sum(out.values()) / len(out)
        return out
