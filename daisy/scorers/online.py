"""The online scorer of a DAiSY run (rank 0): the running ridge (daisy/scorers/ridge.py) learns to rank documents by
their utility labels from the frozen embedding phi, and its weights go to the selector.

Labels y(x) = <P_t ⊙ G, grad l(x)> per token (daisy/labels) arrive on every --daisy_label_every-th step. From the
scorer start t0 (--daisy_scorer_start) on, every labelled step adds its documents to the ridge with the Gaussianized
ranks of their labels as the regression target, and the ridge is refitted every U steps (--daisy_scorer_update_every).
Every labelled step is scored before the ridge learns from it (daisy/scorers/metrics.py, the "scorer" records).
"""

import collections

import torch

from daisy.scorers.metrics import spearman, step_metrics
from daisy.scorers.ridge import RunningRidge, rank_transform


class OnlineScorer:
    def __init__(self, d_emb, start, update_every, ridge_reg, ridge_decay, device, log=print):
        self.t0, self.U, self.dev, self.log = int(start), int(update_every), device, log
        self.ridge = RunningRidge(d_emb, device, lam=ridge_reg, decay=ridge_decay)
        self.active = False
        self.acc = collections.defaultdict(float)
        self.acc_n = collections.defaultdict(int)
        self.updates = 0

    def before_step(self, step):
        """Before the micro-batches of `step`: the scorer switches on at t0, so the ridge uses the labels of steps
        t >= t0 only."""
        if step == self.t0 and not self.active:
            self.active = True
            self.log(f"[scorer] the ridge starts at step {self.t0}")

    @torch.no_grad()
    def on_labels(self, step, phi, y, task_labels=None):
        """phi (n, d), y (n,): the combined-target labels of this step's documents. The ridge is evaluated on them
        before it learns from them; task_labels {task: (n,)} (refresh steps) only enter correlations (the "tasks"
        records). -> those correlations."""
        out = {}
        t = rank_transform(y)
        if self.active and len(y) > 3:
            if self.ridge.w is not None:
                S = self.ridge.score(phi)
                for k, v in step_metrics(S, y, t).items():
                    self._add(k, v)
                if task_labels:
                    for tk, yt in task_labels.items():
                        out[f"spearman/{tk}/ridge"] = spearman(S, yt)
            if task_labels:
                for tk, yt in task_labels.items():
                    out[f"spearman_vs_combined/{tk}"] = spearman(yt, y)       # how much the target tasks agree
        if self.active:
            self.ridge.add(phi, t)
        return out

    def maybe_update(self, step):
        """Every U steps once active: the closed-form refit."""
        if self.active and step % self.U == 0 and self.ridge.n > 0:
            self.ridge.solve()
            self.updates += 1

    def weights(self):
        """[w; b] (d + 1,) of the current fit, None before the first one."""
        return self.ridge.w

    # ------------------------------------------------------------------ metrics
    def _add(self, k, v):
        if v == v:   # not NaN
            self.acc[k] += v
            self.acc_n[k] += 1

    def pop_metrics(self):
        """Means of the per-step metrics since the last call (the "scorer" record)."""
        out = {k: v / self.acc_n[k] for k, v in self.acc.items() if self.acc_n[k]}
        out["n"] = self.acc_n.get("spearman", 0)
        out["updates"] = self.updates
        self.acc, self.acc_n = collections.defaultdict(float), collections.defaultdict(int)
        return out

    # ------------------------------------------------------------------ checkpoint
    def state_dict(self):
        return {"ridge": self.ridge.state_dict(), "active": self.active, "updates": self.updates}

    def load_state_dict(self, st):
        self.ridge.load_state_dict(st["ridge"])
        self.active, self.updates = st["active"], st["updates"]
