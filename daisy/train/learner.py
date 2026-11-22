"""The online learner of a DAiSY run (--daisy_learn): labels the trained documents with the label gates, gathers them
to rank 0, fits the ridge scorer on them and hands its weights to the selector (paper Algorithm 1).
The trainer (daisy/train/trainer.py) calls it at fixed points of the training loop:

  on_refresh(step)                     every R steps (all ranks): the combined target gradient (daisy/labels/targets.py)
  before_step(step)                    before the step's micro-batches: the scorer starts at t0 (rank 0)
  label_gates(step, shape, device)     per micro-batch: the label gates on label steps (every --daisy_label_every-th
                                       step and every refresh step), else None
  collect_micro(...)                   after each labelled micro-batch's backward: per-document phi, keys, labels
  after_step(step)                     after the optimizer step: gather to rank 0, labels -> scorer
                                       (daisy/scorers/online.py), label dumps (daisy/labels/dump.py), metric records
  selector_weights(step)               every U steps (all ranks): the scorer's [w; b], broadcast from rank 0
  state_dict() / load_state_dict()     the scorer's state (rank 0), for exact resume

Labels are per-token utilities y(x) = <P_t ⊙ G, grad mean_x l> of the combined target (raw gate values x N_micro * acc /
len(x)).
"""

import os
import time

import torch
import torch.distributed as dist

from daisy.data.select import PHI_DIM
from daisy.labels.dump import LabelDump
from daisy.labels.targets import SelectionTargets
from daisy.scorers.online import OnlineScorer


def gather_rows(t, world, dst=0):
    """Gather a tensor with a variable first dimension from every rank to rank `dst` (concatenated in rank order).
    Returns the concatenation on dst, None elsewhere."""
    if world == 1:
        return t
    n = torch.tensor([t.shape[0]], device=t.device)
    sizes = [torch.zeros_like(n) for _ in range(world)]
    dist.all_gather(sizes, n)
    sizes = [int(s) for s in sizes]
    mx = max(sizes)
    pad = torch.zeros((mx - t.shape[0],) + tuple(t.shape[1:]), dtype=t.dtype, device=t.device)
    tp = torch.cat([t, pad]) if mx > t.shape[0] else t
    if dist.get_rank() == dst:
        bufs = [torch.empty_like(tp) for _ in range(world)]
        dist.gather(tp.contiguous(), bufs, dst=dst)
        return torch.cat([b[:s] for b, s in zip(bufs, sizes)])
    dist.gather(tp.contiguous(), None, dst=dst)
    return None


class OnlineLearner:
    def __init__(self, cfg, raw_model, opt, type_ctx, world, rank, master, out_dir, device, emit, log=print):
        self.cfg, self.world, self.rank, self.master, self.dev = cfg, world, rank, master, device
        self.out_dir, self.log, self.emit = out_dir, log, emit
        self.R, self.start = cfg.daisy_refresh, cfg.daisy_label_start
        self.U, self.every = cfg.daisy_scorer_update_every, cfg.daisy_metrics_every
        self.label_every = int(cfg.daisy_label_every)
        if master:   # printed so that a truncated flag list is visible at once
            log(f"[learner] labels from step {self.start}, the scorer starts at {cfg.daisy_scorer_start}, refresh "
                f"R={self.R}, scorer update every U={self.U}, labels every {self.label_every} steps (combined target)")
        if cfg.daisy_scorer_start <= self.start:
            raise ValueError(f"--daisy_scorer_start {cfg.daisy_scorer_start} must come after the first labels "
                             f"(--daisy_label_start {self.start})")
        tasks = [t for t in str(cfg.daisy_targets).split(",") if t.strip()]
        if not tasks:
            raise ValueError("--daisy_learn needs target tasks (--daisy_targets)")
        # --daisy_no_task_labels: no per-task references on refresh steps (no per-task labels and "tasks" records);
        # the combined target, the labels the scorer learns from and the selection are unchanged
        self.no_task_labels = bool(cfg.daisy_no_task_labels)
        self.targets = SelectionTargets(raw_model, opt, {t: os.path.join(cfg.daisy_target_dir, t) for t in tasks},
                                        cfg.sequence_length, cfg.daisy_target_tokens, type_ctx, world, rank, log=log)
        self.select = float(cfg.daisy_select_mult) > 0
        self.refs_ready, self.task_step = False, None
        self.scorer = self.label_dump = None
        if master:
            self.scorer = OnlineScorer(PHI_DIM, cfg.daisy_scorer_start, self.U, cfg.daisy_ridge_reg,
                                       cfg.daisy_ridge_decay, device, log=log)
            self.label_dump = LabelDump(os.path.join(out_dir, "labels"), tasks)
        self._reset_step()

    # ------------------------------------------------------------------ helpers
    def _reset_step(self):
        """Per-step buffers, filled micro-batch by micro-batch (this rank's documents with a loss term)."""
        self.step_phi, self.step_ntgt, self.step_keys, self.step_y = [], [], [], []

    def is_label_step(self, step):
        return self.refs_ready and step >= self.start and (step % self.label_every == 0 or step == self.task_step)

    # ------------------------------------------------------------------ hooks
    def on_refresh(self, step):
        """Every R steps (all ranks), from the label start on: the combined target gradient ("targets" record: the
        loss of every target task)."""
        if step >= self.start:
            t0 = time.time()
            target_loss = self.targets.refresh_combined(keep_tasks=not self.no_task_labels)
            self.refs_ready, self.task_step = True, step
            self.emit(step, "targets", {**{f"loss/{t}": v for t, v in target_loss.items()},
                                        "seconds": time.time() - t0})

    def before_step(self, step):
        if self.master and step >= self.start:
            self.scorer.before_step(step)

    def label_gates(self, step, shape, device):
        """The label gates for one micro-batch on label steps, None otherwise."""
        return self.targets.label_part(shape, device) if self.is_label_step(step) else None

    def collect_micro(self, phi, n_tgt, keys, scale, lab_part, seg, n_seg):
        """After a labelled micro-batch's backward. phi (n_seg, d), n_tgt (n_seg,) loss terms per document, keys
        (n_seg,), scale (n_seg,) = N_micro * acc / n_tgt (raw gate values -> per-token units)."""
        live = n_tgt > 0
        self.step_phi.append(phi[live].float())
        self.step_ntgt.append(n_tgt[live])
        self.step_keys.append(keys[live])
        y = self.targets.collect(lab_part, seg, n_seg) * scale[None, None, :]      # (targets, n_layers, n_seg)
        self.step_y.append(y.sum(1)[:, live])

    def after_step(self, step):
        """`step` = the step that was just trained. On label steps: gather the step's documents to rank 0, evaluate
        the scorer on them, add them to the ridge, dump the labels; every U steps refit; log metrics."""
        if step < self.start:
            self._reset_step()
            return
        labelled = bool(self.step_y)
        phi = ntgt = keys = y = None
        if labelled:
            phi = gather_rows(torch.cat(self.step_phi), self.world)
            ntgt = gather_rows(torch.cat(self.step_ntgt), self.world)
            keys = gather_rows(torch.cat(self.step_keys), self.world)
            y = gather_rows(torch.cat(self.step_y, 1).T.contiguous(), self.world)    # (n, targets)
        if self.master:
            if labelled:
                self._learn_from_labels(step, phi, ntgt, keys, y)
            self.scorer.maybe_update(step)
            if self.scorer.active and step % self.every == 0:
                self.emit(step, "scorer", self.scorer.pop_metrics())
        if step == self.task_step:                                  # all ranks: keep only the combined reference
            self.targets.keep_combined()
        self._reset_step()

    def _learn_from_labels(self, step, phi, ntgt, keys, y):
        """Rank 0, a labelled step: the scorer's evaluation and update ("tasks" record: per-task agreement on refresh
        steps), the label dump."""
        names = self.targets.tl.targets
        yc = y[:, names.index("combined")]
        tasks = {t: y[:, names.index(t)] for t in self.targets.targets if t in names} if step == self.task_step else None
        res = self.scorer.on_labels(step, phi, yc, tasks)
        if res:
            self.emit(step, "tasks", res)
        self.label_dump.add(step, step // self.R, keys.cpu().numpy(), ntgt.cpu().numpy(), yc.cpu().numpy(),
                            {t: v.cpu().numpy() for t, v in tasks.items()} if tasks else None)
        self.label_dump.maybe_flush(step)

    def selector_weights(self, step):
        """All ranks: on update steps the scorer's weights [w; b] on phi from rank 0 (None otherwise, or before the
        first fit)."""
        if not self.select or step % self.U != 0:
            return None
        buf = torch.full((PHI_DIM + 1,), float("nan"), device=self.dev)
        if self.master:
            w = self.scorer.weights()
            if w is not None:
                buf.copy_(w)
        if self.world > 1:
            dist.broadcast(buf, 0)
        return None if torch.isnan(buf).any() else buf.cpu().numpy()

    # ------------------------------------------------------------------ checkpoint
    def state_dict(self):
        return {"scorer": self.scorer.state_dict()} if self.master else {}

    def load_state_dict(self, st):
        if self.master and st.get("scorer") is not None:
            self.scorer.load_state_dict(st["scorer"])

    def flush_dumps(self, step):
        """Before a checkpoint (rank 0): write what the label dump holds, so a resume neither loses nor repeats it."""
        if self.label_dump is not None:
            self.label_dump.maybe_flush(step, force=True)

    def close(self, step):
        self.flush_dumps(step)
