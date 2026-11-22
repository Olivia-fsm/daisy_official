"""Label dumps: the labels of every labelled document (EXP/daisy/labels/), for offline inspection of a run."""

import json
import os

import numpy as np

from daisy.runmeta import retry_io


class LabelDump:
    """Per labelled step, every trained document's labels (rank 0, the documents of all ranks).
    labels_{first_step:07d}.npz: step, key (pool key shard << 32 | row, or index row), n_targets (loss terms of the
    document in the step), refresh (step // R), y (combined target, per token, fp32) and y_tasks (n, T), the per-task
    labels on refresh steps (NaN elsewhere). tasks.json: the order of the tasks.
    Flushed every flush_steps steps and before every checkpoint (atomic rename)."""

    def __init__(self, out_dir, tasks, flush_steps=2000):
        self.dir, self.tasks, self.flush_steps = out_dir, list(tasks), flush_steps
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "tasks.json"), "w") as f:
            json.dump(self.tasks, f)
        self._reset()

    def _reset(self):
        self.cols = {"step": [], "key": [], "n_targets": [], "refresh": [], "y": [], "y_tasks": []}
        self.first = None

    def add(self, step, refresh, keys, n_targets, y, y_tasks=None):
        if self.first is None:
            self.first = step
        n = len(keys)
        self.cols["step"].append(np.full(n, step, np.int32))
        self.cols["refresh"].append(np.full(n, refresh, np.int32))
        self.cols["key"].append(np.asarray(keys, np.int64))
        self.cols["n_targets"].append(np.asarray(n_targets, np.int32))
        self.cols["y"].append(np.asarray(y, np.float32))
        yt = np.full((n, len(self.tasks)), np.nan, np.float32)
        if y_tasks is not None:
            for i, t in enumerate(self.tasks):
                if t in y_tasks:
                    yt[:, i] = np.asarray(y_tasks[t], np.float32)
        self.cols["y_tasks"].append(yt)

    def maybe_flush(self, step, force=False):
        if self.first is None or (not force and step - self.first < self.flush_steps):
            return
        arrs = {k: np.concatenate(v) for k, v in self.cols.items()}
        tmp = os.path.join(self.dir, f"labels_{self.first:07d}.tmp.npz")

        def write():
            np.savez(tmp, **arrs)
            os.replace(tmp, os.path.join(self.dir, f"labels_{self.first:07d}.npz"))
        retry_io(write, f"labels dump {self.first}")
        self._reset()
