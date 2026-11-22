"""The MATES output directory (EXP/daisy/mates/) and its atomic writes.

Per stage s (the stage a boundary selects for: the boundary at step (s - 1) U writes stage_{s}/):
  probes_r{rank}.npz   key, length, grad_norm (before clipping), clipped, loss_after, doc_loss, seconds (one row per
                       probe); loss_before, lr, wall_s (the rank's)
  probe_meta.json      loss_before, lr, precision, clip, counts (written once all ranks have probed)
  bert/model.pt        the best influence model (state dict), bert/config.json; bert_log.jsonl (every evaluation);
  bert_done.json       best step and validation metrics, the fine-tuning wall seconds (the completion marker)
  select_r{rank}.npz   cand_key, cand_len, score, gumbel, selected; sel_key, sel_end (the training order)
  select_r{rank}.json  the pool-stream state after the candidate draw, the rank's seconds and summary statistics
Run level: stages.jsonl and stage_costs.jsonl (one record per boundary), budget.json (the running GPU-hour total).
Every file is written to a temporary name and renamed, so a file that exists is complete: a boundary interrupted by
the job's end resumes from the artifacts that exist (stage.py).
"""

import json
import os

import numpy as np

from daisy.runmeta import retry_io


class MatesDir:
    def __init__(self, root):
        self.root = str(root)
        os.makedirs(self.root, exist_ok=True)

    def stage(self, s):
        d = os.path.join(self.root, f"stage_{s}")
        os.makedirs(d, exist_ok=True)
        return d

    def probes(self, s, rank):
        return os.path.join(self.stage(s), f"probes_r{rank}.npz")

    def probe_meta(self, s):
        return os.path.join(self.stage(s), "probe_meta.json")

    def bert(self, s):
        return os.path.join(self.stage(s), "bert")

    def bert_log(self, s):
        return os.path.join(self.stage(s), "bert_log.jsonl")

    def bert_done(self, s):
        return os.path.join(self.stage(s), "bert_done.json")

    def select_npz(self, s, rank):
        return os.path.join(self.stage(s), f"select_r{rank}.npz")

    def select_json(self, s, rank):
        return os.path.join(self.stage(s), f"select_r{rank}.json")

    def run_file(self, name):
        return os.path.join(self.root, name)

    def load_list(self, rank):
        """load_list(stage) for MatesStream: the stage's selected list in training order (keys, ends)."""
        def load(s):
            z = np.load(self.select_npz(s, rank))
            return z["sel_key"], z["sel_end"]
        return load


def _atomic(path, write):
    tmp = f"{path}.tmp{os.getpid()}"

    def go():
        write(tmp)
        os.replace(tmp, path)
    retry_io(go, f"write {path}")


def save_npz(path, **arrays):
    def write(tmp):
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
    _atomic(path, write)


def save_json(path, obj):
    def write(tmp):
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=1)
    _atomic(path, write)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
