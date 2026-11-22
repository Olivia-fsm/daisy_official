#!/usr/bin/env python
"""Held-out losses of saved checkpoints, after training: for each run and checkpoint step, the mean token loss on the
val split of one or more task directories (the task format of daisy/labels/targets.py, e.g. the held-out FineWeb-Edu
documents of daisy/targets/build_fwedu_target.py, OUT/fwedu_theldout/val.*), computed exactly as the trainer's
held-out downstream loss (daisy/eval/heldout.EvalSet): target_batches(split="val", seed=1) packs a fixed seeded subset
of the items up to --max_tokens (0 = all items, up to the last whole row: target_batches packs whole rows) into
document-masked rows of the run's sequence length; validation_pass, forward only, in the run's autocast dtype. So a
run's checkpoint gives, on a benchmark's val split with --max_tokens = --daisy_eval_tokens, the same loss as its
"bench_val" record of that step (the check that --check_bench does, for runs trained with --daisy_eval_dir).

Runs are experiment directories under --exps: EXP/daisy/meta_train.json holds the run's configuration, and the
checkpoints are EXP/ckpts/<step>/main.pt (the permanent ones; --steps all = every numbered one). Results go to
--out/<run>.json ({task: {step: {"loss", "tokens", "items"}}}, merged with what is there: evaluations already in the
file are skipped, so a rerun does only what is left).

  python daisy/eval/heldout_ckpt.py --exps EXPS --runs RUN_A,RUN_B \\
      --tasks fwedu=TARGETS/fineweb_edu/fwedu_theldout --out EXPS/heldout_ckpt
"""

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from daisy.labels.gates import validation_pass  # noqa: E402
from daisy.labels.targets import target_batches  # noqa: E402


def load_run(exp_dir, device, ckpt):
    """As lm_eval_run.load_run (which imports lm_eval): the run's model, built from the configuration in
    EXP/daisy/meta_train.json, with the weights of EXP/ckpts/<ckpt>/main.pt (daisy/train/checkpoint.py).
    -> (model in eval mode, configuration, the checkpoint's step, the run's metadata)."""
    from argparse import Namespace

    from daisy.models import get_model

    meta = json.load(open(os.path.join(exp_dir, "daisy", "meta_train.json")))
    cfg = Namespace(**meta["config"])
    net = get_model(cfg)
    ck = torch.load(os.path.join(exp_dir, "ckpts", str(ckpt), "main.pt"), map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model"], strict=True)
    return net.to(device).eval(), cfg, int(ck["itr"]), meta


def steps_of(exp_dir, steps, min_age_s=0.0):
    """The numbered checkpoints of EXP/ckpts (all of them, or those of the comma list `steps`); with min_age_s > 0 only
    those whose main.pt exists and was last written at least min_age_s seconds ago (the trainer writes main.pt in
    place, so a run that is still training may be in the middle of one)."""
    ck = os.path.join(exp_dir, "ckpts")
    have = sorted(int(d) for d in os.listdir(ck) if d.isdigit())
    if min_age_s > 0:
        now = time.time()
        have = [s for s in have if os.path.exists(f := os.path.join(ck, str(s), "main.pt"))
                and now - os.path.getmtime(f) >= min_age_s]
    if steps == "all":
        return have
    want = [int(s) for s in steps.split(",") if s.strip()]
    return [s for s in want if s in have]


def type_ctx_of(cfg, device):
    if "cuda" not in device:
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype={"float32": torch.float32, "float16": torch.float16,
                                                         "bfloat16": torch.bfloat16}[cfg.dtype])


def evaluate(net, cfg, task_dir, max_tokens, device):
    """-> (mean token loss, tokens, items used) of the val split, as EvalSet.losses()."""
    batches, info = target_batches(task_dir, cfg.sequence_length, max_tokens or 10**12, split="val", seed=1)
    layers = {n: m for n, m in net.named_modules() if isinstance(m, torch.nn.Linear)}
    _, ell, cnt = validation_pass(net, layers, batches, 1, type_ctx_of(cfg, device), compute_grad=False)
    return float(ell[0]), int(cnt[0]), int(info["items_used"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exps", required=True, help="directory of the experiment directories (--results_base_folder)")
    p.add_argument("--runs", required=True, help="comma list of experiment directory names under --exps")
    p.add_argument("--tasks", required=True, help="comma list name=task_dir (a dir with val.bin, val.len)")
    p.add_argument("--steps", default="all", help="'all' or a comma list of checkpoint steps")
    p.add_argument("--min_age_s", type=float, default=0.0, help="skip checkpoints whose main.pt is younger than this "
                   "(runs still training: a checkpoint may be half written); 0 = every checkpoint")
    p.add_argument("--max_tokens", type=int, default=0, help="0 = the whole val split (up to its last whole row)")
    p.add_argument("--out", required=True)
    p.add_argument("--check_bench", default="", help="name=task_dir of a benchmark: also evaluate it with "
                   "--daisy_eval_tokens and compare with the run's bench_val record of each step")
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tasks = dict(t.split("=", 1) for t in a.tasks.split(",") if t.strip())
    os.makedirs(a.out, exist_ok=True)
    for run in [r for r in a.runs.split(",") if r.strip()]:
        exp = os.path.join(a.exps, run)
        path = os.path.join(a.out, f"{run}.json")
        res = json.load(open(path)) if os.path.exists(path) else {}
        for step in steps_of(exp, a.steps, a.min_age_s):
            todo = [n for n in tasks if str(step) not in res.get(n, {})]
            chk = a.check_bench.split("=", 1) if a.check_bench else None
            if not todo and not (chk and str(step) not in res.get("check/" + chk[0], {})):
                continue
            t0 = time.time()
            net, cfg, itr, meta = load_run(exp, device, step)
            if itr != step:
                raise ValueError(f"{exp}/ckpts/{step}: itr {itr}")
            for n in todo:
                loss, tok, items = evaluate(net, cfg, tasks[n], a.max_tokens, device)
                res.setdefault(n, {})[str(step)] = {"loss": loss, "tokens": tok, "items": items}
            if chk:
                loss, tok, items = evaluate(net, cfg, chk[1], cfg.daisy_eval_tokens, device)
                rec = None
                with open(os.path.join(exp, "daisy", "metrics.jsonl")) as f:
                    for line in f:
                        r = json.loads(line)
                        if r.get("kind") == "bench_val" and r.get("step") == step:
                            rec = r.get(f"loss/{chk[0]}")   # a resumed run's last record of a step wins
                res.setdefault("check/" + chk[0], {})[str(step)] = {"loss": loss, "record": rec}
            res["_meta"] = {"run": run, "git": meta.get("git", {}).get("commit"), "device": device, "tasks": tasks}
            with open(path + ".tmp", "w") as f:
                json.dump(res, f, indent=1)
            os.replace(path + ".tmp", path)
            print(f"[heldout_ckpt] {run} @ {step}: " + ", ".join(f"{n} {res[n][str(step)]['loss']:.4f}" for n in todo)
                  + (f", check {chk[0]} {res['check/' + chk[0]][str(step)]}" if chk else "")
                  + f" ({time.time() - t0:.1f}s)", flush=True)
            del net
    print("[heldout_ckpt] done")


if __name__ == "__main__":
    main()
