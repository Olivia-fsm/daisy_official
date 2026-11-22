#!/usr/bin/env python
"""Instruction tuning (supervised fine-tuning, SFT) of a pretrained checkpoint (a run of train.py) on chat data prepared
by prep_tulu.py, so that every pretraining run can be instruction-tuned identically and evaluated again (do the gains
of the selection survive instruction tuning?).

Same model code and checkpoint format as pretraining; the output directory loads with daisy/eval/lm_eval_run.py
(OUT/daisy/meta_train.json = the base run's configuration plus an "sft" record, OUT/ckpts/latest/main.pt).
  * data: the training conversations in one seeded order (identical for every base run), packed whole into rows of
    seq_len + 1 tokens with document-masked attention; loss on assistant tokens only (daisy/sft/data.py);
  * step: --rows rows per GPU per micro-step x --acc micro-steps x GPUs; the loss is the mean over the step's target
    tokens across all GPUs;
  * optimizer: AdamW, fresh state (the pretraining optimizer state is not loaded), linear warmup over --warmup_frac of
    the steps, then linear decay to 0; gradient clipping; bf16 autocast; torch.compile;
  * validation: mean target-token loss over the held-out conversations before SFT (step 0), every --eval_every steps,
    and at the end.

  torchrun --standalone --nproc_per_node=4 daisy/sft/sft_train.py --base EXPS/RUN --data DATAPATH/tulu3-sft \
      --out EXPS/RUN-sft [--ckpt latest] [--lr 3e-5] [--epochs 1] [--rows 16] [--acc 1] [--compile]

Several epochs (--epochs K):
  * epoch 0 visits the packed rows in the packing order, epoch e > 0 in a fresh seeded order; an epoch is
    len(rows) // rows_per_step steps;
  * --save_epochs: a model checkpoint at the end of every epoch, OUT/ckpts/<step>/main.pt (itr = step), which the
    evaluators read with --ckpt <step>; daisy/meta_train.json is written with the first one, so it can be evaluated
    while the run continues;
  * --schedule linear (default): warmup, then linear decay to 0 over the whole run, so the checkpoint after epoch k < K
    is taken at lr = (1 - k/K) * peak; --schedule wsd: warmup, constant peak, and a linear decay to 0 over the last
    --cooldown_frac of an epoch before every epoch end; the cooldown before an inner epoch end runs on a copy of the
    model and optimizer that is discarded after its checkpoint, so every saved checkpoint is a fully decayed k-epoch run
    (cost: (K - 1) * cooldown_frac extra epochs);
  * --save_every N: model + optimizer + step in OUT/ckpts/resume/state.pt every N steps and before every cooldown copy;
    a restart with the same arguments continues from it (removed when the run finishes).
"""

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
from argparse import Namespace
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from daisy.sft.data import ChatSet  # noqa: E402


def load_base(exp_dir, ckpt):
    """The model of EXP/ckpts/<ckpt>/main.pt, rebuilt from the configuration in EXP/daisy/meta_train.json.
    -> (model on the CPU, the run's metadata, the checkpoint's step)."""
    from daisy.models import get_model

    meta = json.load(open(os.path.join(exp_dir, "daisy", "meta_train.json")))
    net = get_model(Namespace(**meta["config"]))
    ck = torch.load(os.path.join(exp_dir, "ckpts", str(ckpt), "main.pt"), map_location="cpu", weights_only=False)
    net.load_state_dict({k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ck["model"].items()},
                        strict=True)
    return net, meta, int(ck["itr"])


def lr_at(step, total, warmup, peak):
    if step < warmup:
        return peak * (step + 1) / warmup
    return peak * max(0.0, (total - step) / max(1, total - warmup))


def lr_wsd(step, end, warmup, peak, cool):
    """Warmup, constant peak, then linear decay to 0 over the `cool` steps before `end` (the same form as lr_at)."""
    if step < warmup:
        return peak * (step + 1) / warmup
    if step >= end - cool:
        return peak * max(0.0, (end - step) / cool)
    return peak


def to_cpu(obj):
    """A deep copy of a (nested) state dict with every tensor cloned to the CPU."""
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return copy.deepcopy(obj)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="pretraining experiment directory")
    p.add_argument("--ckpt", default="latest", help="checkpoint directory under BASE/ckpts (latest or a step)")
    p.add_argument("--data", required=True, help="prep_tulu.py output directory")
    p.add_argument("--out", required=True, help="output experiment directory")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup_frac", type=float, default=0.03)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=0, help="> 0: stop after this many steps (smoke tests)")
    p.add_argument("--max_rows", type=int, default=0, help="> 0: keep only the first rows of the packing (smoke tests)")
    p.add_argument("--rows", type=int, default=16, help="rows of seq_len tokens per GPU per micro-step")
    p.add_argument("--acc", type=int, default=1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--schedule", default="linear", choices=("linear", "wsd"))
    p.add_argument("--cooldown_frac", type=float, default=0.2, help="wsd: decay steps as a fraction of an epoch")
    p.add_argument("--save_epochs", action="store_true", help="a model checkpoint ckpts/<step> at every epoch end")
    p.add_argument("--save_every", type=int, default=0, help="> 0: resumable state every this many steps")
    p.add_argument("--stop_at", type=int, default=0, help="> 0: save the resumable state after this step and exit "
                   "without finishing (tests the resume)")
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--compile", action="store_true")
    a = p.parse_args(argv)

    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    cuda = torch.cuda.is_available()
    dev = torch.device(f"cuda:{local}" if cuda else "cpu")
    if world > 1:
        dist.init_process_group("nccl" if cuda else "gloo")
    if cuda:
        torch.cuda.set_device(dev)
    master = rank == 0
    torch.manual_seed(a.seed)

    net, meta, base_itr = load_base(a.base, a.ckpt)
    T = int(meta["config"]["sequence_length"])
    train, val = ChatSet(a.data, "train"), ChatSet(a.data, "val")
    keep = train.trainable(T)
    order = np.random.default_rng(a.seed).permutation(np.flatnonzero(keep))
    rows = train.pack(T, order)
    if a.max_rows > 0:
        rows = rows[:a.max_rows]
    vkeep = np.flatnonzero(val.trainable(T))
    vrows = val.pack(T, vkeep)
    per_step = a.rows * a.acc * world
    E = len(rows) // per_step                       # steps per epoch (the last len(rows) % per_step rows of an order are left out)
    total = int(a.epochs * E)
    if a.max_steps > 0:
        total = min(total, a.max_steps)
    warmup = max(1, int(a.warmup_frac * total))
    cool = max(1, int(round(a.cooldown_frac * E))) if a.schedule == "wsd" else 0
    # wsd: the inner epoch ends get a cooldown on a copy (the last one is the run's own decay)
    branches = [k for k in range(1, total // E + 1) if k * E < total] if a.schedule == "wsd" else []
    for k in branches:
        assert k * E - cool >= warmup, f"cooldown of epoch {k} would start in the warmup"
    if master:
        print(f"[sft] base {a.base} ckpt {a.ckpt} (iter {base_itr}); {int(keep.sum())}/{len(train)} conversations in "
              f"{len(rows)} rows of {T + 1}; {total} steps of {per_step} rows ({per_step * T} tokens), warmup {warmup}; "
              f"epoch = {E} steps; schedule {a.schedule}" + (f", cooldown {cool} steps before steps {[k * E for k in branches]} "
              f"and {total}" if a.schedule == "wsd" else "") + f"; val {len(vkeep)} conversations in {len(vrows)} rows",
              flush=True)

    out_dir = a.out
    state_path = os.path.join(out_dir, "ckpts", "resume", "state.pt")
    sig = {k: getattr(a, k) for k in ("base", "ckpt", "data", "lr", "warmup_frac", "epochs", "max_steps", "max_rows",
                                      "rows", "acc", "beta1", "beta2", "wd", "clip", "seed", "schedule", "cooldown_frac")}
    sig["world"] = world
    st = None
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        assert st["sig"] == sig, f"resume state {state_path} is from other arguments: {st['sig']} != {sig}"
        net.load_state_dict(st["model"], strict=True)

    net = net.to(dev).train()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(a.beta1, a.beta2), weight_decay=a.wd, eps=1e-8,
                            fused=cuda)
    if st is not None:
        opt.load_state_dict(st["opt"])
    model = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local] if cuda else None) if world > 1 else net
    fwd = torch.compile(model) if a.compile else model
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if cuda else nullcontext()

    def tensors(rs):
        return [torch.from_numpy(v).to(dev, non_blocking=True) for v in train.batch(rs, T)]

    def all_sum(v):
        t = torch.tensor(v, dtype=torch.float64, device=dev)
        if world > 1:
            dist.all_reduce(t)
        return t.tolist()

    def barrier():
        if world > 1:
            dist.barrier()

    @torch.no_grad()
    def evaluate():
        net.eval()
        s = n = 0.0
        for i in range(rank * a.rows, len(vrows), world * a.rows):
            x, y, d = [torch.from_numpy(v).to(dev) for v in val.batch(vrows[i:i + a.rows], T)]
            k = int((y >= 0).sum())
            if k == 0:
                continue
            with amp:
                out = net(x, targets=y, doc_ids=d)
            s += float(out["loss"]) * k
            n += k
        net.train()
        s, n = all_sum([s, n])
        return s / max(n, 1.0), int(n)

    log = None
    if master:
        os.makedirs(os.path.join(out_dir, "sft"), exist_ok=True)
        log = open(os.path.join(out_dir, "sft", "metrics.jsonl"), "a")
    git = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    def emit(rec):
        if master:
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"[sft] {json.dumps(rec)}", flush=True)

    def write_meta(**extra):
        rec = {"base": os.path.abspath(a.base), "base_ckpt": a.ckpt, "base_itr": base_itr, "data": os.path.abspath(a.data),
               "steps": total, "steps_per_epoch": E, "rows_per_step": per_step, "tokens_per_step": per_step * T,
               "args": vars(a), "git": git, "job": os.environ.get("SLURM_JOB_ID"), **extra}
        os.makedirs(os.path.join(out_dir, "daisy"), exist_ok=True)
        tmp = os.path.join(out_dir, "daisy", "meta_train.json.tmp")
        with open(tmp, "w") as f:
            json.dump(dict(meta, sft=rec), f, indent=1)
        os.replace(tmp, os.path.join(out_dir, "daisy", "meta_train.json"))

    def model_sd():
        return {k.replace("_orig_mod.", "").replace("module.", ""): v.detach().cpu() for k, v in net.state_dict().items()}

    def save_model(step):
        """ckpts/<step>/main.pt, written to a temporary directory and renamed; an existing one is kept."""
        d = os.path.join(out_dir, "ckpts", str(step))
        if master and not os.path.exists(d):
            tmp = d + ".tmp"
            os.makedirs(tmp, exist_ok=True)
            torch.save({"model": model_sd(), "itr": step}, os.path.join(tmp, "main.pt"))
            os.replace(tmp, d)
            write_meta(in_progress=True, val_loss_before=v0)
        barrier()

    def save_state(step):
        if master:
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            torch.save({"model": model_sd(), "opt": opt.state_dict(), "step": step, "sig": sig, "val_loss_before": v0},
                       state_path + ".tmp")
            os.replace(state_path + ".tmp", state_path)
        barrier()

    def rows_of(step, cache={}):
        """This rank's micro-batches of `step`: epoch 0 in the packing order, epoch e > 0 in a fresh seeded order."""
        e, i = divmod(step, E)
        if e not in cache:
            cache.clear()
            cache[e] = np.arange(len(rows)) if e == 0 else np.random.default_rng([a.seed, e]).permutation(len(rows))
        idx = cache[e][i * per_step:(i + 1) * per_step]
        base = rank * a.rows * a.acc
        return [[rows[r] for r in idx[base + j * a.rows: base + (j + 1) * a.rows]] for j in range(a.acc)]

    def train_step(step, lr):
        """One optimizer step at `lr` on the rows of `step`; -> (summed target loss, targets) on this rank, grad norm."""
        for g in opt.param_groups:
            g["lr"] = lr
        batches = [tensors(rs) for rs in rows_of(step)]
        n_loc = [int((y >= 0).sum()) for _, y, _ in batches]
        n_glob = all_sum([float(sum(n_loc))])[0]
        sl = sn = 0.0
        for j, (x, y, d) in enumerate(batches):
            if world > 1:
                model.require_backward_grad_sync = j == a.acc - 1     # all-reduce on the last micro-step only
            with amp:
                out = fwd(x, targets=y, doc_ids=d)
            # mean over the step's targets on all GPUs: DDP averages over GPUs, so scale by world / n_glob
            loss = out["loss"] * (n_loc[j] * world / n_glob)
            loss.backward()
            sl += float(out["loss"].detach()) * n_loc[j]
            sn += n_loc[j]
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), a.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        return sl, sn, gn

    first_log = [True]          # the process's first logging window includes compilation: no s_per_step

    def run(steps, lr_fn, tag=None):
        """Train over `steps` with lr_fn(step), logging every log_every steps (tag: the cooldown copy's epoch)."""
        t0, tl, tn, ns = time.time(), 0.0, 0.0, 0
        for k, step in enumerate(steps):
            lr = lr_fn(step)
            sl, sn, gn = train_step(step, lr)
            tl, tn, ns = tl + sl, tn + sn, ns + 1
            if (step + 1) % a.log_every == 0 or step + 1 == total or k + 1 == len(steps):
                tl, tn = all_sum([tl, tn])
                dt = time.time() - t0
                emit({"step": step + 1, "train_loss": tl / max(tn, 1.0), "lr": lr, "grad_norm": float(gn),
                      "s_per_step": None if first_log[0] else dt / ns, "elapsed": round(dt, 1),
                      **({"cooldown_of_epoch": tag} if tag is not None else {})})
                first_log[0] = False
                t0, tl, tn, ns = time.time(), 0.0, 0.0, 0
            yield step

    def cooldown(k):
        """wsd: from the main run at step k*E - cool, decay to 0 at k*E on a copy, save ckpts/<k*E>, restore."""
        end = k * E
        snap_m, snap_o = to_cpu(net.state_dict()), to_cpu(opt.state_dict())
        for _ in run(range(end - cool, end), lambda s: lr_wsd(s, end, warmup, a.lr, cool), tag=k):
            pass
        v, _ = evaluate()
        emit({"step": end, "epoch": k, "val_loss": v, "cooldown_of_epoch": k})
        save_model(end)
        net.load_state_dict(snap_m)
        opt.load_state_dict(snap_o)
        opt.zero_grad(set_to_none=True)

    if st is None:
        start = 0
        v0, nv = evaluate()
        emit({"step": 0, "val_loss": v0, "val_targets": nv})
    else:
        start, v0 = int(st["step"]), float(st["val_loss_before"])
        emit({"step": start, "resumed": True, "job": os.environ.get("SLURM_JOB_ID")})
        del st
    lr_main = (lambda s: lr_at(s, total, warmup, a.lr)) if a.schedule == "linear" else \
        (lambda s: lr_wsd(s, total, warmup, a.lr, cool))
    starts = {k * E - cool: k for k in branches}
    step = start
    while step < total:
        k = starts.get(step)
        if k is not None and not os.path.exists(os.path.join(out_dir, "ckpts", str(k * E))):
            if a.save_every > 0:
                save_state(step)
            cooldown(k)
        nxt = min([s for s in starts if s > step] + [total])    # the next cooldown start (or the end)
        for step in run(range(step, nxt), lr_main):
            done = step + 1
            if (done % a.eval_every == 0 and done < total) or (a.schedule == "linear" and a.save_epochs and done % E == 0
                                                                and done < total):
                v, _ = evaluate()
                emit({"step": done, "val_loss": v, **({"epoch": done // E} if done % E == 0 else {})})
            if a.schedule == "linear" and a.save_epochs and done % E == 0 and done < total:
                save_model(done)
            if a.save_every > 0 and done % a.save_every == 0 and done < total:
                save_state(done)
            if a.stop_at > 0 and done == a.stop_at:
                save_state(done)
                if master:
                    print(f"[sft] --stop_at {done}: stopped without finishing", flush=True)
                if world > 1:
                    dist.destroy_process_group()
                return
        step = nxt
    v1, nv = evaluate()
    emit({"step": total, "val_loss": v1, "val_targets": nv, "final": True})
    if master:
        os.makedirs(os.path.join(out_dir, "ckpts", "latest"), exist_ok=True)
        torch.save({"model": model_sd(), "itr": total}, os.path.join(out_dir, "ckpts", "latest", "main.pt"))
        if a.save_epochs and not os.path.exists(os.path.join(out_dir, "ckpts", str(total))):
            os.makedirs(os.path.join(out_dir, "ckpts", str(total)))       # the last epoch: a hard link to latest
            os.link(os.path.join(out_dir, "ckpts", "latest", "main.pt"), os.path.join(out_dir, "ckpts", str(total), "main.pt"))
        write_meta(val_loss_before=v0, val_loss_after=v1)
        with open(os.path.join(out_dir, "sft", "finished.json"), "w") as f:
            json.dump({"steps": total, "val_loss_before": v0, "val_loss_after": v1, "time": time.strftime("%FT%T")}, f)
        if os.path.exists(state_path):
            os.remove(state_path)                    # the run's own resume state, useless once finished
        print(f"[sft] done: val loss {v0:.4f} -> {v1:.4f}; wrote {out_dir}", flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
