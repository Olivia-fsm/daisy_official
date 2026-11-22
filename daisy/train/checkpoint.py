"""Checkpoint files of a run: ckpts/{latest,<step>}/main.pt (model, optimizer, schedule, step; written by rank 0) and
worker_{rank}.pt (the random generator states of every rank)."""

import random
import shutil
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(model, opt, scheduler, itr, ckpt_dir):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "itr": itr,
    }
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    torch.save(checkpoint, ckpt_dir / "main.pt")


def load_checkpoint(model, opt, scheduler, ckpt_path, device):
    """Restore model, optimizer and schedule from main.pt. -> the step the checkpoint was written at."""
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["itr"]


def save_worker_state(ckpt_dir, rank):
    st = {"rng_torch_cpu": torch.random.get_rng_state(), "rng_np": np.random.get_state(), "rng_python": random.getstate(),
          "rng_torch_gpu": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    torch.save(st, Path(ckpt_dir) / f"worker_{rank}.pt")


def load_worker_state(ckpt_dir, rank):
    st = torch.load(Path(ckpt_dir) / f"worker_{rank}.pt", weights_only=False)
    torch.random.set_rng_state(st["rng_torch_cpu"])
    np.random.set_state(st["rng_np"])
    random.setstate(st["rng_python"])
    if st["rng_torch_gpu"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(st["rng_torch_gpu"])


def swap_latest(ckpts):
    """DAISY_ATOMIC_LATEST=1, for clusters that may kill a job at any time: the finished ckpts/latest.new becomes
    ckpts/latest, so that a kill at any point leaves a whole checkpoint (scripts/restore_latest.sh puts it in place
    before a restart)."""
    latest, new, old = ckpts / "latest", ckpts / "latest.new", ckpts / "latest.old"
    if latest.exists():
        latest.rename(old)
    new.rename(latest)
    if old.exists():
        shutil.rmtree(old)
