"""The DAiSY training loop (entry: train.py).

One step, in order (all ranks unless noted):
  1. checkpoints   permanent (--permanent_ckpt_interval) and latest (--latest_ckpt_interval, and the last step)
  2. chain stop    DAISY_STOP_AT: stop at a refresh step before the job's time limit, with a latest checkpoint
  3. refresh       every R steps: the held-out web-text loss vector (refresh.jsonl); the learner's target gradient; the
                   held-out downstream losses
  4. micro-batches stream.next_microbatch (the selector scores its candidate window with the latest weights), forward
                   with the label gates on label steps, backward
  5. optimizer     clip, step, schedule
  6. learner       labels -> ridge on rank 0, its new weights to the selector (every U steps); a fixed scorer
                   (similarity, DSIR) switches on at the scorer start
The MATES baseline (--mates, daisy/mates/stage.py) adds, before the checkpoints, the GPU-hour budget check (a spent
budget ends the run like a finished one) and, after the chain stop, the stage boundary (probes, influence model,
selection; then a checkpoint); its stream is MatesStream and it starts from --mates_init_from when it has no checkpoint.
Outputs: EXP/daisy/ (meta_train.json, heldout_set.json, refresh.jsonl, metrics.jsonl, labels/, mates/,
step_times_r{rank}.npy, timing.json, finished.json) and EXP/ckpts/{latest,<step>}/ (main.pt, worker_{rank}.pt,
stream_r{rank}.json, daisy_r{rank}.pt).
"""

import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from daisy.data.docstore import DocStore
from daisy.data.pool import DsPool, ShardDocStream
from daisy.data.prefetch import Prefetcher
from daisy.data.select import PHI_DIM, SelectingStream
from daisy.data.stream import UniformDocStream, heldout_batches
from daisy.eval.heldout import EvalSet, eval_tasks
from daisy.labels.gates import attach_gate_names, validation_pass
from daisy.runmeta import append_line, retry_io, write_meta
from daisy.train.checkpoint import (load_checkpoint, load_worker_state, save_checkpoint, save_worker_state,
                                    swap_latest)
from daisy.train.learner import OnlineLearner

try:
    import wandb
except ImportError:      # only needed with --wandb
    wandb = None


def already_finished(cfg, exp_dir, master):
    """A re-submitted job of a finished run returns before touching any output (the loop would repeat the final
    refresh and append duplicate records). A MATES run that stopped at its GPU-hour budget is finished below
    --iterations. -> the step the run reached if it is done, else None."""
    if not cfg.resume_from:
        return None
    fin = Path(exp_dir) / "daisy" / "finished.json"
    if getattr(cfg, "mates", False) and fin.exists():
        with open(fin) as f:
            rec = json.load(f)
        if rec.get("reason") == "budget":
            if master:
                print(f"[daisy] {exp_dir} stopped at its GPU-hour budget at iter {rec['iter']}: nothing to do")
            return int(rec["iter"])
    done = int(torch.load(str(Path(cfg.resume_from) / "main.pt"), map_location="cpu", mmap=True,
                          weights_only=False)["itr"])
    if done < cfg.iterations:
        return None
    if master:
        print(f"[daisy] {exp_dir} is already trained to iter {done} (--iterations {cfg.iterations}): nothing to do")
    return done


def build_stream(cfg, store, idx, stream_seed, rank, world, mates_dir=None):
    """The training document stream: the tokenized pool (--daisy_pool; held-out documents and their text / URL
    duplicates excluded by the index's hashes) or the index itself; wrapped by the selector when --daisy_select_mult > 0
    (daisy/data/select.py), or by MatesStream (--mates: the rank's last shards reserved for probes, the stage lists of
    mates_dir), and by the prefetcher. -> (stream, selector or None, fixed scorer weights or None)."""
    mates = bool(getattr(cfg, "mates", False))
    if mates and not (cfg.daisy_pool and float(cfg.daisy_select_mult) == 0 and not cfg.daisy_learn):
        raise ValueError("--mates trains on the pool without a selector: needs --daisy_pool, --daisy_select_mult 0 and "
                         "no --daisy_learn")
    if cfg.daisy_pool:
        held = np.load(os.path.join(idx, "heldout_rows.npy"))
        th = np.load(os.path.join(idx, "text_hash.npy"), mmap_mode="r")[held]
        uh = np.load(os.path.join(idx, "url_hash.npy"), mmap_mode="r")[held]
        excl = {}
        if cfg.daisy_pool_exclude:
            # documents never to train on beyond the index's held-out ones (e.g. a target drawn from the pool itself),
            # and their text / URL copies everywhere: removed from each shard's order after its permutation
            # (daisy/data/pool.py), so that every other document keeps its place in the stream
            xdir = cfg.daisy_pool_exclude
            excl = {"excl_text": np.load(os.path.join(xdir, "text_hash.npy")).astype(np.uint64),
                    "excl_url": np.load(os.path.join(xdir, "url_hash.npy")).astype(np.uint64)}
            if rank == 0:
                print(f"[daisy] pool exclusion: {len(excl['excl_text'])} text / {len(excl['excl_url'])} URL hashes "
                      f"from {xdir}", flush=True)
        stream = ShardDocStream(DsPool(cfg.daisy_pool), cfg.sequence_length, cfg.daisy_max_doc_tokens, seed=stream_seed,
                                rank=rank, world=world, n_open=cfg.daisy_pool_open_shards,
                                held_text=th, held_url=uh[uh != 0],
                                # limited-pool runs: every rank's subset is its share of the subset tokens
                                subset_tokens=int(cfg.daisy_pool_subset_tokens) // world,
                                reshuffle=bool(cfg.daisy_pool_reshuffle), preload=int(cfg.daisy_pool_preload),
                                keep_dir=cfg.daisy_pool_keep, keep_min=int(cfg.daisy_pool_keep_min),
                                **excl, **({"reserve": cfg.mates_probe_shards} if mates else {}))
        if stream.keep_dir and rank == 0:
            print(f"[daisy] pool filter: only documents with score >= {stream.keep_min} in {stream.keep_dir}")
        if mates:
            from daisy.mates.artifacts import MatesDir
            from daisy.mates.streams import MatesStream

            stream = MatesStream(stream, MatesDir(mates_dir).load_list(rank))
    else:
        if int(cfg.daisy_pool_subset_tokens):
            raise ValueError("--daisy_pool_subset_tokens needs --daisy_pool")
        mask = np.load(os.path.join(idx, "train_mask.u8.npy"), mmap_mode="r")
        stream = UniformDocStream(store, mask, cfg.sequence_length, cfg.daisy_max_doc_tokens, seed=stream_seed,
                                  rank=rank, world=world)
    selector = w_static = None
    if float(cfg.daisy_select_mult) > 0:
        # online selection: candidates scored by the running ridge on phi, or by a fixed scorer
        selector = stream = SelectingStream(stream, cfg.daisy_select_mult, seed=stream_seed, rank=rank,
                                            cosine=bool(cfg.daisy_select_cosine))
    if cfg.daisy_select_dsir:
        # the DSIR baseline (daisy/scorers/dsir.py): the candidates' precomputed importance weights, switched on at the
        # scorer start like the fixed scorer below; w_static (zeros) only switches it on, the weights are unused
        if selector is None or not cfg.daisy_pool:
            raise ValueError("--daisy_select_dsir needs --daisy_select_mult > 0 and --daisy_pool")
        if cfg.daisy_select_static:
            raise ValueError("--daisy_select_dsir and --daisy_select_static are two fixed scorers: pass one")
        from daisy.scorers.dsir import DsirScores

        selector.dsir = DsirScores(cfg.daisy_select_dsir, selector.store, cfg.daisy_max_doc_tokens)
        w_static = np.zeros(PHI_DIM + 1, dtype=np.float32)
        print(f"[daisy] DSIR scorer {cfg.daisy_select_dsir} ({selector.dsir.describe()}), on from step "
              f"{cfg.daisy_scorer_start}")
    elif cfg.daisy_select_static:
        # the similarity baseline: a fixed scorer (daisy/scorers/similarity.py), switched on at the scorer start like
        # the running ridge
        if selector is None:
            raise ValueError("--daisy_select_static needs --daisy_select_mult > 0")
        w_static = np.load(cfg.daisy_select_static).astype(np.float32)
        if w_static.shape[-1] != PHI_DIM + 1 or w_static.ndim not in (1, 2):
            raise ValueError(f"--daisy_select_static: expected ({PHI_DIM + 1},) = [w; b] or (K, {PHI_DIM + 1}) (max "
                             f"over K), got {w_static.shape}")
        print(f"[daisy] fixed scorer {cfg.daisy_select_static} (shape {w_static.shape}, |w| = "
              f"{np.round(np.linalg.norm(np.atleast_2d(w_static)[:, :-1], axis=1), 4).tolist()}, "
              f"cosine={bool(cfg.daisy_select_cosine)}), on from step {cfg.daisy_scorer_start}")
    elif selector is not None and not cfg.daisy_learn:
        raise ValueError("--daisy_select_mult without a fixed scorer selects with the running ridge: needs --daisy_learn")
    if cfg.daisy_prefetch > 0:
        # reads of the next micro-batches run on background threads (same batches, same order; exact resume)
        stream = Prefetcher(stream, cfg.batch_size, depth=cfg.daisy_prefetch, threads=cfg.daisy_io_threads)
    return stream, selector, w_static


class DaisyTrainer:
    """One training run; train_daisy() builds it and calls run()."""

    def __init__(self, model, opt, scheduler, exp_dir, backend, cfg):
        self.model, self.opt, self.scheduler, self.backend, self.cfg = model, opt, scheduler, backend, cfg
        self.exp_dir = Path(exp_dir)
        self.raw_model = backend.get_raw_model(model)
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world = backend.get_world_size()
        self.master = backend.is_master_process()
        if cfg.compile:
            # one graph with and one without the label gates, and one per padded document count; past the limit
            # dynamo silently falls back to eager for new shapes
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
            if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
                torch._dynamo.config.accumulated_cache_size_limit = max(torch._dynamo.config.accumulated_cache_size_limit,
                                                                        512)
        self.fwd = torch.compile(model) if cfg.compile else model
        self.type_ctx = (torch.amp.autocast(device_type="cuda", dtype={"float32": torch.float32, "float16": torch.float16,
                                                                        "bfloat16": torch.bfloat16}[cfg.dtype])
                         if "cuda" in cfg.device else nullcontext())
        self.out_dir = self.exp_dir / "daisy"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = bool(cfg.wandb and self.master)
        self._setup_data()
        self.layers = attach_gate_names(self.raw_model)       # the tracked Linear layers, named for the label gates
        self._setup_learner()

    # ------------------------------------------------------------------ setup
    def _setup_data(self):
        """Document store, the training stream (same data order on every rank count for a seed), the fixed held-out
        web-text set, and the run's metadata."""
        cfg, idx = self.cfg, self.cfg.daisy_index_dir
        self.store = DocStore(idx, phi_path=cfg.daisy_phi_path or None)
        # The document order depends on the run seed, so different seeds see different data. The DDP backend has
        # already offset cfg.seed by the local rank, so take rank 0's value: the permutation must be identical on every
        # rank before it is split rank::world.
        base_seed = torch.tensor([cfg.seed], dtype=torch.long, device=cfg.device)
        if self.world > 1:
            dist.broadcast(base_seed, src=0)
        base_seed = int(base_seed.item())
        stream_seed = cfg.data_seed + base_seed
        self.base_seed, self.stream_seed = base_seed, stream_seed
        self.stream, self.selector, self.w_static = build_stream(
            cfg, self.store, idx, stream_seed, self.rank, self.world,
            mates_dir=self.out_dir / "mates" if getattr(cfg, "mates", False) else None)
        # the fixed held-out set of the loss vector: docs_per_topic documents per topic, truncated
        self.vbatches, vinfo = heldout_batches(self.store, np.load(os.path.join(idx, "heldout_rows.npy")),
                                               np.load(os.path.join(idx, "heldout_topic.npy")),
                                               cfg.daisy_val_docs_per_topic, cfg.sequence_length,
                                               cfg.daisy_val_max_doc_tokens,
                                               rows_per_batch=cfg.daisy_val_rows_per_batch)
        self.D = vinfo["D"]
        if self.master:
            write_meta(self.out_dir, "train", vars(cfg),
                       {"world_size": self.world, "resume_from": cfg.resume_from, "python": sys.executable,
                        "torch": torch.__version__, "base_seed": base_seed, "stream_seed": stream_seed})
            print(f"[daisy] index {idx}: N={self.store.N:,}; stream seed {stream_seed} (data_seed {cfg.data_seed} + "
                  f"seed {base_seed}); held-out set: {vinfo}")
            with open(self.out_dir / "heldout_set.json", "w") as f:
                json.dump(vinfo, f)

    def _setup_learner(self):
        """The online learner (--daisy_learn), the held-out downstream evaluation (--daisy_eval_dir) and the MATES
        controller (--mates)."""
        cfg = self.cfg
        log = print if self.master else (lambda *a, **k: None)
        self.learner = None
        if cfg.daisy_learn:
            self.learner = OnlineLearner(cfg, self.raw_model, self.opt, self.type_ctx, self.world, self.rank,
                                         self.master, self.out_dir, cfg.device, emit=self.emit, log=log)
        self.bench_eval = None
        if cfg.daisy_eval_dir:
            self.bench_eval = EvalSet(self.raw_model, self.layers,
                                      {t: os.path.join(cfg.daisy_eval_dir, t) for t in eval_tasks(cfg)},
                                      cfg.sequence_length, cfg.daisy_eval_tokens, self.type_ctx, self.world, self.rank,
                                      log=log)
        self.mates = None
        if getattr(cfg, "mates", False):
            from daisy.mates.stage import MatesController

            self.mates = MatesController(self)

    def emit(self, step, kind, rec):
        """One record of metrics.jsonl (and wandb), rank 0."""
        if not self.master:
            return
        rec = {"step": step, "kind": kind, **rec}
        append_line(self.out_dir / "metrics.jsonl", json.dumps(rec))
        if self.use_wandb:
            wandb.log({"iter": step, **{f"daisy/{kind}/{k}": v for k, v in rec.items()
                                        if isinstance(v, (int, float)) and k != "step"}})

    # ------------------------------------------------------------------ checkpoints
    def resume(self, ckpt_dir, init=False):
        """Model, optimizer, schedule, random generator states, stream position and the learner (/ MATES) state of
        ckpt_dir. init: ckpt_dir is another run's checkpoint (--mates_init_from), checked against this run's config."""
        cfg = self.cfg
        ckpt_dir = Path(ckpt_dir)
        if init:
            self._check_init_from(ckpt_dir)
        self.it = load_checkpoint(self.model, self.opt, self.scheduler, ckpt_dir / "main.pt", cfg.device)
        load_worker_state(ckpt_dir, self.rank)
        with open(ckpt_dir / f"stream_r{self.rank}.json") as f:
            self.stream.load_state_dict(json.load(f))
        cs_path = ckpt_dir / f"daisy_r{self.rank}.pt"
        if cs_path.exists():
            cs = torch.load(cs_path, map_location=cfg.device, weights_only=False)
            if self.learner is not None and cs.get("learner") is not None:
                self.learner.load_state_dict(cs["learner"])
            if self.mates is not None and not init and cs.get("mates") is not None:
                self.mates.load_state_dict(cs["mates"])
        print(f"[daisy] resumed at iter {self.it}"
              + (f" from another run's checkpoint {ckpt_dir}" if init else ""))

    # what a run started from another run's checkpoint (--mates_init_from) must share with it: model, optimizer,
    # schedule, batch, data stream (the seed is compared as the stream seed: the per-rank cfg.seed is offset by rank)
    INIT_KEYS = ("n_embd", "n_head", "n_layer", "sequence_length", "vocab_size", "batch_size", "acc_steps",
                 "lr", "beta1", "beta2", "weight_decay", "grad_clip", "scheduler", "warmup_steps",
                 "scheduler_total_steps", "final_div_factor", "dtype", "data_seed", "daisy_pool",
                 "daisy_pool_open_shards", "daisy_max_doc_tokens", "daisy_index_dir")

    def _check_init_from(self, ckpt_dir):
        meta_path = ckpt_dir.resolve().parent.parent / "daisy" / "meta_train.json"
        if not meta_path.exists():
            print(f"[daisy] WARNING: {meta_path} not found: the configuration of {ckpt_dir} is not checked")
            return
        with open(meta_path) as f:
            meta = json.load(f)
        src, mine = meta["config"], vars(self.cfg)
        bad = {k: (src.get(k), mine.get(k)) for k in self.INIT_KEYS if k in src and src.get(k) != mine.get(k)}
        for k, want in (("world_size", self.world), ("stream_seed", self.stream_seed)):
            if meta.get(k) is not None and meta[k] != want:
                bad[k] = (meta[k], want)
        if bad:
            raise ValueError(f"--mates_init_from {ckpt_dir}: the source run differs in {bad} (source, this run)")

    def checkpoint(self, ckpt_dir):
        """main.pt (rank 0), worker_{rank}.pt (random generators), stream_r{rank}.json, daisy_r{rank}.pt ("learner": the
        scorer's state; "mates": the MATES cost state). Retried as a whole on a transient file-system error."""
        def write():
            if self.master:
                save_checkpoint(self.model, self.opt, self.scheduler, self.it, ckpt_dir)
            save_worker_state(ckpt_dir, self.rank)
            with open(Path(ckpt_dir) / f"stream_r{self.rank}.json", "w") as f:
                json.dump(self.stream.state_dict(), f)
            torch.save({"learner": self.learner.state_dict() if self.learner is not None else None,
                        **({"mates": self.mates.state_dict()} if self.mates is not None else {})},
                       Path(ckpt_dir) / f"daisy_r{self.rank}.pt")
        retry_io(write, f"checkpoint {ckpt_dir} (rank {self.rank})")

    def _latest_checkpoint(self):
        if self.learner is not None and self.master:
            self.learner.flush_dumps(self.it)
        if os.environ.get("DAISY_ATOMIC_LATEST") != "1":
            self.checkpoint(self.exp_dir / "ckpts" / "latest")
            return
        self.checkpoint(self.exp_dir / "ckpts" / "latest.new")
        if self.world > 1:
            dist.barrier()
        if self.master:
            swap_latest(self.exp_dir / "ckpts")
        if self.world > 1:
            dist.barrier()

    def _save_checkpoints(self):
        """Permanent and latest checkpoints of this step; both when a MATES run stops at its budget (the final ones)."""
        cfg = self.cfg
        final = self.budget_stopped
        if (cfg.permanent_ckpt_interval > 0 and self.it % cfg.permanent_ckpt_interval == 0) or self.it in self.extra_ckpts \
                or final:
            self.checkpoint(self.exp_dir / "ckpts" / str(self.it))
        if (cfg.latest_ckpt_interval > 0 and (self.it % cfg.latest_ckpt_interval == 0 or self.it == cfg.iterations)) \
                or final:
            self._latest_checkpoint()

    def _stop_requested(self):
        """Chained jobs (scripts/train.sh with a time limit): stop cleanly before the job's time limit (DAISY_STOP_AT,
        unix seconds; DAISY_STOP_MIN_ITER only for tests). The stop is decided only at refresh steps (a resume there
        repeats nothing an uninterrupted run would not do), with a latest checkpoint at that step. A MATES stage
        boundary at this step needs --mates_boundary_reserve_min more minutes."""
        cfg = self.cfg
        if not (self.stop_at and self.it % cfg.daisy_refresh == 0 and self.start_iter < self.it < cfg.iterations
                and self.it >= self.stop_min_iter):
            return False
        margin = self.mates.stop_margin(self.it) if self.mates is not None else 0.0
        flag = torch.tensor([float(time.time() + margin > self.stop_at)], device=cfg.device)
        if self.world > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)                # every rank stops at the same step
        if flag.item() <= 0:
            return False
        if not (cfg.latest_ckpt_interval > 0 and self.it % cfg.latest_ckpt_interval == 0):
            self._latest_checkpoint()
        if self.master:
            print(f"[daisy] stopping at iter {self.it} before the job's time limit (checkpoint saved)")
        return True

    # ------------------------------------------------------------------ the loop
    def run(self):
        cfg = self.cfg
        self.it = 0
        self.budget_stopped = False
        init_dir = getattr(cfg, "mates_init_from", "") if self.mates is not None else ""
        if cfg.resume_from:
            self.resume(cfg.resume_from)
        elif init_dir:
            self.resume(init_dir, init=True)                # MATES: stage 1 is another run's (the uniform run)
        if self.mates is not None:
            self.mates.start(self.it, init_dir or None)
        self.extra_ckpts = {int(x) for x in str(cfg.extra_ckpt_iters).split(",") if x.strip()}
        self.substep = self.it * cfg.acc_steps
        self.model.train()
        self.stats = {"train_loss": [], "target_loss": []}
        self.step_times, self.data_times, self.after_times = [], [], []
        # per step: seconds planning, waiting for reads, and (inside planning) waiting for shard preparation (the
        # orders and phi of newly opened shards: prep_stats of the pool stream and the selector)
        self.data_parts, self._prep_sync_last = [], self._prep_sync_s()
        self.last_refresh = None
        self.stop_at = float(os.environ.get("DAISY_STOP_AT", "0") or 0)
        self.stop_min_iter = int(os.environ.get("DAISY_STOP_MIN_ITER", "0") or 0)
        self.start_iter, stopped = self.it, False
        while self.it <= cfg.iterations:
            if self.mates is not None:
                self.budget_stopped = self.mates.budget_spent(self.it)
            self._save_checkpoints()
            if not self.budget_stopped and self._stop_requested():
                stopped = True
                break
            if self.mates is not None and not self.budget_stopped and self.mates.boundary_due(self.it):
                # a boundary whose expected cost would spend the rest of the budget is not run: the run ends here
                self.budget_stopped = self.mates.budget_spent(self.it, force=True, before_boundary=True)
                if self.budget_stopped:
                    self._save_checkpoints()
            if self.mates is not None and not self.budget_stopped and self.mates.boundary_due(self.it):
                if not (cfg.latest_ckpt_interval > 0 and self.it % cfg.latest_ckpt_interval == 0):
                    self._latest_checkpoint()          # the state before the boundary: a job cut inside it resumes here
                self.mates.boundary(self.it)
                self._latest_checkpoint()              # after it: a resume does not repeat it
                self.budget_stopped = self.mates.budget_spent(self.it, force=True)
                if self.budget_stopped:
                    self._save_checkpoints()
            if self.it % cfg.daisy_refresh == 0 or self.last_refresh is None:
                self._refresh()
            if self.it == cfg.iterations or self.budget_stopped:
                break
            self._train_step()
            self.it += 1
            if cfg.log_interval and self.it % cfg.log_interval == 0 and self.master:
                self._log_step()
        return self._finish(stopped)

    def _refresh(self):
        """Every R steps: one forward over the fixed held-out web-text rows -> the per-topic loss vector ell
        (refresh.jsonl); then the learner's target gradient and the held-out downstream losses."""
        cfg, t0 = self.cfg, time.time()
        _, ell, cnt = validation_pass(self.raw_model, self.layers, self.vbatches, self.D, self.type_ctx,
                                      world=self.world, rank=self.rank, compute_grad=False)
        self.last_refresh = self.it
        heldout_loss = float((ell.double() * cnt).sum() / cnt.sum())  # mean over all held-out target tokens
        rec = {"step": self.it, "lr": self.opt.param_groups[0]["lr"], "ell": ell.tolist(),
               "target_tokens": cnt.long().tolist(), "target_loss": heldout_loss,
               "seconds": time.time() - t0, "tokens": self.world * self.substep * cfg.sequence_length * cfg.batch_size}
        self.stats["target_loss"].append((self.it, heldout_loss))
        if self.master:
            append_line(self.out_dir / "refresh.jsonl", json.dumps(rec))
            print(f"[daisy] refresh @ {self.it}: held-out web-text loss {heldout_loss:.4f} ({rec['seconds']:.1f}s)")
            if self.use_wandb:
                wandb.log({"iter": self.it, "daisy/heldout_loss": heldout_loss, "daisy/refresh_s": rec["seconds"],
                           **{f"daisy/ell_{t}": float(v) for t, v in enumerate(ell.tolist())}})
        if self.learner is not None:
            self.learner.on_refresh(self.it)
        if self.bench_eval is not None and self.it % cfg.daisy_refresh == 0:     # all ranks: held-out downstream loss
            t0 = time.time()
            bl = self.bench_eval.losses()
            self.emit(self.it, "bench_val", {**{f"loss/{t}": v for t, v in bl.items()}, "seconds": time.time() - t0})
            if self.master:
                print(f"[daisy] held-out downstream loss @ {self.it}: mean {bl['mean']:.4f}")

    def _train_step(self):
        """One optimizer step over acc_steps micro-batches, then the learner and the selector's weights."""
        cfg, learner, it = self.cfg, self.learner, self.it
        if learner is not None:
            learner.before_step(it)
        t_start = time.perf_counter_ns()
        t_data = 0.0
        for micro in range(cfg.acc_steps):
            _t = time.perf_counter_ns()
            mb = self.stream.next_microbatch(cfg.batch_size)
            if mb is None:
                raise RuntimeError(f"iter {it}: the stream is used up (a MATES stage list ended before its boundary)")
            t_data += (time.perf_counter_ns() - _t) / 1e9
            self._micro_step(micro, mb)
        if hasattr(self.stream, "take_times"):
            plan_s, wait_s = self.stream.take_times()
            sync = self._prep_sync_s()
            self.data_parts.append((plan_s, wait_s, sync - self._prep_sync_last))
            self._prep_sync_last = sync
        if cfg.grad_clip != 0.0:
            self.grad_norm = torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), cfg.grad_clip)
        self.opt.step()
        if cfg.scheduler != "none":
            self.scheduler.step()
        self.opt.zero_grad(set_to_none=True)
        if "cuda" in cfg.device:
            torch.cuda.synchronize()
        self.dt = (time.perf_counter_ns() - t_start) / 1e9
        self.step_times.append(self.dt)
        self.data_times.append(t_data)
        if self.mates is not None:
            self.mates.on_step(self.dt)
        if learner is not None:
            _t = time.perf_counter_ns()
            learner.after_step(it)
            if self.selector is not None and self.w_static is None:
                w_sel = learner.selector_weights(it)                 # all ranks (broadcast from rank 0)
                if w_sel is not None:
                    self.selector.set_weights(w_sel)
            self.after_times.append((time.perf_counter_ns() - _t) / 1e9)
        if (self.selector is not None and self.w_static is not None and it >= cfg.daisy_scorer_start
                and self.selector.w is None):
            self.selector.set_weights(self.w_static)               # the fixed scorer: needs no labels
        return None

    def _prep_objects(self):
        """The stream and every stream it wraps (Prefetcher.stream, SelectingStream.base, ...), outermost first."""
        out, s = [], self.stream
        while s is not None and len(out) < 8:
            out.append(s)
            s = vars(s).get("stream", vars(s).get("base"))
        return out

    def _prep_sync_s(self):
        """Seconds the planning thread has waited for shard preparation so far (pool streams; 0 otherwise)."""
        return sum(v for o in self._prep_objects() for k, v in vars(o).get("prep_stats", {}).items()
                   if k.endswith("_sync_s"))

    def _micro_step(self, micro, mb):
        """Forward and backward of one micro-batch, with the label gates on label steps; hand the labels to the
        learner."""
        cfg, learner = self.cfg, self.learner
        x = torch.from_numpy(mb["x"]).to(cfg.device, non_blocking=True)
        y = torch.from_numpy(mb["y"]).to(cfg.device, non_blocking=True)
        seg = torch.from_numpy(mb["seg"]).to(cfg.device, non_blocking=True).long()
        n_seg = len(mb["rows"])
        gates = learner.label_gates(self.it, x.shape, seg.device) if learner is not None else None
        with self.type_ctx:
            with self.backend.get_context_for_microstep_forward(
                    model=self.model, microstep_idx=micro, gradient_accumulation_steps=cfg.acc_steps):
                outputs = self.fwd(x, targets=y, doc_ids=seg, gates=gates)
        self.loss = outputs["loss"] / cfg.acc_steps
        self.loss.backward()
        self.substep += 1
        if gates is not None:
            valid = y.reshape(-1) >= 0
            n_tgt = torch.bincount(seg.reshape(-1)[valid], minlength=n_seg)
            # the raw gate values are of grad(sum_x l / (N_micro * acc)): x this = per-token units of x
            scale = float(valid.sum()) * cfg.acc_steps / n_tgt.clamp_min(1).float()
            learner.collect_micro(torch.from_numpy(mb["phi"]).to(cfg.device), n_tgt,
                                  torch.from_numpy(mb["rows"]).to(cfg.device), scale, gates, seg, n_seg)

    def _log_step(self):
        cfg, sel = self.cfg, self.selector
        train_loss = self.loss.detach().cpu().item() * cfg.acc_steps
        tokens = self.world * self.substep * cfg.sequence_length * cfg.batch_size
        self.stats["train_loss"].append((self.it, train_loss))
        if sel is not None:
            st_ = sel.stats
            print(f"  select: {st_['microbatches']} micro-batches, {st_['candidates']} candidates -> {st_['chosen']} "
                  f"docs, scorer {'on' if sel.w is not None else 'off'}")
        print(f"Train: Iter={self.it} train_loss={train_loss:.3f} iter_dt={self.dt:.2e}s "
              f"lr={self.opt.param_groups[0]['lr']:.2e}")
        if self.use_wandb:
            wandb.log({"iter": self.it, "tokens": tokens, "train/loss": train_loss, "lr": self.opt.param_groups[0]["lr"],
                       "iter_dt": self.dt, "grad_norm": float(self.grad_norm) if cfg.grad_clip != 0.0 else 0.0})

    # ------------------------------------------------------------------ the end
    def _finish(self, stopped):
        """finished.json (a chained job that finds it exits at once), close the prefetcher / learner, and the timing
        summary: per-step times (step_times_r{rank}.npy: total, data wait, learner hook) and timing.json."""
        cfg = self.cfg
        if self.master and not stopped:
            with open(self.out_dir / "finished.json", "w") as f:
                json.dump({"iter": self.it, "job": os.environ.get("SLURM_JOB_ID"),
                           "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                           **(self.mates.finish_record(self.it) if self.budget_stopped else {})}, f)
        self.stats["stopped_at"] = self.it if stopped else None
        if isinstance(self.stream, Prefetcher):
            self.stream.close()
        if self.learner is not None and self.master:
            self.learner.close(self.it)
        for m in self.layers.values():
            del m.gate_name
        # median step time after warm-up, peak memory, per rank and reduced
        step_times, data_times, after_times = self.step_times, self.data_times, self.after_times
        warm = min(50, len(step_times) // 4)
        st = np.array(step_times[warm:]) if len(step_times) > warm else np.array(step_times)
        rec = {"rank": self.rank, "steps_timed": int(len(st)), "warmup_skipped": warm,
               "step_s_median": float(np.median(st)) if len(st) else None,
               "step_s_p10": float(np.percentile(st, 10)) if len(st) else None,
               "step_s_p90": float(np.percentile(st, 90)) if len(st) else None,
               "step_s_mean": float(st.mean()) if len(st) else None,
               # inside step_s: time waiting for the stream (host reads and packing); outside it: the learner hook
               "data_s_mean": float(np.mean(data_times[warm:])) if len(data_times) > warm else None,
               "data_s_median": float(np.median(data_times[warm:])) if len(data_times) > warm else None,
               "data_s_p90": float(np.percentile(data_times[warm:], 90)) if len(data_times) > warm else None,
               "data_s_max": float(np.max(data_times[warm:])) if len(data_times) > warm else None,
               "compute_s_median": float(np.median(st - np.array(data_times[warm:]))) if len(st) else None,
               "learner_s_median": float(np.median(after_times[warm:])) if len(after_times) > warm else None,
               "learner_s_p90": float(np.percentile(after_times[warm:], 90)) if len(after_times) > warm else None,
               "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30 if "cuda" in cfg.device else None,
               "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30 if "cuda" in cfg.device else None,
               "compile": bool(cfg.compile)}
        if self.data_parts:
            dp = np.array(self.data_parts[warm:] if len(self.data_parts) > warm else self.data_parts)
            split = {"rank": self.rank, "steps": int(len(dp)),
                     "plan_s_mean": float(dp[:, 0].mean()), "plan_s_p90": float(np.percentile(dp[:, 0], 90)),
                     "plan_s_max": float(dp[:, 0].max()), "read_wait_s_mean": float(dp[:, 1].mean()),
                     "read_wait_s_p90": float(np.percentile(dp[:, 1], 90)), "read_wait_s_max": float(dp[:, 1].max()),
                     "shard_prep_wait_s_total": float(dp[:, 2].sum()), "shard_prep_wait_s_max": float(dp[:, 2].max()),
                     "prep_stats": {k: v for o in self._prep_objects() for k, v in vars(o).get("prep_stats", {}).items()}}
            print(f"[daisy] data wait: {json.dumps(split)}", flush=True)
        np.save(self.out_dir / f"step_times_r{self.rank}.npy",
                np.array([step_times, data_times, after_times + [np.nan] * (len(step_times) - len(after_times))]))
        allrec = [None] * self.world
        if self.world > 1:
            dist.all_gather_object(allrec, rec)
        else:
            allrec = [rec]
        if self.master:
            with open(self.out_dir / "timing.json", "w") as f:
                json.dump({"per_rank": allrec,
                           "step_s_median_max_rank": max((r["step_s_median"] or 0) for r in allrec),
                           "peak_alloc_gib_max_rank": max((r["peak_alloc_gib"] or 0) for r in allrec)}, f, indent=1)
            print(f"[daisy] timing: {json.dumps(rec)}")
        self.stats["timing"] = rec
        return self.stats


def train_daisy(model, opt, scheduler, exp_dir, distributed_backend, cfg):
    """Entry point from train.py. -> stats (train / held-out loss, timing; {..., "already_done": step} for a run that
    had already finished)."""
    done = already_finished(cfg, exp_dir, distributed_backend.is_master_process())
    if done is not None:
        return {"train_loss": [], "target_loss": [], "already_done": done}
    return DaisyTrainer(model, opt, scheduler, exp_dir, distributed_backend, cfg).run()
