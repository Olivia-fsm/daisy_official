"""The MATES stage controller: the stage boundaries and the GPU-hour budget of a --mates run
(daisy/train/trainer.py calls it; docs/mates_baseline.md is the specification).

Hooks, at step t before that step's training:
  budget_spent(t)   at refresh steps: the running GPU-hour total (the same on every rank: all_reduce MAX) reached
                    --mates_budget; the trainer then stops like a finished run (final checkpoint, evaluation,
                    finished.json with "reason": "budget")
  boundary_due(t)   t = F + (k - 1) U (k >= 1; F = --mates_first_boundary, default U; t < --iterations) and the
                    stream still plays stage k
  boundary(t)       selects stage s = k + 1 (the global RNG states are restored afterwards, so training does not
                    depend on whether a boundary ran in this job or an earlier one):
     1. probes      this rank's probe documents -> Prober (no collectives) -> stage_s/probes_r{rank}.npz; all ranks'
                    probes gathered; o = L_r(before) - L_r(after)
     2. fine-tune   the influence model: pretrained BERT at the first boundary, stage s-1's best model after that
                    (the paper; the released code re-initializes), on this stage's probes only (DDP over the ranks)
     3. candidates  enumerate_candidates on the rank's live pool stream until 1 / ratio x the stage's token budget
     4. scoring     EncodePool (CPU workers: read, decode, WordPiece) feeding the Scorer (bucketed bf16 BERT)
     5. selection   gumbel_top_k by tokens (tau), seeded by (seed, s, rank) -> stage_s/select_r{rank}.npz / .json
     6. switch      the Prefetcher's read-ahead is dropped, MatesStream plays the selected list
     7. records     stages.jsonl, stage_costs.jsonl (rank 0); the trainer then writes a checkpoint
  on_step(dt)       a training step time for the steady s/step (steps over SPIKE_S = 2 s, such as the compilation
                    steps, are left out)
Artifacts that exist are reused (a boundary cut by the end of a job resumes where it stopped); the seconds recorded with
them still count. GPU-hours = offset (--mates_cost_offset; < 0 with --mates_init_from: the source run's steady s/step x
its steps x its GPUs, stage1_cost) + steps trained in this run x steady s/step x world / 3600 + every boundary's
component wall seconds (max over ranks) x world / 3600. budget.json is rewritten at every check.
"""

import json
import math
import os
import random
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from daisy.data.prefetch import Prefetcher
from daisy.mates import influence, text
from daisy.mates.artifacts import MatesDir, load_json, read_jsonl, save_json, save_npz
from daisy.mates.probes import Prober, per_rank, probe_documents, reference_batches
from daisy.mates.select import gumbel_top_k
from daisy.mates.streams import enumerate_candidates
from daisy.runmeta import append_line, retry_io
from daisy.tok import gpt2_tokenizer_json

SPIKE_S = 2.0          # step times above this are left out of the steady s/step
COMPONENTS = ("probe", "gather", "finetune", "enumerate", "score", "select")


@contextmanager
def rng_preserved(cuda_device=None):
    st = (torch.random.get_rng_state(), np.random.get_state(), random.getstate(),
          torch.cuda.get_rng_state(cuda_device) if cuda_device is not None else None)
    try:
        yield
    finally:
        torch.random.set_rng_state(st[0])
        np.random.set_state(st[1])
        random.setstate(st[2])
        if st[3] is not None:
            torch.cuda.set_rng_state(st[3], cuda_device)


def stage1_cost(init_dir, steps):
    """GPU-hours of the source run's first `steps` steps: its steady s/step (step_times_r0.npy of its last job, the
    first min(50, n/4) steps and steps over SPIKE_S left out) x steps x its GPUs (timing.json: one record per rank)."""
    comp = Path(init_dir).resolve().parent.parent / "daisy"
    if not (comp / "step_times_r0.npy").exists() or not (comp / "timing.json").exists():
        raise FileNotFoundError(f"{comp}: no step_times_r0.npy / timing.json for the stage-1 cost; pass "
                                f"--mates_cost_offset (GPU-hours) instead")
    step, _, hook = np.load(comp / "step_times_r0.npy")
    warm = min(50, len(step) // 4)
    t = step[warm:] + np.nan_to_num(hook[warm:])
    s_per_step = float(t[t <= SPIKE_S].mean())
    with open(comp / "timing.json") as f:
        gpus = len(json.load(f)["per_rank"])
    return steps * s_per_step * gpus / 3600, {"source": str(comp), "s_per_step": s_per_step, "gpus": gpus,
                                               "steps": int(steps), "steps_recorded": int(len(step))}


class MatesController:
    def __init__(self, tr):
        cfg = self.cfg = tr.cfg
        self.tr = tr
        self.U = int(cfg.mates_stage_steps)
        if self.U <= 0:
            raise ValueError("--mates_stage_steps must be > 0")
        self.F = int(cfg.mates_first_boundary) or self.U
        if self.F < 0:
            raise ValueError("--mates_first_boundary must be >= 0")
        if not 0 < cfg.mates_ratio <= 1:
            raise ValueError("--mates_ratio must be in (0, 1]")
        if not cfg.daisy_target_dir or not cfg.mates_bert_dir:
            raise ValueError("--mates needs --daisy_target_dir (the target tasks of the reference loss) and "
                             "--mates_bert_dir (bert-base-uncased)")
        self.dir = MatesDir(tr.out_dir / "mates")
        self.world, self.rank, self.master = tr.world, tr.rank, tr.master
        self.device = cfg.device
        self.cuda = "cuda" in str(cfg.device)
        self.log = print if self.master else (lambda *a, **k: None)
        self.budget_tokens = self.U * cfg.batch_size * cfg.acc_steps * (cfg.sequence_length + 1)
        self.prober = self.influence = self.pretrained = None
        # cost state (checkpointed)
        self.init_iter, self.offset, self.offset_info = None, None, None
        self.steady_sum, self.steady_n = 0.0, 0
        self.boundary_gpu_s = {}
        self.boundary_parts = {}       # stage -> {component: GPU-seconds}: the next boundary's cost estimate
        self.stop_estimate = None      # set when the run stops before a boundary it could not afford

    # ------------------------------------------------------------------ state
    def state_dict(self):
        return {"init_iter": self.init_iter, "offset": self.offset, "offset_info": self.offset_info,
                "steady_sum": self.steady_sum, "steady_n": self.steady_n,
                "boundary_gpu_s": {str(k): v for k, v in self.boundary_gpu_s.items()},
                "boundary_parts": {str(k): v for k, v in self.boundary_parts.items()}}

    def load_state_dict(self, st):
        self.init_iter, self.offset, self.offset_info = st["init_iter"], st["offset"], st["offset_info"]
        self.steady_sum, self.steady_n = float(st["steady_sum"]), int(st["steady_n"])
        self.boundary_gpu_s = {int(k): float(v) for k, v in st["boundary_gpu_s"].items()}
        self.boundary_parts = {int(k): dict(v) for k, v in st["boundary_parts"].items()}

    def start(self, it, init_dir=None):
        """At the start of a job (after any resume): the run's first step and the cost offset, fixed once per run."""
        if self.init_iter is None:
            self.init_iter = int(it)
        if self.offset is None:
            if self.cfg.mates_cost_offset >= 0:
                self.offset, self.offset_info = float(self.cfg.mates_cost_offset), {"source": "--mates_cost_offset"}
            elif init_dir:
                self.offset, self.offset_info = stage1_cost(init_dir, self.init_iter)
            else:
                self.offset, self.offset_info = 0.0, {"source": "no stage-1 run"}
            self.log(f"[mates] cost offset {self.offset:.3f} GPU-h ({self.offset_info})")

    def stage_at(self, it):
        """The stage the boundary at step it selects (2 at F, 3 at F + U, ...), None when no boundary falls on it."""
        if it < self.F or (it - self.F) % self.U:
            return None
        return (it - self.F) // self.U + 2

    def _ms(self):
        s = self.tr.stream
        return s.stream if isinstance(s, Prefetcher) else s

    # ------------------------------------------------------------------ hooks
    def on_step(self, dt):
        if dt <= SPIKE_S:
            self.steady_sum += float(dt)
            self.steady_n += 1

    def gpu_hours(self, it):
        steps = int(it) - int(self.init_iter)
        sps = self.steady_sum / self.steady_n if self.steady_n else 0.0
        train = steps * sps * self.world / 3600
        bnd = sum(self.boundary_gpu_s.values()) / 3600
        return self.offset + train + bnd, {"offset": self.offset, "training": train, "boundaries": bnd,
                                           "steps_this_run": steps, "s_per_step": sps, "steady_steps_timed":
                                           self.steady_n}

    def boundary_estimate(self, s):
        """Expected GPU-hours of the boundary that selects stage s: the previous boundary's measured GPU-seconds per
        component, probing and fine-tuning scaled by the probe counts (candidates and scoring are the same size every
        stage). None at the first boundary (nothing measured yet)."""
        prev = self.boundary_parts.get(s - 1)
        if not prev:
            return None
        n = lambda j: self.cfg.mates_probes_first if j == 2 else self.cfg.mates_probes_later
        r = n(s) / n(s - 1)
        return sum(v * (r if k in ("probe", "finetune") else 1.0) for k, v in prev.items()) / 3600

    def budget_spent(self, it, force=False, before_boundary=False):
        """All ranks, at refresh steps (or force): writes budget.json; True when the GPU-hour budget is spent.
        before_boundary: also True when the boundary about to run is expected to spend the rest of the budget (its
        stage could not train a step within it: the run ends here with the loss it has at the budget, instead of
        paying for a selection it would never train on)."""
        cfg = self.cfg
        if not force and it % cfg.daisy_refresh != 0:
            return False
        total, parts = self.gpu_hours(it)
        est = self.boundary_estimate(self.stage_at(it)) if before_boundary and self.stage_at(it) else None
        t = torch.tensor([total, est or 0.0], dtype=torch.float64, device=self.device)
        if self.world > 1:
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
        total, est_all = float(t[0].item()), float(t[1].item())
        spent = cfg.mates_budget > 0 and total >= cfg.mates_budget
        if before_boundary and est is not None and cfg.mates_budget > 0 and not spent \
                and total + est_all >= cfg.mates_budget:
            spent, self.stop_estimate = True, est_all
            parts = {**parts, "next_boundary_estimate_gpu_h": est_all}
        if self.master:
            save_json(self.dir.run_file("budget.json"),
                      {"step": int(it), "gpu_hours": total, "budget": cfg.mates_budget, "spent": bool(spent),
                       "world": self.world, "parts_rank0": parts, "offset_info": self.offset_info,
                       "boundaries_gpu_s": {str(k): v for k, v in sorted(self.boundary_gpu_s.items())}})
            if spent:
                print(f"[mates] GPU-hour budget spent at iter {it}: {total:.3f} >= {cfg.mates_budget} ({parts})")
        return spent

    def finish_record(self, it):
        rec = {"reason": "budget", "gpu_hours": self.gpu_hours(it)[0], "budget": self.cfg.mates_budget}
        if self.stop_estimate is not None:
            rec["stopped_before_boundary"] = {"stage": self.stage_at(it), "estimate_gpu_h": self.stop_estimate}
        return rec

    def boundary_due(self, it):
        s = self.stage_at(it)
        return it > 0 and s is not None and it < self.cfg.iterations and self._ms().stage < s

    def stop_margin(self, it):
        """Seconds a chained job must have left to start a boundary at step it (else it stops before it)."""
        return 60.0 * self.cfg.mates_boundary_reserve_min if self.boundary_due(it) else 0.0

    # ------------------------------------------------------------------ the boundary
    def boundary(self, it):
        s = self.stage_at(it)
        self.log(f"[mates] stage boundary at iter {it}: selecting stage {s}")
        dev = torch.device(self.device) if self.cuda else None
        wall, reused = {}, []
        with rng_preserved(dev):
            pf = self.tr.stream
            if isinstance(pf, Prefetcher):
                pf.load_state_dict(pf.state_dict())       # drop the read-ahead: back to the last batch handed out
            ms = self._ms()
            if self.cuda:
                torch.cuda.empty_cache()
            self._enc, tok_cpu_s = None, 0.0
            try:
                probes = self._probes(s, it, wall, reused)
                model, ft = self._finetune(s, probes, wall, reused)
                sel = self._select(s, model, wall, reused)
            finally:
                if self._enc is not None:
                    tok_cpu_s = self._enc.cpu_s
                    self._enc.close()
                    self._enc = None
            ms.start_stage(s, sel["sel_key"], sel["sel_end"])
            if isinstance(pf, Prefetcher):
                pf.load_state_dict(ms.state_dict())
            model.zero_grad(set_to_none=True)
            if self.cuda:
                torch.cuda.empty_cache()
            self._records(s, it, probes, ft, sel, wall, reused, tok_cpu_s)

    def _encoder(self):
        """The rank's CPU worker pool (started on first use: a boundary that reuses its artifacts starts none)."""
        if self._enc is None:
            self._enc = text.EncodePool(self.cfg.daisy_pool, self.cfg.daisy_max_doc_tokens,
                                        gpt2_tokenizer_json(self.cfg.mates_gpt2_tokenizer), self.cfg.mates_bert_dir,
                                        workers=self._workers())
        return self._enc

    def _workers(self):
        w = int(self.cfg.mates_tok_workers)
        if w >= 0:
            return w
        try:
            n = len(os.sched_getaffinity(0))
        except AttributeError:
            n = os.cpu_count() or 2
        local = int(os.environ.get("LOCAL_WORLD_SIZE", min(self.world, 4)))
        return max(1, n // max(1, local) - 2)

    def _max(self, values):
        """Element-wise max over ranks of a list of floats."""
        t = torch.tensor(values, dtype=torch.float64, device=self.device)
        if self.world > 1:
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return t.tolist()

    def _gather(self, obj):
        if self.world == 1:
            return [obj]
        out = [None] * self.world
        dist.all_gather_object(out, obj)
        return out

    # 1. probes
    def _probe_counts(self, s):
        cfg = self.cfg
        return {j: per_rank(cfg.mates_probes_first if j == 2 else cfg.mates_probes_later, self.world, self.rank)
                for j in range(2, s + 1)}

    def _probes(self, s, it, wall, reused):
        cfg, tr = self.cfg, self.tr
        path = self.dir.probes(s, self.rank)
        lr = float(tr.opt.param_groups[0]["lr"])
        if os.path.exists(path):
            z = np.load(path)
            mine = {k: z[k] for k in z.files}
            reused.append("probe")
        else:
            t0 = time.perf_counter()
            base = self._ms().base
            keys = probe_documents(base.pool, base.reserved, base.held_text, base.held_url, tr.stream_seed, s,
                                   self.rank, self._probe_counts(s))
            docs = [base.pool.doc(int(k), base.cap) for k in keys]
            if self.prober is None:
                tasks = [t for t in str(cfg.daisy_targets).split(",") if t.strip()]
                ref, n_tasks, info = reference_batches({t: os.path.join(cfg.daisy_target_dir, t) for t in tasks},
                                                       cfg.sequence_length, cfg.daisy_target_tokens,
                                                       cfg.mates_ref_rows_per_batch)
                self.log(f"[mates] reference: train-split target rows {info}")
                self.prober = Prober(tr.raw_model, tr.opt, tr.layers, ref, n_tasks, cfg.sequence_length,
                                     cfg.grad_clip, cfg.mates_probe_precision, self.device, compile=cfg.compile)
            before, res = self.prober.run(docs, log=print if self.master else None,
                                          every=max(1, len(docs) // 10))
            mine = {"key": keys, "length": np.array([len(d) for d in docs], dtype=np.int32),
                    "grad_norm": res["grad_norm"], "clipped": res["clipped"], "loss_after": res["loss_after"],
                    "doc_loss": res["doc_loss"], "seconds": res["seconds"], "loss_before": np.float64(before),
                    "lr": np.float64(lr), "wall_s": np.float64(time.perf_counter() - t0)}
            save_npz(path, **mine)
        wall["probe"] = float(mine["wall_s"])
        t0 = time.perf_counter()
        parts = self._gather({k: mine[k] for k in ("key", "length", "grad_norm", "clipped", "loss_after", "seconds",
                                                   "loss_before")})
        out = {k: np.concatenate([p[k] for p in parts]) for k in ("key", "length", "grad_norm", "clipped",
                                                                  "loss_after", "seconds")}
        out["loss_before_ranks"] = [float(p["loss_before"]) for p in parts]
        out["oracle"] = np.concatenate([float(p["loss_before"]) - p["loss_after"] for p in parts])
        out["lr"] = float(mine["lr"])
        wall["gather"] = time.perf_counter() - t0
        if self.master and not os.path.exists(self.dir.probe_meta(s)):
            save_json(self.dir.probe_meta(s),
                      {"stage": s, "step": int(it), "loss_before": out["loss_before_ranks"][0],
                       "loss_before_ranks": out["loss_before_ranks"], "lr": out["lr"],
                       "precision": cfg.mates_probe_precision, "clip": cfg.grad_clip, "n": int(len(out["key"])),
                       "per_rank": [int(len(p["key"])) for p in parts], "reserved_shards_rank0":
                       self._ms().base.reserved, "compile": bool(cfg.compile)})
        o = out["oracle"]
        self.log(f"[mates] stage {s}: {len(o)} probes at lr {out['lr']:.3e}: oracle mean {o.mean():.3e}, std "
                 f"{o.std():.3e}, positive {float((o > 0).mean()):.3f}, clipped {float(out['clipped'].mean()):.3f}")
        return out

    # 2. the influence model
    def _model(self, weights):
        """The run's one InfluenceModel object (its compiled graphs serve every boundary): built once from
        --mates_bert_dir with the head seeded identically on every rank; weights = None: those pretrained weights and
        that head (the first boundary), else a stage's bert/ dir (its best model)."""
        if self.influence is None:
            torch.manual_seed(1000003 * (self.tr.base_seed + 1))
            self.influence = influence.InfluenceModel(influence.build_bert(self.cfg.mates_bert_dir)).to(self.device)
            self.pretrained = {k: v.detach().to("cpu", copy=True) for k, v in self.influence.state_dict().items()}
        sd = self.pretrained if weights is None else torch.load(os.path.join(weights, "model.pt"), map_location="cpu",
                                                                 weights_only=True)
        self.influence.load_state_dict(sd)
        return self.influence

    def _bert_ctx(self):
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if self.cuda else nullcontext()

    def _scorer_kw(self):
        return {"bucket": self.cfg.mates_score_bucket, "tokens": self.cfg.mates_score_tokens,
                "padded": bool(self.cfg.mates_score_padded), "compile": bool(self.cfg.compile)}

    def _finetune(self, s, probes, wall, reused):
        cfg = self.cfg
        out_dir, done = self.dir.bert(s), self.dir.bert_done(s)
        if os.path.exists(done):
            ft = load_json(done)
            model = self._model(out_dir)
            reused.append("finetune")
            wall["finetune"] = float(ft["seconds"])
            return model, ft
        t0 = time.perf_counter()
        prev = self.dir.bert(s - 1) if s > 2 and os.path.exists(self.dir.bert_done(s - 1)) else None
        if s > 2 and prev is None:
            raise FileNotFoundError(f"stage {s}: stage {s - 1}'s influence model is missing ({self.dir.bert(s - 1)})")
        model = self._model(prev)
        seqs = self._encoder().encode_keys(probes["key"])
        log_path = self.dir.bert_log(s)
        if self.master and os.path.exists(log_path):
            os.remove(log_path)                        # records of an interrupted fine-tuning
        ddp = None
        if self.world > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP

            dev_ids = [torch.device(self.device).index] if self.cuda else None
            ddp = (lambda m: DDP(m, device_ids=dev_ids))
        torch.manual_seed(1000003 * (self.tr.base_seed + 1) + 7919 * s + self.rank)       # dropout masks
        summary, _ = influence.finetune(
            model, seqs, probes["oracle"], world=self.world, rank=self.rank, device=self.device, ctx=self._bert_ctx(),
            scorer_kw=self._scorer_kw(), epochs=cfg.mates_bert_epochs, batch=cfg.mates_bert_batch,
            micro=cfg.mates_bert_micro_batch, lr=cfg.mates_bert_lr, wd=cfg.mates_bert_wd,
            val_share=cfg.mates_bert_val_share, eval_every=cfg.mates_bert_eval_every, seed=1000003 * s + 17,
            ddp=ddp, compile=bool(cfg.compile), log=self.log,
            on_record=(lambda r: append_line(log_path, json.dumps({"stage": s, **r}))) if self.master else None)
        summary["init"] = prev or cfg.mates_bert_dir
        summary["seconds"] = time.perf_counter() - t0
        if self.master:
            os.makedirs(out_dir, exist_ok=True)
            sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            retry_io(lambda: torch.save(sd, os.path.join(out_dir, "model.pt.tmp")), f"save {out_dir}/model.pt")
            os.replace(os.path.join(out_dir, "model.pt.tmp"), os.path.join(out_dir, "model.pt"))
            model.bert.config.to_json_file(os.path.join(out_dir, "config.json"))
            save_json(done, summary)
        if self.world > 1:
            dist.barrier()
        wall["finetune"] = summary["seconds"]
        return model, summary

    # 3.-5. candidates, scoring, selection
    def _select(self, s, model, wall, reused):
        cfg = self.cfg
        npz, js = self.dir.select_npz(s, self.rank), self.dir.select_json(s, self.rank)
        base = self._ms().base
        if os.path.exists(js):
            meta = load_json(js)
            z = np.load(npz)
            base.load_state_dict(meta["stream_after"])
            reused.append("select")
            for k in ("enumerate", "score", "select"):
                wall[k] = float(meta["seconds"][k])
            return {**{k: z[k] for k in z.files}, "meta": meta}
        t0 = time.perf_counter()
        keys, lens = enumerate_candidates(base, int(round(self.budget_tokens / cfg.mates_ratio)))
        t1 = time.perf_counter()
        scores, sstats = self._score(model, keys, self._encoder())
        t2 = time.perf_counter()
        rng = np.random.default_rng([self.tr.stream_seed, s, self.rank, 7])
        g = gumbel_top_k(scores, lens, self.budget_tokens, cfg.mates_tau, rng)
        arrays = {"cand_key": keys, "cand_len": lens.astype(np.int32), "score": scores.astype(np.float32),
                  "gumbel": g["gumbel"].astype(np.float32), "selected": g["selected"],
                  "sel_key": keys[g["order"]], "sel_end": g["ends"].astype(np.int64)}
        save_npz(npz, **arrays)
        sel = g["selected"]
        stats = {"n_candidates": int(len(keys)), "candidate_tokens": int(lens.sum()), "selected_docs": int(sel.sum()),
                 "selected_tokens": int(g["ends"].sum()), "score_sum": float(scores.sum()),
                 "score_sumsq": float((scores ** 2).sum()), "sel_score_sum": float(scores[sel].sum()),
                 "cut_docs": int((g["ends"] < lens[g["order"]]).sum()), **sstats}
        seconds = {"enumerate": t1 - t0, "score": t2 - t1, "select": time.perf_counter() - t2}
        meta = {"stage": s, "stream_after": base.state_dict(), "seconds": seconds, "stats": stats}
        save_json(js, meta)
        wall.update(seconds)
        return {**arrays, "meta": meta}

    def _score(self, model, keys, enc, chunk=512):
        order = np.argsort(keys, kind="stable")        # read each shard's documents in file order
        chunks = [order[i:i + chunk] for i in range(0, len(order), chunk)]
        wait = [0.0]

        def blocks():
            it = enc.map(keys[c] for c in chunks)
            for c in chunks:
                t0 = time.perf_counter()
                flat, lens = next(it)
                wait[0] += time.perf_counter() - t0
                yield c, text.split_flat(flat, lens)

        scorer = influence.Scorer(model, self.device, self._bert_ctx(), **self._scorer_kw())
        scores = scorer.score(blocks(), len(keys))
        return scores, {"tokenize_wait_s": wait[0], **{f"bert_{k}": v for k, v in scorer.stats.items()}}

    # 7. records
    def _records(self, s, it, probes, ft, sel, wall, reused, tok_cpu_s):
        cfg = self.cfg
        mx = dict(zip(COMPONENTS, self._max([float(wall.get(k, 0.0)) for k in COMPONENTS])))
        st = sel["meta"]["stats"]
        parts = self._gather({k: st[k] for k in ("n_candidates", "candidate_tokens", "selected_docs",
                                                 "selected_tokens", "score_sum", "score_sumsq", "sel_score_sum",
                                                 "tokenize_wait_s")} | {"tok_cpu_s": tok_cpu_s})
        tot = {k: sum(p[k] for p in parts) for k in parts[0]}
        gpu_s = {k: v * self.world for k, v in mx.items()}
        self.boundary_gpu_s[s] = float(sum(gpu_s.values()))
        self.boundary_parts[s] = {k: float(v) for k, v in gpu_s.items()}
        if not self.master:
            return
        n = tot["n_candidates"]
        mean = tot["score_sum"] / n
        o = probes["oracle"]
        best = ft.get("best", {})
        rec = {"stage": s, "step": int(it), "probe_lr": probes["lr"], "precision": cfg.mates_probe_precision,
               "n_probes": int(len(o)), "loss_before": probes["loss_before_ranks"][0], "oracle_mean": float(o.mean()),
               "oracle_std": float(o.std()), "oracle_pos_share": float((o > 0).mean()),
               "clip_rate": float(np.mean(probes["clipped"])),
               "grad_norm_median": float(np.median(probes["grad_norm"])),
               "val_spearman": best.get("spearman"), "val_pearson": best.get("val_pearson"),
               "val_mse": best.get("val_mse"), "best_step": best.get("step"), "bert_steps": ft.get("steps"),
               "pred_mean": mean, "pred_std": math.sqrt(max(0.0, tot["score_sumsq"] / n - mean ** 2)),
               "sel_minus_cand_mean": tot["sel_score_sum"] / tot["selected_docs"] - mean,
               "n_candidates": int(n), "candidate_tokens": int(tot["candidate_tokens"]),
               "selected_docs": int(tot["selected_docs"]), "selected_tokens": int(tot["selected_tokens"]),
               "sel_doc_share": tot["selected_docs"] / n, "sel_token_share": tot["selected_tokens"] /
               tot["candidate_tokens"], "tau": cfg.mates_tau, "ratio": cfg.mates_ratio, "reused": reused}
        cost = {"stage": s, "step": int(it), "world": self.world, "wall_s": mx, "gpu_s": gpu_s,
                "total_wall_s": float(sum(mx.values())), "total_gpu_s": self.boundary_gpu_s[s],
                "gpu_hours": self.boundary_gpu_s[s] / 3600, "tokenize_cpu_s": tot["tok_cpu_s"],
                "tokenize_wait_s_sum": tot["tokenize_wait_s"], "probe_s_mean": float(np.mean(probes["seconds"])),
                "reused": reused}
        for name, r in (("stages.jsonl", rec), ("stage_costs.jsonl", cost)):
            path = self.dir.run_file(name)
            if not any(x.get("stage") == s for x in read_jsonl(path)):
                append_line(path, json.dumps(r))
        print(f"[mates] stage {s} selected: {rec['selected_docs']}/{n} documents ({rec['sel_token_share']:.3f} of the "
              f"candidate tokens), val spearman {rec['val_spearman']}, sel - cand mean score "
              f"{rec['sel_minus_cand_mean']:.4f}; boundary {cost['total_wall_s']:.0f} s wall = "
              f"{cost['gpu_hours']:.3f} GPU-h ({ {k: round(v, 1) for k, v in mx.items()} })")
        if cfg.wandb:
            import wandb

            wandb.log({"iter": int(it), **{f"mates/{k}": v for k, v in rec.items() if isinstance(v, (int, float))},
                       "mates/boundary_gpu_hours": cost["gpu_hours"]})
