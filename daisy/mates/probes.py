"""Oracle probing (MATES Alg. 1 step 1; docs/mates_baseline.md "Probe step").

probe_documents: one rank's probe documents for stage s, from its reserved pool shards (ShardDocStream never opens
them, so a probe is never a candidate), held-out documents excluded by the training stream's hashes (allowed_rows),
documents with fewer than 2 tokens left out (no target). Fresh per stage: stage s draws its documents uniformly without
replacement, seeded by (seed, s, rank), from the documents no earlier stage drew (earlier draws are replayed from their
seeds, so nothing but the seed needs to be stored).

reference_batches: the target tasks' TRAIN-split rows (daisy/labels/targets.py: the rows DAiSY's selection target is
built from, never the held-out val rows), all tasks in fixed-size batches with one topic per task. The reference loss
L_r is the mean over tasks of their mean token loss (DAiSY's combined target; validation_pass, no collectives).

Prober (one rank, no collectives; MATES's probe_oracle_data_influence.py reloads model + AdamW state from disk before
every probe, here a GPU snapshot is restored):
  snapshot   fp32 parameters and the optimizer state (AdamW moments and step count);
  probe(x)   the document in one (1, T) row (pads: ignored targets and their own document id, so attention is
             document-masked as in training); the document's mean token loss; backward on the UNWRAPPED model (no
             DDP all-reduce); the total gradient norm clipped at the training clip; optimizer.step() with the param
             groups as they stand at the boundary (the scheduler's lr of step t; the scheduler is not stepped); L_r
             after the step; the snapshot restored (bit-exact). The oracle is o = L_r(before) - L_r(after);
  precision  "bf16": the training autocast; "fp32": no autocast and TF32 off (document step and reference forward).
With --compile both forwards are compiled on fixed shapes ((1, T) and (rows, T)), so each compiles once.
"""

import copy
import time
from contextlib import contextmanager, nullcontext

import numpy as np
import torch

from daisy.data.pool import allowed_rows
from daisy.labels.gates import validation_pass
from daisy.labels.targets import target_batches

EOS = 50256


def per_rank(n, world, rank):
    """Rank `rank`'s share of n items split evenly over `world` ranks."""
    return n // world + (1 if rank < n % world else 0)


def probe_documents(pool, shards, held_text, held_url, seed, stage, rank, counts):
    """counts: {stage j: this rank's number of probes} for j = 2 .. stage. -> keys (int64) of stage `stage`."""
    avail = []
    for s in sorted(int(x) for x in shards):
        idx = allowed_rows(pool, s, held_text, held_url)[0]
        _, ends, _, starts = pool.shard(s)
        idx = idx[(ends - starts)[idx] >= 2]
        avail.append((np.int64(s) << np.int64(32)) | idx.astype(np.int64))
    remaining = np.concatenate(avail) if avail else np.zeros(0, np.int64)
    for j in range(2, stage + 1):
        n = int(counts[j])
        if n > len(remaining):
            raise ValueError(f"stage {j}: {n} probes requested, {len(remaining)} unused documents left in the reserved "
                             f"shards {sorted(shards)}")
        pick = np.random.default_rng([seed, j, rank, 31337]).choice(len(remaining), size=n, replace=False)
        if j == stage:
            return remaining[pick]
        remaining = np.delete(remaining, pick)
    raise ValueError(f"stage {stage}: probes start at stage 2")


def reference_batches(target_dirs, seq_len, target_tokens, rows_per_batch):
    """target_dirs: {task: train-split dir} (ordered). -> (batches of x, y, doc, topic = task index; n_tasks, info)."""
    xs, ys, docs, topics, info = [], [], [], [], {}
    for i, (task, d) in enumerate(target_dirs.items()):
        b, info[task] = target_batches(d, seq_len, target_tokens, rows_per_batch=1 << 30)   # one batch of all rows
        xs.append(b[0]["x"])
        ys.append(b[0]["y"])
        docs.append(b[0]["doc"])
        topics.append(torch.full_like(b[0]["x"], i))
    x, y, doc, topic = (torch.cat(v) for v in (xs, ys, docs, topics))
    batches = [{"x": x[i:i + rows_per_batch], "y": y[i:i + rows_per_batch], "doc": doc[i:i + rows_per_batch],
                "topic": topic[i:i + rows_per_batch]} for i in range(0, len(x), rows_per_batch)]
    return batches, len(target_dirs), info


def doc_row(doc, seq_len, eos=EOS):
    """One document (capped tokens, >= 2) as a (1, T) row: x, y (pads -1), doc ids (document 0, pads 1)."""
    doc = np.asarray(doc, dtype=np.int64)[: seq_len + 1]
    n = len(doc) - 1
    x = np.full((1, seq_len), eos, dtype=np.int64)
    y = np.full((1, seq_len), -1, dtype=np.int64)
    seg = np.ones((1, seq_len), dtype=np.int64)
    x[0, :n], y[0, :n], seg[0, :n] = doc[:-1], doc[1:], 0
    return x, y, seg


class Prober:
    def __init__(self, raw_model, optimizer, layers, ref_batches, n_tasks, seq_len, clip, precision, device,
                 compile=False):
        if precision not in ("bf16", "fp32"):
            raise ValueError(f"probe precision {precision!r}: bf16 or fp32")
        self.raw, self.opt, self.layers = raw_model, optimizer, layers
        self.ref, self.n_tasks, self.T = ref_batches, n_tasks, seq_len
        self.clip, self.precision, self.device = float(clip), precision, device
        seen, self.params = set(), []
        for g in optimizer.param_groups:
            for p in g["params"]:
                if id(p) not in seen:
                    seen.add(id(p))
                    self.params.append(p)
        self.cuda = "cuda" in str(device)
        self.ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if (self.cuda and precision == "bf16")
                    else nullcontext())
        # static shapes: the (1, T) probe and the (rows, T) reference batches each compile once, never as dynamic
        self.fwd = torch.compile(raw_model, dynamic=False) if compile else raw_model
        self.times = None          # set to a dict to time the parts of a probe: {part: seconds}; None = no timing
        self._p = self._st = None

    @contextmanager
    def numerics(self):
        """fp32: TF32 off for the duration (the trainer turns it on globally)."""
        if not (self.cuda and self.precision == "fp32"):
            yield
            return
        old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old

    # ---------------------------------------------------------------- state
    @torch.no_grad()
    def snapshot(self):
        self._p = [p.detach().clone() for p in self.params]
        self._st = [({k: (v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v))
                      for k, v in self.opt.state[p].items()} if p in self.opt.state else None) for p in self.params]

    @torch.no_grad()
    def restore(self):
        for p, s in zip(self.params, self._p):
            p.copy_(s)
        for p, st in zip(self.params, self._st):
            if st is None:
                self.opt.state.pop(p, None)
                continue
            cur = self.opt.state[p]
            for k in [k for k in cur if k not in st]:
                del cur[k]
            for k, v in st.items():
                c = cur.get(k)
                if torch.is_tensor(v) and torch.is_tensor(c) and c.shape == v.shape and c.dtype == v.dtype \
                        and c.device == v.device:
                    c.copy_(v)
                else:
                    cur[k] = v.clone() if torch.is_tensor(v) else copy.deepcopy(v)

    def release(self):
        self._p = self._st = None

    # ---------------------------------------------------------------- probes
    def _tick(self, part, t0):
        if self.times is None:
            return t0
        if self.cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        self.times[part] = self.times.get(part, 0.0) + t - t0
        return t

    def reference_loss(self):
        """L_r: the mean over target tasks of the mean token loss on their train-split rows."""
        with torch.no_grad(), self.numerics():
            _, ell, _ = validation_pass(self.fwd, self.layers, self.ref, self.n_tasks, self.ctx, world=1, rank=0,
                                        compute_grad=False, out_dtype=torch.float64)
        return float(ell.mean())

    def probe(self, doc):
        """One probe from the snapshot state. -> (L_r after the step, gradient norm before clipping, document loss)."""
        t0 = time.perf_counter()
        x, y, seg = (torch.from_numpy(a).to(self.device) for a in doc_row(doc, self.T))
        self.raw.train()
        with self.numerics():
            with self.ctx:
                loss = self.fwd(x, targets=y, doc_ids=seg)["loss"]
            loss.backward()
            t0 = self._tick("doc_step", t0)
            gnorm = torch.nn.utils.clip_grad_norm_(self.raw.parameters(), self.clip if self.clip > 0 else float("inf"))
            t0 = self._tick("clip", t0)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            t0 = self._tick("opt_step", t0)
        after = self.reference_loss()
        t0 = self._tick("reference", t0)
        self.restore()
        self._tick("restore", t0)
        return after, float(gnorm), float(loss.detach())

    def run(self, docs, log=None, every=1000):
        """Probe every document from the current state. -> (L_r before, dict of per-probe arrays)."""
        self.snapshot()
        try:
            before = self.reference_loss()
            out = {"loss_after": [], "grad_norm": [], "doc_loss": [], "seconds": []}
            t_all = time.perf_counter()
            for i, d in enumerate(docs):
                t0 = time.perf_counter()
                a, g, l = self.probe(d)
                out["loss_after"].append(a)
                out["grad_norm"].append(g)
                out["doc_loss"].append(l)
                out["seconds"].append(time.perf_counter() - t0)
                if log is not None and (i + 1) % every == 0:
                    log(f"[mates] probes {i + 1}/{len(docs)} ({(time.perf_counter() - t_all) / (i + 1):.3f} s/probe)")
        finally:
            self.release()
        res = {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}
        res["clipped"] = res["grad_norm"] > self.clip if self.clip > 0 else np.zeros(len(docs), dtype=bool)
        return before, res
