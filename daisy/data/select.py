"""Candidate window -> scoring -> selection: online data selection by a linear scorer on phi (paper Section 3.4).

SelectingStream wraps a document stream (the tokenized pool, or the index). For every micro-batch (B rows x (T+1)
tokens) it
  1. draws candidate documents in the stream's order until they hold `mult` (lambda) x the tokens needed (whole
     documents, capped at max_doc_tokens);
  2. fills the micro-batch with the candidates of highest score w^T [phi, 1]. With cosine=True the score is
     w^T phi / |phi| + b (the similarity baseline, with a fixed unit w); with K rows of weights (K, d + 1) the score is
     the maximum over the K scorers; with `dsir` set (the DSIR baseline) the score is the candidate's precomputed
     importance weight (daisy/scorers/dsir.py). Before the scorer has weights, the micro-batch is a uniformly random
     subset of the window;
  3. cuts the last chosen document (in selection order: the lowest-scored selected one) to fill the micro-batch
     exactly (its tail is dropped), and packs the chosen documents in random order.
Unchosen candidates are consumed (not revisited within a pass).
The weights come from rank 0's running ridge (broadcast to all ranks every U steps) and apply to the micro-batches
planned from then on (the prefetcher plans a few micro-batches ahead). Each weight version is recorded with the number
of micro-batches planned when it arrived, and a micro-batch is planned with the version current at its index; the
checkpoint keeps the versions that micro-batches planned but not yet trained on still need, so a resumed run re-plans
them with the same weights (exact resume).
phi of the candidates is read from the store's phi memmap (index stream) or, over the tokenized pool (ShardDocStream),
from RAM copies of the sidecars of the rank's open shards.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from daisy.data.stream import load_segments

PHI_DIM = 768


class SelectingStream:
    def __init__(self, base, mult, seed, rank=0, cosine=False):
        if mult < 1:
            raise ValueError("the window factor must be >= 1")
        # base: the index stream (UniformDocStream: a fixed order over the index, phi in a memmap) or the pool stream
        # (ShardDocStream over the tokenized pool: candidates from its open shards, phi read from the shard sidecars)
        self.pool_mode = hasattr(base, "_next_doc")
        self.base, self.store = base, (base.pool if self.pool_mode else base.store)
        # pool mode: the phi of every OPEN shard is held in RAM (one sequential read of its sidecar), loaded in the
        # background; a rank only draws candidates from its open shards. With the base stream's preload > 0, the phi of
        # its upcoming shards (base.upcoming) is read in the background too, so that a shard's phi is usually in RAM
        # when it opens and the planning thread does not wait for the read. Reads only: the candidates, scores and
        # picks are unchanged. prep_stats: shards whose phi was ready, waited for, or read cold, and the planning
        # thread's time waiting.
        self._phi_io = ThreadPoolExecutor(4, thread_name_prefix="daisy-phi") if self.pool_mode else None
        self._phi_ram = {}
        self.prep_stats = {"phi_sync_s": 0.0, "phi_ready": 0, "phi_waited": 0, "phi_cold": 0}
        if self.pool_mode:
            self._phi_want()
        self.T, self.cap = base.T, base.cap
        self.mult, self.cosine = float(mult), bool(cosine)
        # the DSIR baseline (daisy/scorers/dsir.py DsirScores, set by build_stream; pool mode): its per-document scores
        # replace w^T phi, and the weights only switch the scorer on
        self.dsir = None
        self.rng = np.random.default_rng([seed, rank, 7919])
        self.seed, self.rank = seed, rank
        # index mode: this rank's first-pass order (later passes reshuffle it)
        self.order0 = None if self.pool_mode else np.array(base.order, copy=True)
        self.w = None
        self.n_planned = 0         # micro-batches planned so far
        self.w_hist = []           # [(n_planned when the version arrived, w)], oldest first
        self.stats = {"microbatches": 0, "candidates": 0, "chosen": 0, "random_tokens": 0, "selected_tokens": 0}

    def set_weights(self, w):
        self.w = None if w is None else np.asarray(w, dtype=np.float32)
        self.w_hist.append((self.n_planned, self.w))
        del self.w_hist[:-256]     # a pending plan never needs more than the last few versions

    def _w_at(self, n):
        """The weights a micro-batch planned as number n is scored with: the last version that arrived at or before n."""
        w = None
        for n_at, w_ in self.w_hist:
            if n_at > n:
                break
            w = w_
        return w

    def hist_state(self, since):
        """The weight versions needed to re-plan micro-batches from number `since` on (JSON-serializable)."""
        keep = [i for i, (n_at, _) in enumerate(self.w_hist) if n_at > since]
        first = (keep[0] if keep else len(self.w_hist)) - 1           # plus the version current at `since`
        return [[n_at, None if w is None else w.tolist()] for n_at, w in self.w_hist[max(first, 0):]]

    def _pass_order(self, epoch):
        """Index mode: this rank's candidate order for pass `epoch` (0 = the stream's own order; later passes reshuffle
        it with a deterministic seed, so a resumed run rebuilds the same order from the epoch number alone)."""
        if epoch == 0:
            return self.order0
        return self.order0[np.random.default_rng([self.seed, self.rank, 104729, epoch]).permutation(len(self.order0))]

    def _next_row(self):
        if self.pool_mode:
            return int(self.base._next_doc())
        if self.base.pos >= len(self.base.order):
            # a selecting run draws mult x more candidates than it trains on, so a long run can use up this rank's
            # share of the index: start another pass over it in a fresh order (documents are seen again as candidates)
            self.base.epoch += 1
            self.base.order = self._pass_order(self.base.epoch)
            self.base.pos = 0
            self.stats["passes"] = self.base.epoch
            print(f"[select] rank {self.rank}: candidates used up, pass {self.base.epoch} starts", flush=True)
        r = int(self.base.order[self.base.pos])
        self.base.pos += 1
        return r

    def plan(self, B):
        w_now = self.w
        self.w = self._w_at(self.n_planned) if self.w_hist else self.w
        try:
            out = self._plan(B)
            if self.pool_mode:
                self._phi_want()
            return out
        finally:
            self.w = w_now
            self.n_planned += 1

    def _plan(self, B):
        """One micro-batch of B rows: candidate window -> top scores (a random subset while the scorer is off) -> cut
        and shuffle. -> (segments [(key, 0, take)], flags: 1 = picked at random, 0 = picked by score)."""
        need = B * (self.T + 1)
        cands, lens = self._draw_window(need)
        chosen, flags, used = [], [], np.zeros(len(cands), bool)
        # the random order is drawn for every micro-batch, whether or not it is used, so that the generator (which also
        # shuffles the packing order below) advances in the same way before and after the scorer starts
        order = self.rng.permutation(len(lens))
        have = 0
        if self.w is None:
            have = self._take_random(order, lens, used, chosen, flags, need)
        else:
            have = self._take_top_scores(cands, lens, used, chosen, flags, have, need)
        while have < need:                                           # rare: the window ran short after long cuts
            r = self._next_row()
            cands, lens = np.append(cands, r), np.append(lens, self.base._length(r))
            chosen.append(len(cands) - 1)
            flags.append(1)
            have += int(lens[-1])
        return self._cut_and_shuffle(cands, lens, chosen, flags, need)

    def _draw_window(self, need):
        """The candidate window: documents in the stream's order until they hold mult x need tokens (capped lengths)."""
        cands, lens, tok = [], [], 0
        while tok < self.mult * need:
            r = self._next_row()
            L = self.base._length(r)
            cands.append(r)
            lens.append(L)
            tok += L
        return np.array(cands, dtype=np.int64), np.array(lens, dtype=np.int64)

    def _take_random(self, order, lens, used, chosen, flags, need):
        """Before the scorer starts: candidates in the random order until they hold `need` tokens."""
        have = 0
        for i in order:
            if have >= need:
                break
            chosen.append(int(i))
            flags.append(1)
            used[i] = True
            have += int(lens[i])
        return have

    def _scores(self, phi):
        """s = max over the K weight rows of w^T phi + b (or w^T phi / |phi| + b with cosine)."""
        W = self.w if self.w.ndim == 2 else self.w[None]        # (K, d + 1): K scorers, score = max over them
        s = phi @ W[:, :-1].T
        if self.cosine:                                         # the similarity baseline: w^T phi / |phi|
            s = s / np.maximum(np.linalg.norm(phi, axis=1), 1e-12)[:, None]
        return (s + W[:, -1]).max(1)

    def _take_top_scores(self, cands, lens, used, chosen, flags, have, need):
        """Fill the micro-batch with the highest-scored candidates."""
        if self.dsir is not None:
            s = self.dsir.scores(cands)
            self.dsir.keep_only({int(o[0]) for o in self.base.open} | {int(c) for c in np.unique(cands >> 32)})
        else:
            s = self._scores(self._cand_phi(cands))
        for i in np.argsort(-s, kind="stable"):
            if have >= need:
                break
            if used[i]:
                continue
            chosen.append(int(i))
            flags.append(0)
            used[i] = True
            have += int(lens[i])
        return have

    def _cut_and_shuffle(self, cands, lens, chosen, flags, need):
        """Takes in SELECTION order: only the last chosen document (the lowest score among the selected) is cut to fill
        exactly; then the packing order is shuffled."""
        takes, acc = [], 0
        for i in chosen:
            take = int(min(lens[i], need - acc))
            takes.append(take)
            acc += take
        order = self.rng.permutation(len(chosen))
        segs = [(int(cands[chosen[j]]), 0, takes[j]) for j in order if takes[j] > 0]
        fl = np.array([flags[j] for j in order if takes[j] > 0], dtype=np.int8)
        st = self.stats
        st["microbatches"] += 1
        st["candidates"] += len(cands)
        st["chosen"] += len(segs)
        st["random_tokens"] += int(sum(e - s for (_, s, e), f in zip(segs, fl) if f))
        st["selected_tokens"] += int(sum(e - s for (_, s, e), f in zip(segs, fl) if not f))
        return segs, fl

    def _load_shard_phi(self, sh, chunk=64 << 20):
        """The shard's whole phi sidecar, (n, 768) fp16: the same bytes np.fromfile reads, read in 64 MB chunks with
        FileIO.readinto, which releases the GIL during each read (the planning thread keeps running)."""
        path = os.path.join(self.store.root, "sidecar", f"{sh:05d}.phi.f16")
        n = os.path.getsize(path)
        if n % (2 * PHI_DIM):
            raise ValueError(f"{path}: {n} bytes is not a whole number of {PHI_DIM}-dim fp16 rows")
        out = np.empty(n // 2, dtype=np.float16)
        buf = memoryview(out).cast("B")
        with open(path, "rb", buffering=0) as f:
            pos = 0
            while pos < n:
                k = f.readinto(buf[pos:pos + chunk])
                if not k:
                    raise OSError(f"{path}: read ended at byte {pos} of {n}")
                pos += k
        return out.reshape(-1, PHI_DIM)

    def _phi_want(self):
        """pool mode: keep the phi of the open shards, and of the base stream's upcoming shards (preload), in RAM,
        starting the missing reads in the background; free the others."""
        want = [int(s) for s, _ in self.base.open]
        if getattr(self.base, "preload", 0) > 0:
            want += self.base.upcoming(self.base.preload_count())
        for s in want:
            if s not in self._phi_ram:
                self._phi_ram[s] = self._phi_io.submit(self._load_shard_phi, s)
        keep = set(want)
        for s in [s for s in self._phi_ram if s not in keep]:
            del self._phi_ram[s]

    def _shard_phi(self, sh):
        v = self._phi_ram.get(sh)
        if isinstance(v, np.ndarray):
            return v
        t = time.perf_counter()
        if v is None:                                   # opened during this plan and not prepared: read it now
            self.prep_stats["phi_cold"] += 1
            v = self._load_shard_phi(sh)
        else:                                           # read in the background: ready, or wait for it
            self.prep_stats["phi_ready" if v.done() else "phi_waited"] += 1
            v = v.result()
        self._phi_ram[sh] = v
        self.prep_stats["phi_sync_s"] += time.perf_counter() - t
        return v

    def _cand_phi(self, cands):
        """phi (n, d) float32 of the candidates: index rows from the memmap, or pool keys from the RAM copies of their
        open shards' sidecars."""
        if self.pool_mode:
            out = np.empty((len(cands), PHI_DIM), dtype=np.float32)
            sh, row = cands >> 32, cands & 0xFFFFFFFF
            for s_ in np.unique(sh):
                m = sh == s_
                out[m] = self._shard_phi(int(s_))[row[m]]
            return out                                      # (plan frees the copies of closed shards: _phi_want)
        return np.asarray(self.store.phi_rows[cands], dtype=np.float32)

    def load(self, plan, pool=None):
        segs, _ = plan
        return load_segments(self.store, segs, self.T, self.cap, pool)

    def next_microbatch(self, B):
        return self.load(self.plan(B))

    def state_dict(self):
        return {"base": self.base.state_dict(), "rng": self.rng.bit_generator.state, "selecting": True,
                "n_planned": self.n_planned, "w_hist": self.hist_state(self.n_planned)}

    def load_state_dict(self, st):
        self.base.load_state_dict(st["base"])
        if self.pool_mode:                               # the phi of the restored open (and upcoming) shards
            self._phi_want()
        if not self.pool_mode and self.base.epoch > 0:
            self.base.order = self._pass_order(self.base.epoch)
        self.rng.bit_generator.state = st["rng"]
        self.n_planned = int(st["n_planned"])
        self.w_hist = [(int(n_at), None if w is None else np.asarray(w, dtype=np.float32)) for n_at, w in st["w_hist"]]
        self.w = self.w_hist[-1][1] if self.w_hist else None

    def __getattr__(self, name):
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)
