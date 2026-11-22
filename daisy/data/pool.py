"""Streaming document loader over the tokenized pool (daisy/tokenize/datatrove_tokenize.py).

Pool layout (root/): manifest.json (task rank -> input file), tokens/*_{rank:05d}*.ds + .ds.index (uint16 tokens,
uint64 cumulative document ends), sidecar/{rank:05d}.phi.f16 (standardized phi, 768 per doc) and
sidecar/{rank:05d}.meta.parquet (id, url, dump, text_hash, url_hash, ...), all in the same document order.

ShardDocStream (per rank):
  * shard order: one seeded permutation of all shards (identical on every rank), split rank::world;
  * `n_open` shards are open at a time; every document comes from a randomly drawn open shard, and each shard is
    read in its own seeded permutation of documents. A rank never sits in one crawl file for thousands of steps, so
    the stream is close to uniform across dumps; when a shard runs out, the next shard of the rank's list opens;
  * documents whose text or url hash matches a held-out document are skipped (exclusion by the same 64-bit hashes
    as the index's train_mask);
  * `excl_text` / `excl_url` (--daisy_pool_exclude): further documents never to train on, by the same hashes. They
    are removed from each shard's order after its permutation, so every other document keeps its place (the stream
    is the stream without the exclusion minus these documents);
  * `keep_dir` (--daisy_pool_keep): a per-document uint8 score of every shard (keep_dir/{rank:05d}.u8, one byte per
    document in shard order; e.g. the FineWeb-Edu int_score written by daisy/targets/edu_scores.py, 0 = not in the
    FineWeb-Edu score >= 2 set); only documents with score >= `keep_min` are drawn, the rest are skipped like held-out
    ones (the shard's document order is its seeded permutation of the allowed documents, as without a filter);
  * documents are capped at max_doc_tokens (EOS kept) and packed into micro-batches of exactly B x (T+1) tokens with
    carry-over (no token dropped), exactly like daisy/data/stream.py;
  * every segment carries its global key = shard << 32 | local index, and its phi row;
  * `reserve` > 0 (the MATES baseline, daisy/mates/): the last `reserve` shards of the rank's list are never opened;
    they hold the rank's probe documents (daisy/mates/probes.py), so a probe is never a training candidate;
  * `subset_tokens` > 0 (limited-pool runs, --daisy_pool_subset_tokens): the rank's subset is the documents its
    stream draws until their capped lengths reach subset_tokens, so pass 0 is exactly the stream without a subset, and
    a run resumed from a checkpoint of a run without a subset, taken inside pass 0, continues it unchanged until the
    subset is complete. Every later pass is a uniformly random permutation of the subset: all subset shards open at
    once, each in a fresh per-pass order of its subset documents, and a document is drawn from a shard with
    probability proportional to what the shard has left in the pass;
  * `reshuffle` (whole pool, --daisy_pool_reshuffle): when the rank's shards are used up, the stream starts another
    pass over them instead of stopping: pass p > 0 opens the rank's (usable) shards in a fresh seeded permutation of
    its list and reads each shard in a fresh seeded permutation of its documents (held-out still excluded), both
    seeded by the pass, so every document comes once per pass. Pass 0 is exactly the stream without it;
  * `preload` > 0 (--daisy_pool_preload): opening a shard (its index, its held-out filter from the meta parquet, its
    document order) is slow, and a selecting run with a large window factor lambda uses up shards quickly, so without
    preload the planning thread, and with it the training step, waits for every newly opened shard. With preload, the
    order of each of the next shards the stream will open (upcoming(): the rest of the pass's list, then the next
    pass's) is computed on a background thread before the stream needs it. The order is a pure function of
    (seed, shard), so the documents, their order and the state are exactly those of preload 0; only when they are
    computed changes. prep_stats: time the planning thread still waited for orders.
State (for exact resume): rank shard cursor, open shards with their cursors, RNG state, pending segment; with a subset
also the pass, the capped tokens pass 0 has drawn and, once pass 0 is complete, the subset ([shard, documents]).
"""

import collections
import glob
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from daisy.data.docstore import _pread_capped, _pread_exact
from daisy.data.stream import load_segments

DIM = 768


class DsPool:
    def __init__(self, root):
        self.root = root
        with open(os.path.join(root, "manifest.json")) as f:
            self.manifest = json.load(f)
        self.n_shards = len(self.manifest["files"])
        # token width: 2 bytes (GPT-2 pools) unless the manifest says otherwise (4 for vocabularies above 65,536:
        # daisy/multilingual/fw2_tokenize.py)
        self.dtype = np.dtype({2: np.uint16, 4: np.uint32}[int(self.manifest.get("token_bytes", 2))])
        self._cache = {}
        self._fds = collections.OrderedDict()      # shard -> (fd tokens, fd phi), LRU; see read_segment
        self._lock = threading.Lock()              # the stream plans on the main thread while reader threads load
        self.has_phi = True

    def _paths(self, s):
        hits = glob.glob(os.path.join(self.root, "tokens", "**", f"*{s:05d}*.ds"), recursive=True)
        if len(hits) != 1:
            raise FileNotFoundError(f"shard {s}: expected one .ds file, found {hits}")
        return hits[0]

    def shard(self, s):
        """(tokens memmap, ends int64 array, phi memmap (n, 768) fp16, starts int64 array), cached."""
        with self._lock:
            if s not in self._cache:
                ds = self._paths(s)
                ends = np.fromfile(ds + ".index", dtype=np.uint64).astype(np.int64)
                starts = np.concatenate([[0], ends[:-1]])
                toks = np.memmap(ds, dtype=self.dtype, mode="r")
                phi = np.memmap(os.path.join(self.root, "sidecar", f"{s:05d}.phi.f16"), dtype=np.float16, mode="r",
                                shape=(len(ends), DIM))
                if len(self._cache) > 160:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[s] = (toks, ends, phi, starts)
            return self._cache[s]

    def _fd(self, s):
        """Open file descriptors of shard s (LRU of 160). A shard leaves the LRU only after ~100 other shards were
        used since it, long after the stream stopped planning reads from it, so no in-flight read loses its fd."""
        with self._lock:
            if s in self._fds:
                self._fds.move_to_end(s)
                return self._fds[s]
        ds = self._paths(s)
        fds = (os.open(ds, os.O_RDONLY), os.open(os.path.join(self.root, "sidecar", f"{s:05d}.phi.f16"), os.O_RDONLY))
        with self._lock:
            if s in self._fds:                      # another thread opened it meanwhile
                for fd in fds:
                    os.close(fd)
                return self._fds[s]
            self._fds[s] = fds
            while len(self._fds) > 160:
                for fd in self._fds.popitem(last=False)[1]:
                    os.close(fd)
            return fds

    def read_segment(self, key, start, end, cap=None, retries=3):
        """doc(key, cap)[start:end] via pread. A failed read (a transient file-system error) is retried on freshly
        opened descriptors and logged with its arguments; a negative offset is a bug and raises at once."""
        s, i = int(key) >> 32, int(key) & 0xFFFFFFFF
        _, ends, _, starts = self.shard(s)
        a, L = int(starts[i]), int(ends[i] - starts[i])
        if a < 0 or L < 0 or start < 0 or end < start:
            raise ValueError(f"bad segment: shard {s} row {i} file offset {a} length {L} [{start}, {end}) cap {cap}")
        try:
            return _pread_capped(self._fd(s)[0], a, L, start, end, cap, self.dtype)
        except OSError as e:
            err = e
        for k in range(retries):
            print(f"[pool] read failed (shard {s} row {i}, offset {a}, length {L}, [{start}, {end}), cap {cap}): "
                  f"{err!r}; retry {k + 1}/{retries} on a fresh descriptor", flush=True)
            time.sleep(1.0 + k)
            fd = os.open(self._paths(s), os.O_RDONLY)
            try:
                return _pread_capped(fd, a, L, start, end, cap, self.dtype)
            except OSError as e:
                err = e
            finally:
                os.close(fd)
        raise OSError(err.errno, f"{err.strerror} (shard {s} row {i}, offset {a}, length {L}, [{start}, {end}), cap {cap}; "
                                 f"{retries} retries)") from err

    def read_phi(self, key):
        s, i = int(key) >> 32, int(key) & 0xFFFFFFFF
        return np.frombuffer(_pread_exact(self._fd(s)[1], 2 * DIM, 2 * DIM * i), dtype=np.float16)

    def shard_langs(self):
        """The language of every shard: the first path component of its input file ("<lang>/<file>", multilingual
        pools of daisy/multilingual/fw2_tokenize.py); None for pools without one."""
        return [f.split("/")[0] if "/" in f and not f.startswith("CC-MAIN") else None for f in self.manifest["files"]]

    def hashes(self, s):
        import pyarrow.parquet as pq

        t = pq.read_table(os.path.join(self.root, "sidecar", f"{s:05d}.meta.parquet"), columns=["text_hash", "url_hash"])
        return t.column("text_hash").to_numpy(), t.column("url_hash").to_numpy()

    # store interface for pack_segments: doc(key, cap)
    def doc(self, key, cap=None):
        s, i = int(key) >> 32, int(key) & 0xFFFFFFFF
        toks, ends, _, starts = self.shard(s)
        t = np.asarray(toks[starts[i]:ends[i]])
        if cap is not None and len(t) > cap:
            t = np.concatenate([t[: cap - 1], t[-1:]])
        return t

    def phi(self, keys):
        out = np.empty((len(keys), DIM), dtype=np.float16)
        for j, k in enumerate(keys):
            s, i = int(k) >> 32, int(k) & 0xFFFFFFFF
            out[j] = self.shard(s)[2][i]
        return out


def keep_scores(keep_dir, s, n):
    """keep_dir/{s:05d}.u8: one uint8 score per document of shard s (n documents)."""
    sc = np.fromfile(os.path.join(keep_dir, f"{s:05d}.u8"), dtype=np.uint8)
    if len(sc) != n:
        raise ValueError(f"{keep_dir}/{s:05d}.u8: {len(sc)} scores, shard {s} has {n} documents")
    return sc


def allowed_rows(pool, s, held_text, held_url, keep_dir="", keep_min=0):
    """Local indices of shard s whose text and url hashes match no held-out document (held_text / held_url: sorted
    unique uint64 arrays; url hash 0 = no url) and, with keep_dir, whose score is >= keep_min.
    -> (indices, number skipped as held-out, number filtered out by the score)."""
    n = len(pool.shard(s)[1])
    keep = np.ones(n, dtype=bool)
    if len(held_text) or len(held_url):
        th, uh = pool.hashes(s)
        keep &= ~np.isin(th, held_text) & ~(np.isin(uh, held_url) & (uh != 0))
    held = int((~keep).sum())
    if keep_dir:
        keep &= keep_scores(keep_dir, s, n) >= keep_min
    return np.flatnonzero(keep), held, int((~keep).sum()) - held


class ShardDocStream:
    def __init__(self, pool, seq_len, max_doc_tokens, seed, rank=0, world=1, n_open=64, held_text=(), held_url=(),
                 reserve=0, subset_tokens=0, reshuffle=False, preload=0, keep_dir="", keep_min=0,
                 excl_text=(), excl_url=()):
        self.pool, self.T, self.cap = pool, seq_len, max_doc_tokens
        # --daisy_pool_exclude: documents removed from each shard's order AFTER its permutation, so every other
        # document keeps its place (the stream is the stream without the exclusion minus these documents)
        self.excl_text = np.unique(np.asarray(excl_text, dtype=np.uint64))
        self.excl_url = np.unique(np.asarray(excl_url, dtype=np.uint64))
        self.excl_url = self.excl_url[self.excl_url != 0]
        self.keep_dir, self.keep_min = keep_dir or "", int(keep_min)
        self.filtered = 0       # documents of the opened shards left out by the keep filter
        self.seed, self.rank, self.n_open = seed, rank, n_open
        self.held_text = np.unique(np.asarray(held_text, dtype=np.uint64))
        self.held_url = np.unique(np.asarray(held_url, dtype=np.uint64))
        perm = np.random.default_rng(seed).permutation(pool.n_shards)   # the shard order, the same on every rank
        self.shards = perm[rank::world]
        if reserve and not 0 < reserve < len(self.shards):
            raise ValueError(f"reserve {reserve}: the rank has {len(self.shards)} shards")
        if subset_tokens and reserve:
            raise ValueError("a token subset with reserved (MATES probe) shards is not supported")
        if subset_tokens and reshuffle:
            raise ValueError("reshuffle is for the whole pool: a token subset has its own passes")
        self.reshuffle = bool(reshuffle)
        self._pass_list = None  # (pass, the rank's usable shards in that pass's order), reshuffle passes > 0
        self.n_usable = len(self.shards) - reserve     # shards [n_usable, len) of the list are never opened
        self.next_shard = 0
        self.open = []          # list of [shard, cursor]
        self.orders = {}        # shard -> permuted local doc indices (held-out excluded)
        self.rng = np.random.default_rng([seed, rank])
        self.pending = None     # (key, start)
        self.skipped = 0
        self.subset_tokens = int(subset_tokens)
        self.epoch = 0          # the pass over the subset, or over the whole pool with reshuffle (else always 0)
        self.drawn_tokens = 0   # capped tokens pass 0 has drawn (subset runs)
        self.subset = None      # [[shard, n]] once pass 0 is complete: the first n documents of the shard's pass-0 order
        self.subset_n = {}
        self.preload = int(preload)
        self._prep = ThreadPoolExecutor(2, thread_name_prefix="daisy-prep") if self.preload > 0 else None
        self._order0_ready = {}  # shard -> future of _order0_compute(shard), for the shards about to open (preload)
        self.prep_stats = {"order_sync_s": 0.0, "order_ready": 0, "order_waited": 0, "order_cold": 0}
        self._fill()
        self.prepare_upcoming()

    def _order0_compute(self, s):
        """Shard s's pass-0 order: its documents (held-out excluded) in the shard's own seeded permutation. Pure in
        (seed, s): safe on a background thread."""
        idx, skipped, filtered = allowed_rows(self.pool, s, self.held_text, self.held_url, self.keep_dir, self.keep_min)
        order = idx[np.random.default_rng([self.seed, int(s)]).permutation(len(idx))].astype(np.int64)
        if len(self.excl_text) or len(self.excl_url):
            th, uh = self.pool.hashes(s)
            th, uh = np.asarray(th, dtype=np.uint64)[order], np.asarray(uh, dtype=np.uint64)[order]
            drop = np.isin(th, self.excl_text) | (np.isin(uh, self.excl_url) & (uh != 0))
            order = order[~drop]
            skipped += int(drop.sum())
        return order, (skipped, filtered)

    def _order0(self, s):
        """_order0_compute(s), from the background preload when it prepared s (prep_stats: whether it was ready and
        how long the planning thread waited), else computed here."""
        t = time.perf_counter()
        fut = self._order0_ready.pop(int(s), None)
        if fut is not None:
            self.prep_stats["order_ready" if fut.done() else "order_waited"] += 1
            out = fut.result()
        else:
            self.prep_stats["order_cold"] += 1
            out = self._order0_compute(s)
        self.prep_stats["order_sync_s"] += time.perf_counter() - t
        return out

    def _shards_of_pass(self, epoch):
        """The rank's usable shards in pass `epoch`'s opening order (see _pass_shards)."""
        if epoch == 0 or self.subset_tokens:
            return self.shards[:self.n_usable]
        base = self.shards[:self.n_usable]
        return base[np.random.default_rng([self.seed, self.rank, 104723, epoch]).permutation(len(base))]

    def upcoming(self, m):
        """The next m shards this stream will open, in order: the rest of the current pass's list and, with reshuffle,
        the start of the next pass's (a later subset pass opens no new shards: none)."""
        if m <= 0 or (self.subset_tokens and self.epoch > 0):
            return []
        out = [int(s) for s in self._pass_shards()[self.next_shard:self.n_usable][:m]]
        if len(out) < m and self.reshuffle:
            out += [int(s) for s in self._shards_of_pass(self.epoch + 1)[:m - len(out)]]
        return out

    def preload_count(self):
        """How many upcoming shards to keep prepared: `preload`, plus the open slots the stream cannot refill in this
        pass (at the end of a pass, the next pass's first shards get ready while the last ones drain)."""
        return self.preload + max(self.n_open - len(self.open), 0) if self.preload > 0 else 0

    def prepare_upcoming(self):
        """preload > 0: start computing the orders of the upcoming shards in the background; drop prepared orders of
        shards no longer upcoming (after a resume)."""
        if self.preload <= 0:
            return
        up = self.upcoming(self.preload_count())
        for s in up:
            if s not in self._order0_ready:
                self._order0_ready[s] = self._prep.submit(self._order0_compute, s)
        keep = set(up)
        for s in [s for s in self._order0_ready if s not in keep]:
            del self._order0_ready[s]

    def _order(self, s):
        if s not in self.orders:
            order, (skipped, filtered) = self._order0(s)
            if self.epoch == 0:
                self.skipped += skipped
                self.filtered += filtered
            elif self.subset_tokens:                # pass >= 1: the shard's subset documents, reshuffled per pass
                order = order[: self.subset_n[s]]
                order = order[np.random.default_rng([self.seed, int(s), 104729, self.epoch]).permutation(len(order))]
            else:                                   # whole pool, pass >= 1 (reshuffle): all its documents, reshuffled
                order = order[np.random.default_rng([self.seed, int(s), 104729, self.epoch]).permutation(len(order))]
            self.orders[s] = order
        return self.orders[s]

    @property
    def reserved(self):
        """The shards this rank never opens (the last `reserve` of its list)."""
        return [int(s) for s in self.shards[self.n_usable:]]

    def _pass_shards(self):
        """The rank's usable shards in the current pass's order: the stream's own list in pass 0 (and in a subset run,
        which opens shards from it only in pass 0); with reshuffle, pass p > 0 a seeded permutation of that list."""
        if self.epoch == 0 or self.subset_tokens:
            return self.shards
        if self._pass_list is None or self._pass_list[0] != self.epoch:
            self._pass_list = (self.epoch, self._shards_of_pass(self.epoch))
        return self._pass_list[1]

    def _fill(self):
        while len(self.open) < self.n_open and self.next_shard < self.n_usable:
            s = int(self._pass_shards()[self.next_shard])
            self.next_shard += 1
            self._order(s)
            self.open.append([s, 0])

    def _next_doc(self):
        if self.subset_tokens:
            return self._next_subset_doc()
        return self._next_open_doc()

    def _next_open_doc(self):
        while True:
            if not self.open:
                if self.reshuffle:
                    self._new_pool_pass()
                    continue
                raise RuntimeError("shard stream exhausted (--daisy_pool_reshuffle starts another pass instead)")
            j = int(self.rng.integers(len(self.open)))
            s, c = self.open[j]
            order = self.orders[s]
            if c < len(order):
                self.open[j][1] += 1
                return (s << 32) | int(order[c])
            self.open.pop(j)          # shard done
            self.orders.pop(s, None)
            self._fill()
            self.prepare_upcoming()

    def _next_subset_doc(self):
        """Subset runs: pass 0 draws as the stream without a subset until it holds subset_tokens; later passes draw
        each document from a shard with probability proportional to what the shard has left (a uniform permutation)."""
        if self.epoch == 0:
            if self.drawn_tokens < self.subset_tokens:
                key = self._next_open_doc()
                self.drawn_tokens += self._length(key)
                return key
            self._new_pass()
        if not self.open:
            self._new_pass()
        cum = np.cumsum([len(self.orders[s]) - c for s, c in self.open])
        j = int(np.searchsorted(cum, int(self.rng.integers(cum[-1])), side="right"))
        s, c = self.open[j]
        key = (s << 32) | int(self.orders[s][c])
        self.open[j][1] += 1
        if c + 1 == len(self.orders[s]):              # the shard's subset documents are used up for this pass
            self.open.pop(j)
            self.orders.pop(s, None)
        return key

    def _pass0_counts(self):
        """[[shard, documents drawn]] of every shard pass 0 has opened, in the rank's list order (open shards: up to
        their cursor; used-up shards: all their documents)."""
        cursor = {int(s): int(c) for s, c in self.open}
        out = []
        for s in map(int, self.shards[:self.next_shard]):
            n = cursor[s] if s in cursor else len(self._order0(s)[0])
            if n:
                out.append([s, n])
        return out

    def _pass0_tokens(self):
        """The capped tokens pass 0 has drawn so far, from the shard cursors (for a state without drawn_tokens: the
        checkpoint of a run without a subset, from which a subset run starts)."""
        cursor = {int(s): int(c) for s, c in self.open}
        tok = 0
        for s in map(int, self.shards[:self.next_shard]):
            order = self.orders[s] if s in self.orders else self._order0(s)[0]
            rows = order[: cursor.get(s, len(order))]
            _, ends, _, starts = self.pool.shard(s)
            L = ends[rows] - starts[rows]
            tok += int((np.minimum(L, self.cap) if self.cap else L).sum())
        return tok

    def _new_pool_pass(self):
        """reshuffle: the rank's shards are used up; pass + 1 opens them again in the next pass's orders."""
        self.epoch += 1
        self.next_shard, self.orders, self.open = 0, {}, []
        self._fill()
        self.prepare_upcoming()
        if not self.open:
            raise RuntimeError("shard stream exhausted: the rank has no documents")
        print(f"[pool] rank {self.rank}: pass {self.epoch} over the rank's {self.n_usable} shards starts (shard and "
              f"document orders reshuffled)", flush=True)

    def _new_pass(self):
        """The next pass over the subset (fixed when pass 0 ends): every subset shard opens at cursor 0 in its order
        for the pass."""
        if self.epoch == 0:
            self.subset = self._pass0_counts()
            self.subset_n = {s: n for s, n in self.subset}
        self.epoch += 1
        self.orders, self.open = {}, [[s, 0] for s, _ in self.subset]
        for s, _ in self.open:
            self._order(s)
        print(f"[pool] rank {self.rank}: pass {self.epoch} over the subset starts ({len(self.subset)} shards, "
              f"{sum(n for _, n in self.subset):,} documents, {self.drawn_tokens:,} capped tokens)", flush=True)

    def _length(self, key):
        s, i = key >> 32, key & 0xFFFFFFFF
        _, ends, _, starts = self.pool.shard(s)
        L = int(ends[i] - starts[i])
        return min(L, self.cap) if self.cap else L

    def plan(self, B):
        """Which (key, start, end) segments fill the next micro-batch; advances the stream."""
        need = B * (self.T + 1)
        segs, have = [], 0
        while have < need:
            if self.pending is not None:
                key, start = self.pending
                self.pending = None
            else:
                key, start = self._next_doc(), 0
            L = self._length(key)
            take = min(L - start, need - have)
            segs.append((key, start, start + take))
            have += take
            if start + take < L:
                self.pending = (key, start + take)
        return segs

    def load(self, segs, pool=None):
        """Read and pack planned segments (pure; may run on another thread)."""
        return load_segments(self.pool, segs, self.T, self.cap, pool)

    def next_microbatch(self, B):
        return self.load(self.plan(B))

    def state_dict(self):
        st = {"next_shard": self.next_shard, "open": [list(map(int, o)) for o in self.open],
              "rng": self.rng.bit_generator.state, "pending": None if self.pending is None else list(self.pending),
              "n_shards_rank": len(self.shards), "seed": self.seed}
        if self.subset_tokens:
            st.update(subset_tokens=self.subset_tokens, epoch=self.epoch, drawn_tokens=self.drawn_tokens,
                      subset=self.subset)
        elif self.epoch:                            # reshuffle, pass > 0 (pass 0: the state is unchanged)
            st["epoch"] = self.epoch
        return st

    def load_state_dict(self, st):
        if st["n_shards_rank"] != len(self.shards) or st["seed"] != self.seed:
            raise ValueError("stream state from a different pool / world size / seed")
        if st["next_shard"] > self.n_usable:
            raise ValueError(f"stream state has opened {st['next_shard']} of the rank's {len(self.shards)} shards; the "
                             f"last {len(self.shards) - self.n_usable} are reserved")
        if st.get("subset_tokens", self.subset_tokens) != self.subset_tokens:
            raise ValueError(f"stream state of a {st.get('subset_tokens')}-token subset, this run has "
                             f"{self.subset_tokens or 'none'}")
        self.epoch = int(st.get("epoch", 0))
        self._pass_list = None
        self.subset = st.get("subset")
        self.subset_n = {int(s): int(n) for s, n in self.subset} if self.subset else {}
        self.next_shard = st["next_shard"]
        self.open = [list(o) for o in st["open"]]
        self.orders = {}
        for s, _ in self.open:
            self._order(s)
        self.rng.bit_generator.state = st["rng"]
        self.pending = None if st["pending"] is None else tuple(st["pending"])
        self.prepare_upcoming()
        if self.subset_tokens:
            if "drawn_tokens" in st:
                self.drawn_tokens = int(st["drawn_tokens"])
            else:
                # the checkpoint of a run without a subset: it must still be inside the subset
                self.drawn_tokens = self._pass0_tokens()
                if self.drawn_tokens > self.subset_tokens:
                    raise ValueError(f"the stream state has drawn {self.drawn_tokens:,} capped tokens, past the "
                                     f"{self.subset_tokens:,}-token subset: start from an earlier checkpoint")
            print(f"[pool] rank {self.rank}: subset of {self.subset_tokens:,} capped tokens, pass {self.epoch}; pass 0 "
                  f"has drawn {self.drawn_tokens:,}", flush=True)
