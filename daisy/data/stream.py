"""Document streams over the index, and the packing of document segments into micro-batches.

UniformDocStream trains directly on the index: a seeded permutation of its trainable rows (train_mask == 1), the
same on every rank, split rank::world and consumed in order. Micro-batches are filled to exactly
B * (seq_len + 1) tokens: if the last document does not fit, its remaining tokens open the next micro-batch, so no
token is dropped. Each piece of a document in a micro-batch is a "segment"; segments are the units that get a label.

pack_segments / pack_parts / load_segments turn planned segments into x, y and the segment index of every position;
the pool stream (daisy/data/pool.py) and the selector (daisy/data/select.py) pack with the same functions.

heldout_batches: the fixed held-out rows (per topic) packed once into rows, with the topic of every position; they
give the held-out loss vector ell_r of refresh.jsonl (daisy/labels/gates.py: validation_pass).
"""

import numpy as np


def pack_segments(store, segs, seq_len, max_doc_tokens):
    """segs: list of (row, start, end) token ranges of capped documents, total = R * (seq_len + 1).
    Returns x, y (R, T) int64 with cross-segment targets = -1; seg (R, T) int32 segment index per input position."""
    return pack_parts([store.doc(r, max_doc_tokens)[s:e] for r, s, e in segs], seq_len)


def pack_parts(parts, seq_len):
    """pack_segments on token arrays that were already read (one per segment, in order)."""
    lens = np.array([len(p) for p in parts], dtype=np.int64)
    flat = np.concatenate(parts).astype(np.int64)
    W = seq_len + 1
    if len(flat) % W:
        raise ValueError(f"segments hold {len(flat)} tokens, not a multiple of {W}")
    owner = np.repeat(np.arange(len(parts), dtype=np.int32), lens).reshape(-1, W)
    rows = flat.reshape(-1, W)
    x, y = rows[:, :-1].copy(), rows[:, 1:].copy()
    y[owner[:, 1:] != owner[:, :-1]] = -1
    return x, y, owner[:, :-1].copy()


class UniformDocStream:
    def __init__(self, store, train_mask, seq_len, max_doc_tokens, seed, rank=0, world=1):
        self.store, self.T, self.cap = store, seq_len, max_doc_tokens
        rows = np.flatnonzero(np.asarray(train_mask)).astype(np.int64)
        rng = np.random.default_rng(seed)
        self.order = rows[rng.permutation(len(rows))][rank::world]  # identical permutation on every rank
        self.pos = 0          # next index into self.order
        self.pending = None   # (row, start): unfinished document carried into the next micro-batch
        self.epoch = 0

    def _length(self, row):
        L = int(self.store.offsets[row + 1] - self.store.offsets[row])
        return min(L, self.cap) if self.cap else L

    def plan(self, B):
        """Which (row, start, end) segments fill the next micro-batch; advances the stream (cheap: lengths only)."""
        need = B * (self.T + 1)
        segs, have = [], 0
        while have < need:
            if self.pending is not None:
                row, start = self.pending
                self.pending = None
            else:
                if self.pos >= len(self.order):
                    raise RuntimeError("uniform stream exhausted: the run consumes more documents than the shard has")
                row, start = int(self.order[self.pos]), 0
                self.pos += 1
            L = self._length(row)
            take = min(L - start, need - have)
            segs.append((row, start, start + take))
            have += take
            if start + take < L:
                self.pending = (row, start + take)
        return segs

    def load(self, segs, pool=None):
        """Read and pack planned segments (pure: no stream state is touched, so it may run on another thread)."""
        return load_segments(self.store, segs, self.T, self.cap, pool)

    def next_microbatch(self, B):
        return self.load(self.plan(B))

    def state_dict(self):
        return {"pos": self.pos, "pending": self.pending, "epoch": self.epoch, "n": len(self.order)}

    def load_state_dict(self, st):
        if st["n"] != len(self.order):
            raise ValueError("stream state from a different index/world size")
        self.pos, self.epoch = st["pos"], st["epoch"]
        self.pending = tuple(st["pending"]) if st["pending"] is not None else None


def load_segments(store, segs, seq_len, cap, pool=None):
    """Read the segments (store.read_segment, os.pread: releases the GIL, so a thread pool reads in parallel), pack
    them, and attach the phi rows when the store has them. Identical output to pack_segments + store.phi."""
    def seg_tokens(sg):
        return store.read_segment(sg[0], sg[1], sg[2], cap)

    parts = list(pool.map(seg_tokens, segs)) if pool is not None else [seg_tokens(sg) for sg in segs]
    x, y, seg = pack_parts(parts, seq_len)
    rows = np.array([sg[0] for sg in segs], dtype=np.int64)
    out = {"x": x, "y": y, "seg": seg, "rows": rows, "starts": np.array([sg[1] for sg in segs], dtype=np.int32),
           "ends": np.array([sg[2] for sg in segs], dtype=np.int32)}
    if store.has_phi:
        out["phi"] = np.stack(list(pool.map(store.read_phi, rows)) if pool is not None else
                              [store.read_phi(r) for r in rows])
    return out


def heldout_batches(store, heldout_rows, heldout_topic, docs_per_topic, seq_len, max_doc_tokens, rows_per_batch=4):
    """Fixed held-out set: the first `docs_per_topic` held-out documents of every topic (in row order), packed into
    rows (the trailing partial row is dropped). Returns a list of dicts x, y, doc, topic (CPU tensors) and a summary."""
    import torch

    heldout_rows = np.asarray(heldout_rows)
    heldout_topic = np.asarray(heldout_topic)
    D = int(heldout_topic.max()) + 1
    pick, ptopic = [], []
    for t in range(D):
        r = np.sort(heldout_rows[heldout_topic == t])[:docs_per_topic]
        pick += r.tolist()
        ptopic += [t] * len(r)
    segs, topic_of_seg = [], []
    for r, t in zip(pick, ptopic):
        L = len(store.doc(r, max_doc_tokens))
        segs.append((r, 0, L))
        topic_of_seg.append(t)
    W = seq_len + 1
    total = sum(e - s for _, s, e in segs)
    keep_tokens = (total // W) * W
    # drop the tail: truncate the segment list to keep_tokens
    out, acc = [], 0
    for (r, s, e), t in zip(segs, topic_of_seg):
        if acc >= keep_tokens:
            break
        e2 = min(e, s + keep_tokens - acc)
        out.append(((r, s, e2), t))
        acc += e2 - s
    x, y, seg = pack_segments(store, [o[0] for o in out], seq_len, max_doc_tokens)
    topic = np.array([o[1] for o in out], dtype=np.int64)[seg]
    batches = []
    for i in range(0, len(x), rows_per_batch):
        sl = slice(i, i + rows_per_batch)
        batches.append({"x": torch.from_numpy(x[sl]), "y": torch.from_numpy(y[sl]), "doc": torch.from_numpy(seg[sl]).long(),
                        "topic": torch.from_numpy(topic[sl])})
    tgt = np.bincount(topic[y >= 0], minlength=D)
    return batches, {"D": D, "docs": len(out), "rows": len(x), "target_tokens_per_topic": tgt.tolist()}
