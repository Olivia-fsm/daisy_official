"""The MATES training stream of one rank, its stage lists, and the candidate draw.

MatesStream (wrapped by the Prefetcher like every other stream):
  * stage 1: the pool stream (ShardDocStream with the rank's last shards reserved for probes), micro-batch for
    micro-batch the uniform run's stream;
  * from the first boundary on: the stage's selected list (StageListStream: segments (key, 0, end) in list order,
    packed into B x (T+1) micro-batches with carry-over exactly like the other streams; end is the capped length,
    except for the one document cut to fill the stage's token budget). The list holds exactly the stage's tokens, so
    it ends with the stage; plans made past its end (the Prefetcher's read-ahead) are empty and dropped at the
    boundary;
  * the pool stream stays the source of candidates: at each boundary enumerate_candidates draws whole documents from
    it (lengths only, from the shard index) until they hold the stage's candidate tokens. The pool stream never
    revisits a document, so no document is a candidate twice. The stage-1 document carried into the next micro-batch
    when stage 1 ends is dropped (its tail is not trained, and it is not a candidate).
State (stream_r{rank}.json): {"mates_stage", "base" (the pool stream), "list" (position in the stage list and the
start of a carried-over document)}; the list itself is stage_{s}/select_r{rank}.npz (load_list). The state of a bare
pool stream (the uniform run's checkpoint, --mates_init_from) loads as stage 1.
"""

import numpy as np

from daisy.data.stream import load_segments


class StageListStream:
    """Plays a fixed list of document prefixes (key, end) in order: B x (T+1) tokens per micro-batch, carry-over."""

    def __init__(self, pool, keys, ends, seq_len, cap):
        self.pool, self.T, self.cap = pool, seq_len, cap
        self.keys, self.ends = np.asarray(keys, dtype=np.int64), np.asarray(ends, dtype=np.int64)
        self.pos = 0            # next list entry
        self.pending = None     # start of the carried-over document (entry pos - 1)

    @property
    def tokens(self):
        return int(self.ends.sum())

    def plan(self, B):
        """The next micro-batch's segments; [] once the list is used up (the Prefetcher plans a few micro-batches past
        the end of a stage; the boundary drops them, and training on one fails in load)."""
        need = B * (self.T + 1)
        if self.pending is None and self.pos >= len(self.keys):
            return []
        segs, have = [], 0
        while have < need:
            if self.pending is not None:
                i, start = self.pos - 1, self.pending
                self.pending = None
            else:
                if self.pos >= len(self.keys):
                    raise RuntimeError("stage list ends inside a micro-batch: its tokens are not a multiple of B (T+1)")
                i, start = self.pos, 0
                self.pos += 1
            key, end = int(self.keys[i]), int(self.ends[i])
            take = min(end - start, need - have)
            segs.append((key, start, start + take))
            have += take
            if start + take < end:
                self.pending = start + take
        return segs

    def load(self, segs, pool=None):
        return load_segments(self.pool, segs, self.T, self.cap, pool) if segs else None

    def next_microbatch(self, B):
        mb = self.load(self.plan(B))
        if mb is None:
            raise RuntimeError("stage list exhausted: the stage trains more tokens than were selected")
        return mb

    def state_dict(self):
        return {"pos": self.pos, "pending": self.pending, "n": len(self.keys), "tokens": self.tokens}

    def load_state_dict(self, st):
        if st["n"] != len(self.keys) or st["tokens"] != self.tokens:
            raise ValueError("stage-list state of a different selection")
        self.pos, self.pending = int(st["pos"]), st["pending"]


class MatesStream:
    """Stage 1: the pool stream; stage s >= 2: the stage's selected list. load_list(stage) -> (keys, ends)."""

    def __init__(self, base, load_list):
        if not hasattr(base, "reserved"):
            raise TypeError("MATES needs the pool stream (--daisy_pool)")
        self.base, self.load_list = base, load_list
        self.pool, self.T, self.cap = base.pool, base.T, base.cap
        self.stage, self.list = 1, None

    def plan(self, B):
        return (self.list if self.list is not None else self.base).plan(B)

    def load(self, segs, pool=None):
        """The packed micro-batch; None for the empty plan past the end of a stage list."""
        return load_segments(self.pool, segs, self.T, self.cap, pool) if segs else None

    def next_microbatch(self, B):
        mb = self.load(self.plan(B))
        if mb is None:
            raise RuntimeError("stage list exhausted: the stage trains more tokens than were selected")
        return mb

    def start_stage(self, stage, keys, ends):
        """Switch to stage `stage`'s selected list (from its first entry). The pool stream's carried-over document is
        dropped: the pool stream is only drawn from for candidates from now on."""
        self.base.pending = None
        self.stage = int(stage)
        self.list = StageListStream(self.pool, keys, ends, self.T, self.cap)

    def state_dict(self):
        return {"mates_stage": self.stage, "base": self.base.state_dict(),
                "list": self.list.state_dict() if self.list is not None else None}

    def load_state_dict(self, st):
        if "mates_stage" not in st:                       # a bare pool stream's state: the uniform run's stage 1
            self.base.load_state_dict(st)
            self.stage, self.list = 1, None
            return
        self.base.load_state_dict(st["base"])
        stage = int(st["mates_stage"])
        if stage == 1:
            self.stage, self.list = 1, None
            return
        if self.list is None or self.stage != stage:
            self.list = StageListStream(self.pool, *self.load_list(stage), self.T, self.cap)
        self.stage = stage
        self.list.load_state_dict(st["list"])


def enumerate_candidates(base, tokens):
    """Draw whole documents from the live pool stream until they hold `tokens` (capped lengths): the stage's candidates
    in draw order. -> (keys int64, capped lengths int64). Asserts that none comes from a reserved (probe) shard."""
    keys, lens, have = [], [], 0
    nxt, length = base._next_doc, base._length
    while have < tokens:
        k = nxt()
        n = length(k)
        keys.append(k)
        lens.append(n)
        have += n
    keys, lens = np.array(keys, dtype=np.int64), np.array(lens, dtype=np.int64)
    if np.isin(keys >> 32, np.array(base.reserved, dtype=np.int64)).any():
        raise AssertionError("a candidate comes from a reserved probe shard")
    return keys, lens
