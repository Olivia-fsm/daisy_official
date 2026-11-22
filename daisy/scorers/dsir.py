"""DSIR (Xie et al. 2023, data selection by importance resampling) as a fixed selection scorer.

A document's score is its log importance weight under DSIR's bag-of-hashed-n-grams model:
  score(x) = log p_target(x) - log p_raw(x) = sum over the n-grams of x of r[b],   r[b] = log p_target(b) - log p_raw(b),
with b the n-gram's hash bucket.
  * features: every token (unigram) and every pair of consecutive tokens of the same document (bigram), hashed into
    `buckets` buckets (DSIR: words, 10,000 buckets; the pool holds only GPT-2 tokens, so tokens stand in for words) of
    the CAPPED document the trainer trains on (t[:cap - 1] + t[-1:] when longer than cap, EOS kept, as DsPool.doc);
  * p_target: the bucket distributions of the target tasks' items the utility labels use
    (daisy.labels.targets.target_items, seed 0), averaged with equal weights, as the similarity baseline averages
    the tasks' centroids; p_raw: a uniform sample of pool documents (held-out documents excluded); every bucket count
    gets `alpha` added before normalizing;
  * scores are precomputed per pool shard by daisy/scorers/dsir_build.py (DIR/{shard:05d}.f32, one float32 per
    document in shard order) and looked up during selection (DsirScores); a shard without its file is scored on the fly.
DSIR itself resamples the pool once, globally; here the score replaces w^T phi inside the selector's candidate windows
(daisy/data/select.py), so a DSIR run differs from the similarity run only in the scorer.
"""

import json
import os

import numpy as np

_MUL = np.uint64(0x9E3779B97F4A7C15)         # Fibonacci hashing: key * 2^64 / golden ratio, mod 2^64
_BIGRAM = np.uint64(1 << 32)                 # bigram keys t1 * 65536 + t2 + 2^32 never collide with unigram keys


def bucket(keys, buckets):
    """Hash bucket in [0, buckets) of uint64 n-gram keys."""
    h = np.asarray(keys, dtype=np.uint64) * _MUL
    return ((h >> np.uint64(32)) % np.uint64(buckets)).astype(np.int64)


def ngram_buckets(seg, lens, cap, buckets):
    """The n-grams of consecutive documents: seg holds their uint16 tokens back to back, lens their lengths.
    -> (unigram buckets, their document, bigram buckets, their document), over the capped documents."""
    lens = np.asarray(lens, dtype=np.int64)
    owner = np.repeat(np.arange(len(lens)), lens)
    pos = np.arange(len(seg), dtype=np.int64) - np.repeat(np.cumsum(lens) - lens, lens)
    L = np.repeat(lens, lens)
    keep = (L <= cap) | (pos < cap - 1) | (pos == L - 1)
    t = np.asarray(seg)[keep].astype(np.uint64)
    d = owner[keep]
    same = d[1:] == d[:-1]
    bi = bucket(t[:-1][same] * np.uint64(65536) + t[1:][same] + _BIGRAM, buckets)
    return bucket(t, buckets), d, bi, d[1:][same]


def bucket_counts(seg, lens, cap, buckets):
    """Summed n-gram bucket counts (buckets,) of the documents."""
    uni, _, bi, _ = ngram_buckets(seg, lens, cap, buckets)
    return (np.bincount(uni, minlength=buckets) + np.bincount(bi, minlength=buckets)).astype(np.float64)


def doc_scores(seg, lens, cap, r):
    """Per-document scores sum_b count_x(b) r[b] (float64) of the documents."""
    uni, d, bi, bd = ngram_buckets(seg, lens, cap, len(r))
    n = len(lens)
    return np.bincount(d, weights=r[uni], minlength=n) + np.bincount(bd, weights=r[bi], minlength=n)


def fit(target_counts, raw_counts, alpha=1.0):
    """target_counts (K, buckets): one row per target task; raw_counts (buckets,). -> r (buckets,) float64."""
    t = np.atleast_2d(np.asarray(target_counts, dtype=np.float64)) + alpha
    p_target = (t / t.sum(1, keepdims=True)).mean(0)
    raw = np.asarray(raw_counts, dtype=np.float64) + alpha
    return np.log(p_target) - np.log(raw / raw.sum())


def score_shard(toks, ends, cap, r, chunk_tokens=1 << 24):
    """Scores (float32, one per document) of a whole shard: toks its uint16 tokens, ends the cumulative document ends.
    Reads and scores about chunk_tokens tokens at a time."""
    ends = np.asarray(ends, dtype=np.int64)
    starts = np.concatenate([[0], ends[:-1]])
    out = np.empty(len(ends), dtype=np.float32)
    i = 0
    while i < len(ends):
        j = max(int(np.searchsorted(ends, starts[i] + chunk_tokens, side="right")), i + 1)
        out[i:j] = doc_scores(np.asarray(toks[starts[i]:ends[j - 1]]), ends[i:j] - starts[i:j], cap, r)
        i = j
    return out


def write_scores(path, scores):
    tmp = f"{path}.tmp{os.getpid()}"
    np.asarray(scores, dtype=np.float32).tofile(tmp)
    os.replace(tmp, path)


class DsirScores:
    """The selector's DSIR scores: DIR/dsir.json (cap, buckets, ...), DIR/dsir_model.npz (r) and DIR/{shard:05d}.f32
    (dsir_build.py). Each shard's scores are read once; a shard without a file is scored here from its tokens (and the
    file written). pool: the DsPool the stream draws from (keys = shard << 32 | row)."""

    def __init__(self, root, pool, cap):
        with open(os.path.join(root, "dsir.json")) as f:
            self.meta = json.load(f)
        if int(self.meta["cap"]) != int(cap):
            raise ValueError(f"DSIR scores {root} were computed with cap {self.meta['cap']}, the run caps at {cap}")
        self.r = np.load(os.path.join(root, "dsir_model.npz"))["r"]
        self.root, self.pool, self.cap = root, pool, int(cap)
        self._shard = {}

    def describe(self):
        m = self.meta
        return (f"{m['buckets']} buckets, alpha {m['alpha']}, cap {m['cap']}, {len(m.get('shards', []))} shards "
                f"precomputed")

    def shard(self, s):
        v = self._shard.get(s)
        if v is None:
            n = len(self.pool.shard(s)[1])
            path = os.path.join(self.root, f"{s:05d}.f32")
            if os.path.exists(path):
                v = np.fromfile(path, dtype=np.float32)
            else:
                print(f"[dsir] shard {s}: no precomputed scores, scoring its {n} documents now", flush=True)
                toks, ends, _, _ = self.pool.shard(s)
                v = score_shard(toks, ends, self.cap, self.r)
                write_scores(path, v)
            if len(v) != n:
                raise ValueError(f"DSIR scores of shard {s}: {len(v)} values for {n} documents")
            self._shard[s] = v
        return v

    def scores(self, keys):
        keys = np.asarray(keys, dtype=np.int64)
        sh, row = keys >> 32, keys & 0xFFFFFFFF
        out = np.empty(len(keys), dtype=np.float32)
        for s in np.unique(sh):
            m = sh == s
            out[m] = self.shard(int(s))[row[m]]
        return out

    def keep_only(self, shards):
        """Free the scores of shards the stream has closed."""
        for s in [k for k in self._shard if k not in shards]:
            del self._shard[s]
