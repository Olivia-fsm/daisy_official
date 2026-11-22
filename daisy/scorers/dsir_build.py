#!/usr/bin/env python
"""The DSIR baseline's scores (daisy/scorers/dsir.py) for the pool shards a run will draw from.

  1. the target: the n-gram bucket counts of each target task's items that the utility labels use (target_items,
     seed 0, --target_tokens; the similarity baseline embeds the same items);
  2. the raw pool: the counts of --raw_docs uniformly drawn documents (held-out documents excluded, with --index) of
     each of --raw_shards shards drawn from the shards to score;
  3. r = log p_target - log p_raw (fit, add-alpha smoothing) -> OUT/dsir_model.npz, OUT/dsir.json;
  4. every document of every shard to score -> OUT/{shard:05d}.f32 (one float32 per document, shard order), on
     --workers processes; existing files of the right length are kept (a rerun completes an interrupted one);
  5. check: --check random documents scored again from DsPool.doc (the trainer's own capped read), one at a time.

The shards to score are those the run's ranks open: rank k opens the first counts[k] shards of its list
perm(seed)[k::world] (ShardDocStream), plus --margin per rank. --counts (one number per rank, rank 0 first) comes from
a finished run on the same pool with the same seed, number of ranks and window factor lambda (--daisy_select_mult),
for example the similarity run: the candidate windows do not depend on the scorer, so every such run opens the same
shards. The count of rank k is "next_shard" in stream_r{k}.json of that run's final checkpoint (EXP/ckpts/latest;
under "base" for a selecting run; the run must still be in its first pass over the pool). --shards names the shards
explicitly instead (e.g. all of a small pool). A shard that the run opens without a score file is scored during
training (DsirScores), only slower.

  python daisy/scorers/dsir_build.py --pool POOL --index INDEX --bench BENCH --out OUT --counts N0,N1,N2,N3

Train with it: --daisy_select_dsir OUT --daisy_select_mult LAMBDA (with --daisy_pool POOL).
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from daisy.data.pool import DsPool, allowed_rows  # noqa: E402
from daisy.labels.targets import TASKS, target_items  # noqa: E402
from daisy.runmeta import write_meta  # noqa: E402
from daisy.scorers.dsir import bucket_counts, doc_scores, fit, score_shard, write_scores  # noqa: E402


def run_shards(n_shards, seed, world, counts, margin):
    """The shards the ranks of a run open (ShardDocStream's order), plus `margin` more per rank, sorted."""
    perm = np.random.default_rng(seed).permutation(n_shards)
    out = set()
    for k in range(world):
        out.update(int(s) for s in perm[k::world][: counts[k] + margin])
    return sorted(out)


def held_hashes(index):
    """The held-out documents' text and url hashes, as build_stream passes them to ShardDocStream."""
    if not index:
        return np.zeros(0, np.uint64), np.zeros(0, np.uint64)
    held = np.load(os.path.join(index, "heldout_rows.npy"))
    th = np.load(os.path.join(index, "text_hash.npy"), mmap_mode="r")[held]
    uh = np.load(os.path.join(index, "url_hash.npy"), mmap_mode="r")[held]
    return np.unique(np.asarray(th, np.uint64)), np.unique(np.asarray(uh[uh != 0], np.uint64))


def _raw_counts(job):
    root, s, n_docs, seed, cap, buckets, held_text, held_url = job
    pool = DsPool(root)
    rows, _, _ = allowed_rows(pool, s, held_text, held_url)
    rows = np.sort(np.random.default_rng([seed, s]).choice(rows, size=min(n_docs, len(rows)), replace=False))
    docs = [pool.doc((s << 32) | int(i), cap) for i in rows]
    return bucket_counts(np.concatenate(docs), [len(d) for d in docs], cap, buckets), len(docs), sum(map(len, docs))


def _score(job):
    root, s, cap, r, out = job
    path = os.path.join(out, f"{s:05d}.f32")
    toks, ends, _, _ = DsPool(root).shard(s)
    if os.path.exists(path) and os.path.getsize(path) == 4 * len(ends):
        return s, len(ends), 0.0, True
    t0 = time.time()
    write_scores(path, score_shard(toks, ends, cap, r))
    return s, len(ends), time.time() - t0, False


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True, help="the tokenized pool (the run's --daisy_pool)")
    p.add_argument("--bench", required=True, help="directory of the target tasks (the run's --daisy_target_dir)")
    p.add_argument("--out", required=True)
    p.add_argument("--index", default="", help="held-out documents excluded from the raw sample (as the trainer does)")
    p.add_argument("--tasks", default=",".join(TASKS), help="the target tasks under --bench (the run's "
                   "--daisy_targets)")
    p.add_argument("--seq_len", type=int, default=2048, help="the run's --sequence_length")
    p.add_argument("--target_tokens", type=int, default=32768, help="the run's --daisy_target_tokens")
    p.add_argument("--seed", type=int, default=1337, help="the run's stream seed: data_seed + seed")
    p.add_argument("--world", type=int, default=4, help="the run's number of ranks (GPUs)")
    p.add_argument("--counts", default="", help="shards each rank opens, comma-separated, rank 0 first: next_shard in "
                   "stream_r*.json of the final checkpoint of a finished run with the same seed, ranks and window "
                   "factor")
    p.add_argument("--margin", type=int, default=4, help="shards scored per rank beyond --counts")
    p.add_argument("--shards", default="", help="explicit shard list instead of --counts")
    p.add_argument("--cap", type=int, default=2048, help="the run's --daisy_max_doc_tokens")
    p.add_argument("--buckets", type=int, default=10000)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--raw_shards", type=int, default=64)
    p.add_argument("--raw_docs", type=int, default=2000)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--check", type=int, default=200)
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    pool = DsPool(a.pool)
    if a.shards:
        shards = sorted({int(s) for s in a.shards.split(",")})
    else:
        counts = [int(c) for c in a.counts.split(",")]
        if len(counts) != a.world:
            raise ValueError(f"--counts: {len(counts)} values for world {a.world}")
        shards = run_shards(pool.n_shards, a.seed, a.world, counts, a.margin)
    print(f"[dsir] {len(shards)} shards to score (of {pool.n_shards})", flush=True)

    t0 = time.time()
    tasks = a.tasks.split(",")
    target = np.zeros((len(tasks), a.buckets))
    target_info = {}
    for k, task in enumerate(tasks):
        items = target_items(os.path.join(a.bench, task), a.seq_len, a.target_tokens)
        target[k] = bucket_counts(np.concatenate(items), [len(i) for i in items], a.cap, a.buckets)
        target_info[task] = {"items": len(items), "tokens": int(sum(map(len, items)))}
    held_text, held_url = held_hashes(a.index)
    raw_from = [int(s) for s in np.sort(np.random.default_rng([a.seed, 2]).choice(
        shards, size=min(a.raw_shards, len(shards)), replace=False))]
    jobs = [(a.pool, s, a.raw_docs, a.seed, a.cap, a.buckets, held_text, held_url) for s in raw_from]
    with Pool(min(a.workers, len(jobs))) as wp:
        res = wp.map(_raw_counts, jobs)
    raw = np.sum([c for c, _, _ in res], axis=0)
    raw_docs, raw_tokens = sum(n for _, n, _ in res), sum(t for _, _, t in res)
    r = fit(target, raw, a.alpha)
    np.savez(os.path.join(a.out, "dsir_model.npz"), r=r, target_counts=target, raw_counts=raw)
    meta = {"cap": a.cap, "buckets": a.buckets, "alpha": a.alpha, "features": "GPT-2 token unigrams + bigrams",
            "hash": "Fibonacci (key * 0x9E3779B97F4A7C15) >> 32 mod buckets", "tasks": target_info,
            "target_tokens": a.target_tokens, "raw_shards": raw_from, "raw_docs": raw_docs, "raw_tokens": raw_tokens,
            "held_out_excluded": bool(a.index), "seed": a.seed, "world": a.world, "counts": a.counts,
            "margin": a.margin, "shards": shards, "pool": a.pool,
            "r": {"min": float(r.min()), "max": float(r.max()), "mean": float(r.mean()), "std": float(r.std())}}
    with open(os.path.join(a.out, "dsir.json"), "w") as f:
        json.dump(meta, f, indent=1)
    write_meta(a.out, "dsir", vars(a))
    print(f"[dsir] model: target {target.sum(1).astype(int).tolist()} n-grams per task, raw {raw_docs} documents "
          f"({raw_tokens} tokens) of {len(raw_from)} shards; r in [{r.min():.3f}, {r.max():.3f}] ({time.time() - t0:.0f} s)",
          flush=True)

    t0, done, n_docs = time.time(), 0, 0
    with Pool(a.workers) as wp:
        for s, n, dt, kept in wp.imap_unordered(_score, [(a.pool, s, a.cap, r, a.out) for s in shards]):
            done += 1
            n_docs += n
            if done % 25 == 0 or done == len(shards):
                print(f"[dsir] {done}/{len(shards)} shards ({n_docs} documents, {time.time() - t0:.0f} s; shard {s}: "
                      f"{'kept' if kept else f'{dt:.0f} s'})", flush=True)

    rng = np.random.default_rng([a.seed, 3])
    worst, vals = 0.0, []
    for s in rng.choice(shards, size=min(a.check, len(shards))):
        v = np.fromfile(os.path.join(a.out, f"{int(s):05d}.f32"), dtype=np.float32)
        i = int(rng.integers(len(v)))
        d = pool.doc((int(s) << 32) | i, a.cap)
        ref = float(doc_scores(d, [len(d)], a.cap, r)[0])
        worst = max(worst, abs(ref - float(v[i])) / max(1.0, abs(ref)))
        vals.append(v)
    allv = np.concatenate(vals)
    print(f"[dsir] check: {min(a.check, len(shards))} documents rescored from DsPool.doc, largest relative difference "
          f"{worst:.2e}; scores of the checked shards: mean {allv.mean():.2f}, std {allv.std():.2f}, "
          f"p1/p50/p99 {np.percentile(allv, [1, 50, 99]).round(2).tolist()}", flush=True)
    if worst > 1e-4:
        raise SystemExit("[dsir] check FAILED: the precomputed scores differ from the trainer's capped documents")


if __name__ == "__main__":
    main()
