#!/usr/bin/env python
"""Multilingual FineWeb-Edu scores of the FineWeb2 pool, one byte per document: the input of the multilingual
FineWeb-Edu target (build_fw2edu_target.py) and of a quality-filtered uniform run (--daisy_pool_keep).

Every document's mfwedu is in its shard's sidecar meta (daisy/multilingual/fw2_tokenize.py: the FineWeb-Edu head of
the quality classifier on the document's mmBERT vector; raw output on FineWeb-Edu's 0-5 scale). This writes, for every
shard s, OUT/{s:05d}.u8 = clip(floor(10 * mfwedu), 0, 255), one byte per document in shard order: the
--daisy_pool_keep format (daisy/data/pool.py). --daisy_pool_keep_min K then keeps exactly the documents with
raw mfwedu >= K / 10 (25: rounded score >= 3, as FineWeb-Edu's int_score >= 3; 35: rounded >= 4). NaN scores -> 0 (counted).

OUT/report.json: per language and in total, documents and tokens per byte value (0.1-wide score bins: value v holds
[v/10, (v+1)/10), 0 also everything below 0.1), the shares of documents and tokens at raw cutoffs 2.0 .. 4.5; per
shard its language and its documents and tokens at >= 2.5 / >= 3.5.
A shard's .u8 is written atomically and kept on a rerun (unless --redo); the report is always computed from the scores.

  python daisy/multilingual/fw2_edu_scores.py --pool DATAPATH/fineweb2-tokenized --out DATAPATH/fineweb2-edu-scores
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from daisy.data.pool import DsPool  # noqa: E402
from daisy.runmeta import write_meta  # noqa: E402

CUTS = (20, 25, 30, 35, 40, 45)


def to_u8(x):
    """raw mfwedu (float32) -> uint8 floor(10 x) clipped to [0, 255]; NaN -> 0. -> (bytes, number of NaN)."""
    x = np.asarray(x, dtype=np.float64)
    bad = ~np.isfinite(x)
    v = np.floor(10.0 * np.where(bad, 0.0, x))
    return np.clip(v, 0, 255).astype(np.uint8), int(bad.sum())


def one(pool, out, s, redo):
    import pyarrow.parquet as pq

    _, ends, _, starts = pool.shard(s)
    n = len(ends)
    lens = (ends - starts).astype(np.int64)
    path = os.path.join(out, f"{s:05d}.u8")
    nan = 0
    if redo or not os.path.exists(path):
        t = pq.read_table(os.path.join(pool.root, "sidecar", f"{s:05d}.meta.parquet"), columns=["mfwedu"])
        if t.num_rows != n:
            raise ValueError(f"shard {s}: meta has {t.num_rows} rows, tokens index {n} documents")
        u8, nan = to_u8(t.column("mfwedu").to_numpy(zero_copy_only=False))
        tmp = path + f".tmp{os.getpid()}"
        u8.tofile(tmp)
        os.replace(tmp, path)
    else:
        u8 = np.fromfile(path, dtype=np.uint8)
        if len(u8) != n:
            raise ValueError(f"{path}: {len(u8)} bytes, shard {s} has {n} documents (rerun with --redo)")
    docs = np.bincount(u8, minlength=256).astype(np.int64)
    toks = np.bincount(u8, weights=lens, minlength=256).astype(np.int64)
    return s, docs, toks, nan


def summary(docs, toks):
    D, T = int(docs.sum()), int(toks.sum())
    return {"docs": D, "tokens": T,
            "share_docs_ge": {f"{c / 10:.1f}": float(docs[c:].sum() / max(D, 1)) for c in CUTS},
            "share_tokens_ge": {f"{c / 10:.1f}": float(toks[c:].sum() / max(T, 1)) for c in CUTS},
            "docs_ge": {f"{c / 10:.1f}": int(docs[c:].sum()) for c in CUTS},
            "tokens_ge": {f"{c / 10:.1f}": int(toks[c:].sum()) for c in CUTS},
            "docs_per_bin": docs[:61].tolist() + [int(docs[61:].sum())],     # bins 0.0 .. 6.0, last = >= 6.1
            "tokens_per_bin": toks[:61].tolist() + [int(toks[61:].sum())]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True, help="the tokenized multilingual pool (fw2_tokenize.py)")
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--redo", action="store_true")
    p.add_argument("--shards", default="", help="comma list (tests); default every shard")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pool = DsPool(a.pool)
    langs = pool.shard_langs()
    shards = [int(x) for x in a.shards.split(",")] if a.shards else list(range(pool.n_shards))
    t0 = time.time()
    res = {}
    with ThreadPoolExecutor(a.threads) as ex:
        for k, (s, d, t, nan) in enumerate(ex.map(lambda s: one(pool, a.out, s, a.redo), shards)):
            res[s] = (d, t, nan)
            if (k + 1) % 200 == 0:
                print(f"[fw2_edu_scores] {k + 1}/{len(shards)} shards, {time.time() - t0:.0f} s", flush=True)
    by = {}
    for s, (d, t, _) in res.items():
        L = langs[s] or "all"
        if L not in by:
            by[L] = [np.zeros(256, np.int64), np.zeros(256, np.int64)]
        by[L][0] += d
        by[L][1] += t
    D = sum(v[0] for v in by.values())
    T = sum(v[1] for v in by.values())
    rep = {"pool": a.pool, "out": a.out, "shards": len(shards), "nan": int(sum(v[2] for v in res.values())),
           "seconds": round(time.time() - t0, 1), "byte": "clip(floor(10 * mfwedu), 0, 255)",
           "total": summary(D, T), "langs": {L: summary(*v) for L, v in sorted(by.items())},
           "per_shard": {str(s): {"lang": langs[s], "docs": int(d.sum()), "tokens": int(t.sum()),
                                  "docs_ge2.5": int(d[25:].sum()), "docs_ge3.5": int(d[35:].sum()),
                                  "tokens_ge2.5": int(t[25:].sum()), "tokens_ge3.5": int(t[35:].sum())}
                         for s, (d, t, _) in sorted(res.items())}}
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    write_meta(a.out, "fw2_edu_scores", vars(a))
    print(f"[fw2_edu_scores] {len(shards)} shards in {rep['seconds']} s, NaN {rep['nan']}")
    print(f"{'lang':>9} {'docs':>14} {'tokens':>16} " + " ".join(f"{'d>=' + k:>8}" for k in rep["total"]["share_docs_ge"])
          + " " + " ".join(f"{'t>=' + k:>8}" for k in rep["total"]["share_tokens_ge"]))
    for L, r in list(rep["langs"].items()) + [("total", rep["total"])]:
        print(f"{L:>9} {r['docs']:>14,} {r['tokens']:>16,} "
              + " ".join(f"{100 * v:>7.2f}%" for v in r["share_docs_ge"].values()) + " "
              + " ".join(f"{100 * v:>7.2f}%" for v in r["share_tokens_ge"].values()))


if __name__ == "__main__":
    main()
