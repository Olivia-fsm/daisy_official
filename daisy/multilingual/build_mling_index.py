#!/usr/bin/env python
"""Held-out web-text set of the multilingual pool: the "index" a multilingual run passes as --daisy_index_dir. It holds
only held-out documents: --per_lang documents of every language, drawn from the pool itself
(daisy/multilingual/fw2_tokenize.py) and excluded from training by the trainer's hash exclusion (text and url hashes
of every row: daisy/data/pool.py allowed_rows), so no copy of them - exact text or same URL - is ever trained on.

Draw (seeded): per language, --shards_per_lang of its shards at random, then --per_lang / shards_per_lang rows of each
at random; a row whose text hash was drawn already is replaced. Rows are grouped by language (topic = the language's
index in sorted order), so the held-out FineWeb2 loss has one entry per language (daisy/data/stream.py
heldout_batches: the first --daisy_val_docs_per_topic rows of every topic).

Layout (as daisy/index/build_index.py, read by daisy/data/docstore.py DocStore and the trainer):
  tokens.u32 (uncapped documents, </s> included), offsets.i64.npy (N + 1), phi.f16 (N x 768, the pool's standardized
  phi), text_hash.npy, url_hash.npy (uint64), heldout_rows.npy (= arange N), heldout_topic.npy (int64 language index),
  train_mask.u8.npy (zeros: nothing here is trainable), stats.npz (the pool's), meta.json (languages, source keys).

  python daisy/multilingual/build_mling_index.py --pool DATAPATH/fineweb2-tokenized --out INDEX [--per_lang 1024] \
      [--seed 0]
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from daisy.data.pool import DsPool  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True, help="the tokenized multilingual pool (fw2_tokenize.py), with stats.npz")
    p.add_argument("--out", required=True)
    p.add_argument("--per_lang", type=int, default=1024, help="held-out documents per language")
    p.add_argument("--shards_per_lang", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    pool = DsPool(a.pool)
    if pool.dtype != np.uint32:
        raise SystemExit(f"expected a uint32 pool, got {pool.dtype}")
    langs_of = pool.shard_langs()
    langs = sorted({l for l in langs_of if l is not None})
    os.makedirs(a.out, exist_ok=True)
    keys, topic, seen = [], [], set()
    per_shard = -(-a.per_lang // a.shards_per_lang)
    for k, lang in enumerate(langs):
        rng = np.random.default_rng([a.seed, k])
        shards = np.array([i for i, l in enumerate(langs_of) if l == lang])
        pick = rng.choice(shards, size=min(len(shards), a.shards_per_lang), replace=False)
        got = []
        for s in pick:
            th, _ = pool.hashes(int(s))
            rows = rng.permutation(len(th))
            n = 0
            for r in rows:
                if int(th[r]) in seen or th[r] == 0:
                    continue
                seen.add(int(th[r]))
                got.append((int(s) << 32) | int(r))
                n += 1
                if n == per_shard or len(got) == a.per_lang:
                    break
            if len(got) == a.per_lang:
                break
        if len(got) < a.per_lang:
            raise SystemExit(f"{lang}: only {len(got)} documents drawn")
        keys += got
        topic += [k] * len(got)
        print(f"[mling index] {lang}: {len(got)} documents from {len(pick)} shards", flush=True)
    N = len(keys)
    offs = np.zeros(N + 1, dtype=np.int64)
    th_all = np.zeros(N, dtype=np.uint64)
    uh_all = np.zeros(N, dtype=np.uint64)
    phi = np.zeros((N, 768), dtype=np.float16)
    hcache = {}
    with open(os.path.join(a.out, "tokens.u32.tmp"), "wb") as f:
        for j, key in enumerate(keys):
            s, i = key >> 32, key & 0xFFFFFFFF
            t = pool.doc(key)
            f.write(np.asarray(t, dtype=np.uint32).tobytes())
            offs[j + 1] = offs[j] + len(t)
            if s not in hcache:
                hcache[s] = pool.hashes(s)
            th_all[j], uh_all[j] = hcache[s][0][i], hcache[s][1][i]
            phi[j] = pool.shard(s)[2][i]
    os.replace(os.path.join(a.out, "tokens.u32.tmp"), os.path.join(a.out, "tokens.u32"))
    np.save(os.path.join(a.out, "offsets.i64.npy"), offs)
    phi.tofile(os.path.join(a.out, "phi.f16"))
    np.save(os.path.join(a.out, "text_hash.npy"), th_all)
    np.save(os.path.join(a.out, "url_hash.npy"), uh_all)
    np.save(os.path.join(a.out, "heldout_rows.npy"), np.arange(N, dtype=np.int64))
    np.save(os.path.join(a.out, "heldout_topic.npy"), np.asarray(topic, dtype=np.int64))
    np.save(os.path.join(a.out, "train_mask.u8.npy"), np.zeros(N, dtype=np.uint8))
    shutil.copy(os.path.join(a.pool, "stats.npz"), os.path.join(a.out, "stats.npz"))
    L = np.diff(offs)
    meta = {"pool": a.pool, "langs": langs, "per_lang": a.per_lang, "shards_per_lang": a.shards_per_lang,
            "seed": a.seed, "docs": N, "tokens": int(offs[-1]),
            "tokens_per_lang": {l: int(L[np.asarray(topic) == k].sum()) for k, l in enumerate(langs)},
            "median_tokens_per_doc": {l: float(np.median(L[np.asarray(topic) == k])) for k, l in enumerate(langs)},
            "keys": [int(k) for k in keys], "git": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip()}
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(json.dumps({k: v for k, v in meta.items() if k != "keys"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
