#!/usr/bin/env python
"""Download the Tulu 3 SFT mixture and tokenize it for instruction tuning of pretrained checkpoints (GPT-2 ids, Tulu
chat template written out as ordinary text, assistant-only loss mask: daisy/sft/data.py).

A fixed seeded subset of --val conversations is held out (validation loss after SFT); the rest is the training split.
Writes, under --out: {train,val}.tok (uint16), {train,val}.mask (uint8), {train,val}.idx (int64 offsets, n + 1),
{train,val}.ids (the conversation ids, one per line) and meta.json (counts, token totals per split and per source,
skipped conversations, the dataset revision, the git commit).

  python daisy/sft/prep_tulu.py --out DATAPATH/tulu3-sft [--dataset allenai/tulu-3-sft-mixture] [--val 2000] [--workers 64]
"""

import argparse
import collections
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from daisy.sft.data import render  # noqa: E402

_TOK = None


def _init():
    global _TOK
    from daisy.tok import gpt2_tokenizer

    _TOK = gpt2_tokenizer()


def _encode(text):
    return _TOK.encode(text, add_special_tokens=False)


def _render(messages):
    try:
        t, m = render(messages, _encode)
    except ValueError as e:
        return None, None, str(e)
    if not m.any():
        return None, None, "no assistant tokens"
    return t, m, None


def write_split(out, split, convs, ids, sources, workers):
    t0 = time.time()
    with Pool(workers, initializer=_init) as pool:
        res = pool.map(_render, convs, chunksize=256)
    toks, masks, kept_ids, skipped = [], [], [], collections.Counter()
    by_source = collections.defaultdict(lambda: [0, 0, 0])
    for (t, m, err), cid, src in zip(res, ids, sources):
        if err is not None:
            skipped[err] += 1
            continue
        toks.append(t)
        masks.append(m)
        kept_ids.append(cid)
        s = by_source[src]
        s[0] += 1
        s[1] += len(t)
        s[2] += int(m.sum())
    lens = np.array([len(t) for t in toks], dtype=np.int64)
    np.concatenate(toks).astype(np.uint16).tofile(os.path.join(out, f"{split}.tok"))
    np.concatenate(masks).astype(np.uint8).tofile(os.path.join(out, f"{split}.mask"))
    np.concatenate([[0], np.cumsum(lens)]).astype(np.int64).tofile(os.path.join(out, f"{split}.idx"))
    with open(os.path.join(out, f"{split}.ids"), "w") as f:
        f.write("\n".join(kept_ids) + "\n")
    stats = {"conversations": len(toks), "tokens": int(lens.sum()), "target_tokens": int(sum(int(m.sum()) for m in masks)),
             "tokens_p50": int(np.median(lens)), "tokens_p99": int(np.percentile(lens, 99)),
             "over_2049_tokens": int((lens > 2049).sum()), "skipped": dict(skipped),
             "by_source": {k: {"conversations": v[0], "tokens": v[1], "target_tokens": v[2]} for k, v in
                           sorted(by_source.items())}, "seconds": round(time.time() - t0, 1)}
    print(f"[prep_tulu] {split}: {stats['conversations']} conversations, {stats['tokens'] / 1e6:.1f}M tokens "
          f"({stats['target_tokens'] / 1e6:.1f}M targets), skipped {dict(skipped)}, {stats['seconds']} s", flush=True)
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--dataset", default="allenai/tulu-3-sft-mixture")
    p.add_argument("--val", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=64)
    a = p.parse_args()
    from datasets import load_dataset

    os.makedirs(a.out, exist_ok=True)
    ds = load_dataset(a.dataset, split="train")
    print(f"[prep_tulu] {a.dataset}: {len(ds)} conversations, columns {ds.column_names}", flush=True)
    n = len(ds)
    perm = np.random.default_rng(a.seed).permutation(n)
    val_idx = np.sort(perm[: a.val])
    is_val = np.zeros(n, dtype=bool)
    is_val[val_idx] = True
    msgs, ids, srcs = ds["messages"], ds["id"], ds["source"]
    meta = {"dataset": a.dataset, "revision": getattr(ds.info, "version", None) and str(ds.info.version),
            "fingerprint": ds._fingerprint, "rows": n, "val": a.val, "seed": a.seed, "template": "tulu (plain text)",
            "tokenizer": "gpt2", "git": subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True,
                                                       text=True).stdout.strip()}
    for split, sel in (("val", is_val), ("train", ~is_val)):
        rows = np.flatnonzero(sel)
        meta[split] = write_split(a.out, split, [msgs[i] for i in rows], [ids[i] for i in rows],
                                  [srcs[i] for i in rows], a.workers)
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[prep_tulu] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
