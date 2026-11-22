#!/usr/bin/env python
"""The similarity baseline's fixed scorer for the multilingual pool: the target centroid in the pool's standardized
mmBERT space.

The pool's phi is the stored mmBERT-base vector (mean over all tokens, <= 8192 tokens), standardized with the pool's
stats.npz (daisy/multilingual/fw2_tokenize.py). The targets' items carry the same vector, computed the same way
(daisy/multilingual/build_mling_target.py and build_fw2edu_target.py: train.emb.npy, raw), so nothing is embedded here.
For every task: the items the utility labels use (daisy/labels/targets.py target_item_indices: the seeded subset packed
into the target rows, --seq_len / --target_tokens as the run), standardized with the pool's statistics, averaged; then
the equal-weight mean of the task centroids, as daisy/scorers/similarity.py. w = c / |c|, b = 0 -> OUT/w_static.npy
(769,), the file a run takes as --daisy_select_static (with --daisy_select_cosine the score is cos(phi(x), c)); also
OUT/centroids.npy (one row per task) and OUT/phi_target.json (items used, cosines).

  python daisy/multilingual/similarity_mling.py --bench TARGETS/mling --tasks gmmlu_fr,...,include_it \\
      --stats DATAPATH/fineweb2-tokenized/stats.npz --out EXPS/similarity-mling
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from daisy.labels.targets import load_items, target_item_indices  # noqa: E402


def cos_rows(a, b):
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", required=True, help="directory of the target tasks (train.bin / train.len / train.emb.npy)")
    p.add_argument("--tasks", required=True, help="comma-separated target tasks")
    p.add_argument("--stats", required=True, help="the pool's stats.npz (mean / std of the raw mmBERT vectors)")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--target_tokens", type=int, default=32768)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    st = np.load(a.stats)
    mean, std = st["mean"].astype(np.float64), st["std"].astype(np.float64)
    tasks = [t for t in a.tasks.split(",") if t.strip()]
    cents, res = {}, {"bench": a.bench, "stats": a.stats, "seq_len": a.seq_len, "target_tokens": a.target_tokens,
                      "targets": {}, "git": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip()}
    for task in tasks:
        d = os.path.join(a.bench, task)
        emb = np.load(os.path.join(d, "train.emb.npy")).astype(np.float64)
        n_items = len(load_items(d, "train"))
        if n_items != len(emb):
            raise SystemExit(f"{task}: {n_items} items but {len(emb)} embeddings")
        idx = target_item_indices(d, a.seq_len, a.target_tokens)
        z = (emb[idx] - mean) / std
        cents[task] = z.mean(0)
        res["targets"][task] = {"items_used": len(idx), "items_total": n_items,
                                "mean_cos_to_own_centroid": float(np.mean(cos_rows(z, cents[task][None])))}
    c = np.mean([cents[t] for t in tasks], axis=0)
    w = np.append(c / np.linalg.norm(c), 0.0).astype(np.float32)
    np.save(os.path.join(a.out, "w_static.npy"), w)
    np.save(os.path.join(a.out, "centroids.npy"), np.stack([cents[t] for t in tasks]).astype(np.float32))
    res["tasks"] = tasks
    res["centroid_cos_to_mean"] = {t: float(cos_rows(cents[t], c)) for t in tasks}
    with open(os.path.join(a.out, "phi_target.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
