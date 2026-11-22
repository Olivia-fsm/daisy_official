#!/usr/bin/env python
"""The fixed scorer of the similarity baseline: the target's centroid in the index's phi space.

  1. Check that the embedding of the index is reproduced: `--verify` trainable documents of the index (their GPT-2
     tokens decoded to text) are embedded again with XLM-RoBERTa-base on the first <= 512-token chunk, under each
     candidate pooling, and compared with the stored phi (de-standardized). The best pooling must reach a median
     cosine >= 0.999, otherwise the script stops (the baseline must live in the same space as phi).
  2. Embed the same target items the utility labels use (each task's train-split items packed into the label rows:
     daisy.labels.targets.target_items, seed 0, --target_tokens), standardize with the index statistics, average per
     task, then average the task centroids with equal weights (like the combined target). --bench is the directory of
     the target tasks and --tasks their names (the run's --daisy_target_dir and --daisy_targets): by default the five
     benchmarks of daisy/targets/prepare_benchmarks.py; any other target works the same way, e.g. the FineWeb-Edu
     tasks of daisy/targets/build_fwedu_target.py.
  3. w = c / |c|, b = 0 -> OUT/w_static.npy (769,): with --daisy_select_cosine the score is cos(phi(x), c).
     Also OUT/centroids.npy (the per-task centroids) and OUT/phi_target.json (the pooling check, items per task).

  python daisy/scorers/similarity.py --index INDEX --bench BENCH --out OUT [--tasks T1,T2,...]

Train with it: --daisy_select_static OUT/w_static.npy --daisy_select_cosine --daisy_select_mult LAMBDA.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from daisy.labels.targets import TASKS, target_items  # noqa: E402
from daisy.runmeta import write_meta  # noqa: E402

EOT = 50256


def embed(texts, tok, model, pooling, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=512, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            h = model(**enc).last_hidden_state.float()
        m = enc["attention_mask"].float()
        if pooling == "mean":
            v = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True)
        elif pooling == "cls":
            v = h[:, 0]
        elif pooling == "mean_nospecial":
            m2 = m.clone()
            m2[:, 0] = 0
            m2[torch.arange(len(m2)), m.sum(1).long() - 1] = 0
            v = (h * m2[..., None]).sum(1) / m2.sum(1, keepdim=True)
        else:
            raise ValueError(pooling)
        out.append(v.cpu().numpy())
    return np.concatenate(out)


def cos_rows(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def main():
    from transformers import AutoModel, AutoTokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True, help="the embedding index (daisy/index/build_index.py)")
    p.add_argument("--bench", required=True, help="directory of the target tasks: <task>/train.bin, train.len (the "
                   "run's --daisy_target_dir)")
    p.add_argument("--tasks", default=",".join(TASKS), help="comma list of the target's tasks under --bench "
                   "(default: the five benchmarks); the run's --daisy_targets")
    p.add_argument("--model", default="FacebookAI/xlm-roberta-base", help="the embedding model of the index")
    p.add_argument("--gpt2", default="gpt2", help="the GPT-2 tokenizer (hub name or local directory)")
    p.add_argument("--out", required=True)
    p.add_argument("--verify", type=int, default=256, help="index documents embedded again for the pooling check")
    p.add_argument("--target_tokens", type=int, default=32768, help="the run's --daisy_target_tokens")
    p.add_argument("--seq_len", type=int, default=2048, help="the run's --sequence_length")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    write_meta(a.out, "phi_target", vars(a))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from daisy.tok import gpt2_tokenizer

    g2 = gpt2_tokenizer(a.gpt2)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model).to(dev).eval()
    st = np.load(os.path.join(a.index, "stats.npz"))
    mean, std = st["mean"].astype(np.float32), st["std"].astype(np.float32)
    phi = np.memmap(os.path.join(a.index, "phi.f16"), dtype=np.float16, mode="r").reshape(-1, len(mean))
    off = np.load(os.path.join(a.index, "offsets.i64.npy"), mmap_mode="r")
    toks = np.memmap(os.path.join(a.index, "tokens.u16"), dtype=np.uint16, mode="r")
    mask = np.load(os.path.join(a.index, "train_mask.u8.npy"), mmap_mode="r")

    # 1. reproduce the index
    rng = np.random.default_rng(0)
    rows = []
    while len(rows) < a.verify:
        r = int(rng.integers(len(mask)))
        if mask[r]:
            rows.append(r)
    texts = []
    for r in rows:
        t = np.asarray(toks[off[r]:min(off[r + 1], off[r] + 4096)])
        texts.append(g2.decode(t[t != EOT].tolist()))
    stored = phi[rows].astype(np.float32) * std + mean
    res = {"verify_docs": len(rows), "pooling": {}}
    for pool in ("mean", "cls", "mean_nospecial"):
        c = cos_rows(embed(texts, tok, model, pool), stored)
        res["pooling"][pool] = {"median_cos": float(np.median(c)), "p5_cos": float(np.percentile(c, 5)),
                                "min_cos": float(c.min())}
        print(f"[phi_target] pooling {pool}: median cos {np.median(c):.5f}, p5 {np.percentile(c, 5):.5f}, "
              f"min {c.min():.5f}", flush=True)
    best = max(res["pooling"], key=lambda k: res["pooling"][k]["median_cos"])
    res["pooling_used"] = best
    with open(os.path.join(a.out, "phi_target.json"), "w") as f:
        json.dump(res, f, indent=1)
    if res["pooling"][best]["median_cos"] < 0.999:
        raise SystemExit(f"[phi_target] STOP: no pooling reproduces the index (best {best}: "
                         f"{res['pooling'][best]['median_cos']:.4f} < 0.999); not building a baseline in another space")

    # 2. target centroid in the standardized phi space
    tasks = [t for t in a.tasks.split(",") if t.strip()]
    cents, res["targets"] = {}, {}
    for task in tasks:
        items = target_items(os.path.join(a.bench, task), a.seq_len, a.target_tokens)
        tt = [g2.decode([int(x) for x in it if x != EOT]) for it in items]
        z = (embed(tt, tok, model, best) - mean) / std
        cents[task] = z.mean(0)
        res["targets"][task] = {"items": len(tt), "mean_cos_to_own_centroid": float(np.mean(cos_rows(z, cents[task][None])))}
    c = np.mean([cents[t] for t in tasks], axis=0)
    w = np.append(c / np.linalg.norm(c), 0.0).astype(np.float32)
    np.save(os.path.join(a.out, "w_static.npy"), w)
    np.save(os.path.join(a.out, "centroids.npy"), np.stack([cents[t] for t in tasks]))
    res["tasks"] = tasks
    res["centroid_cos"] = {f"{s}|{t}": float(cents[s] @ cents[t] / np.linalg.norm(cents[s]) / np.linalg.norm(cents[t]))
                           for i, s in enumerate(tasks) for t in tasks[i + 1:]}
    print(f"[phi_target] targets {json.dumps(res['targets'])}", flush=True)
    with open(os.path.join(a.out, "phi_target.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
