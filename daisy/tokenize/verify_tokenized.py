#!/usr/bin/env python
"""Check a tokenized pool (the output of daisy/tokenize/datatrove_tokenize.py).

For every task in the manifest: the sidecar has as many documents as the .ds.index, the .ds holds index[-1] tokens,
every document ends with EOS, and the phi sidecar has docs x 768 fp16 values.
With --compare-index DIR (an index built by daisy/index/build_index.py from some of the same files): for those
files, compare document by document (a) the token ids and (b) phi, in raw units (so that the comparison also holds
when the two were standardized with different statistics).
The report is written to OUT/verify.json ("problems": the shards that fail a check) and summarized on stdout.

  python daisy/tokenize/verify_tokenized.py --out OUT [--workers W] [--compare-index INDEX]
"""

import argparse
import glob
import json
import os

import numpy as np

EOS = 50256


def ds_for(tok_dir, rank):
    hits = [f for f in glob.glob(os.path.join(tok_dir, "**", f"*{rank:05d}*.ds"), recursive=True)]
    if len(hits) != 1:
        raise RuntimeError(f"rank {rank}: expected one .ds file, found {hits}")
    return hits[0]


def check_rank(args):
    """Per-shard consistency: document counts (index, sidecar count, phi rows), token count, EOS at every end."""
    out, rank, f = args
    try:
        ds = ds_for(os.path.join(out, "tokens"), rank)
    except RuntimeError as e:
        return rank, None, 0, 0, str(e)
    idx = np.fromfile(ds + ".index", dtype=np.uint64)
    ntok = os.path.getsize(ds) // 2
    cnt = json.load(open(os.path.join(out, "sidecar", f"{rank:05d}.count.json")))["docs"]
    phi_n = os.path.getsize(os.path.join(out, "sidecar", f"{rank:05d}.phi.f16")) // (768 * 2)
    tok = np.memmap(ds, dtype=np.uint16, mode="r")
    ends = idx.astype(np.int64) - 1
    eos_ok = bool((tok[ends] == EOS).all()) if len(ends) else True
    ok = len(idx) == cnt == phi_n and int(idx[-1]) == ntok and eos_ok
    prob = None if ok else {"rank": rank, "file": f, "index_docs": len(idx), "sidecar_docs": cnt, "phi_docs": phi_n,
                            "index_last": int(idx[-1]), "ds_tokens": ntok, "eos_ok": eos_ok}
    return rank, ds, cnt, ntok, prob


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="the tokenized pool")
    p.add_argument("--compare-index", default=None, help="an index built from some of the pool's files")
    p.add_argument("--max-compare-docs", type=int, default=200000)
    p.add_argument("--workers", type=int, default=1, help="shards checked in parallel")
    a = p.parse_args()
    man = json.load(open(os.path.join(a.out, "manifest.json")))
    st_new = np.load(man["stats"])
    rep = {"tasks": len(man["files"]), "docs": 0, "tokens": 0, "problems": []}
    per_rank = {}
    jobs = [(a.out, rank, f) for rank, f in enumerate(man["files"])]
    if a.workers > 1:
        from multiprocessing import Pool

        with Pool(a.workers) as pool:
            results = pool.map(check_rank, jobs, chunksize=4)
    else:
        results = [check_rank(j) for j in jobs]
    for rank, ds, cnt, ntok, prob in results:
        if prob is not None:
            rep["problems"].append(prob)
        if ds is not None:
            per_rank[rank] = ds
            rep["docs"] += cnt
            rep["tokens"] += ntok
    if a.compare_index:
        ci = a.compare_index
        plan = json.load(open(os.path.join(ci, "plan.json")))
        st_old = np.load(os.path.join(ci, "stats.npz"))
        offs = np.load(os.path.join(ci, "offsets.i64.npy"), mmap_mode="r")
        toks = np.memmap(os.path.join(ci, "tokens.u16"), dtype=np.uint16, mode="r")
        phi_old = np.memmap(os.path.join(ci, "phi.f16"), dtype=np.float16, mode="r", shape=(plan["N"], 768))
        import pyarrow.parquet as pq

        cmp = {"files": 0, "docs": 0, "token_mismatch_docs": 0, "phi_raw_maxabs": 0.0, "phi_raw_rel_p99": 0.0}
        rels = []
        for fi, fobj in enumerate(plan["files"]):
            rel = os.path.relpath(fobj["path"], man["pool"])
            if rel not in man["files"]:
                continue
            rank = man["files"].index(rel)
            ds = per_rank[rank]
            idx = np.fromfile(ds + ".index", dtype=np.uint64)       # only the compared shards are loaded
            tnew = np.memmap(ds, dtype=np.uint16, mode="r")
            phi_new = np.memmap(os.path.join(a.out, "sidecar", f"{rank:05d}.phi.f16"), dtype=np.float16, mode="r",
                                shape=(len(idx), 768))
            starts = np.concatenate([[0], idx[:-1].astype(np.int64)])
            for u in plan["units"]:
                if u["file"] != fi:
                    continue
                dm = pq.read_table(os.path.join(ci, "units", f"u{u['unit']:05d}.docmap.parquet"),
                                   columns=["row", "file_row"]).to_pydict()
                for row, frow in zip(dm["row"], dm["file_row"]):
                    if cmp["docs"] >= a.max_compare_docs:
                        break
                    old = toks[offs[row]:offs[row + 1]]
                    new = tnew[starts[frow]:idx[frow]]
                    if len(old) != len(new) or not np.array_equal(old, new):
                        cmp["token_mismatch_docs"] += 1
                    r_old = phi_old[row].astype(np.float64) * st_old["std"] + st_old["mean"]
                    r_new = phi_new[frow].astype(np.float64) * st_new["std"] + st_new["mean"]
                    d = np.abs(r_old - r_new)
                    cmp["phi_raw_maxabs"] = max(cmp["phi_raw_maxabs"], float(d.max()))
                    rels.append(float(np.linalg.norm(r_old - r_new) / np.linalg.norm(r_old)))
                    cmp["docs"] += 1
            cmp["files"] += 1
        cmp["phi_raw_rel_p99"] = float(np.percentile(rels, 99)) if rels else None
        cmp["phi_raw_rel_max"] = float(np.max(rels)) if rels else None
        rep["compare_index"] = cmp
    with open(os.path.join(a.out, "verify.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "problems"}, indent=1))
    print("problems:", rep["problems"][:10], "... total", len(rep["problems"]))


if __name__ == "__main__":
    main()
