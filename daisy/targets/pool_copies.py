#!/usr/bin/env python
"""Exact copies, in the training pool, of the FineWeb-Edu target and held-out documents (build_fwedu_target.py).

Why: build_fwedu_target.py drops candidates whose text or URL hash equals that of an INDEX document, but the index
may hold only a sample of the pool, and the training stream (daisy/train/trainer.build_stream) excludes only the
index's held-out FineWeb documents and their text / URL repeats. A FineWeb-Edu document whose page was crawled again in
one of the pool's dumps can therefore sit in the pool outside the index, and a run can train on it.

What: per document of TARGET/docs.parquet (role "heldout" or a target task), the pool documents with the same text
hash (an exact text copy) and those with the same URL hash but another text (the same page, changed), from the pool's
sidecars (POOL/sidecar/{shard:05d}.meta.parquet, columns text_hash and url_hash: the hashes the stream filter uses,
daisy/index/build_index.h64 = daisy/tokenize/datatrove_tokenize.h64), with each copy's dump (the directory of the
shard's file in POOL/manifest.json) and, with --keep_dir, its FineWeb-Edu int_score (the output of edu_scores.py:
whether a run restricted to FineWeb-Edu documents, --daisy_pool_keep, can draw it). The documents' texts are re-read
from the FineWeb-Edu files named in docs.parquet (EDU_DIR/data/<dump>/<file> or EDU_DIR/<dump>/<file>, columns id and
text, in either column layout of daisy/targets/edu_scores.read_edu_table; every id must be found; --edu_dir defaults
to the one build_fwedu_target.py recorded in TARGET/report.json).

Writes OUT/report.json: per group of documents ("heldout", "target_written", "target_used" = the target documents a
run reads at the builder's token budget) the documents with a text copy, with a URL copy only and with any copy, and
the number of copies; per held-out document its copies. With --clean_out, the held-out documents that have no copy of
either kind are written as their own val split (CLEAN_OUT/val.bin, val.len: same format and order as fwedu_theldout;
for daisy/eval/heldout_ckpt.py).

  python daisy/targets/pool_copies.py --pool POOL --target TARGETS/fineweb_edu --keep_dir DATAPATH/edu_scores \\
      --out TARGETS/fineweb_edu/pool_copies --clean_out TARGETS/fineweb_edu/fwedu_theldout_nocopy
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from daisy.index.build_index import h64  # noqa: E402
from daisy.labels.targets import load_items  # noqa: E402
from daisy.targets.edu_scores import dump_files, read_edu_table  # noqa: E402

_G = {}


def _scan(s):
    """-> (shard, rows, text match, url match, int_score or None) of the shard's documents matching a target hash"""
    import pyarrow.parquet as pq
    t = pq.read_table(os.path.join(_G["pool"], "sidecar", f"{s:05d}.meta.parquet"), columns=["text_hash", "url_hash"])
    th, uh = t.column("text_hash").to_numpy(), t.column("url_hash").to_numpy()
    mt = np.isin(th, _G["th"])
    mu = np.isin(uh, _G["uh"]) & (uh != 0)
    rows = np.flatnonzero(mt | mu)
    sc = None
    if _G["keep_dir"] and len(rows):
        k = np.fromfile(os.path.join(_G["keep_dir"], f"{s:05d}.u8"), dtype=np.uint8)
        if len(k) != len(th):
            raise ValueError(f"shard {s}: {len(k)} scores, {len(th)} documents")
        sc = k[rows]
    return s, rows, th[rows], uh[rows], sc


def main():
    import pyarrow.parquet as pq

    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True, help="the tokenized pool (manifest.json, sidecar/)")
    p.add_argument("--target", required=True, help="build_fwedu_target.py's OUT (docs.parquet, report.json)")
    p.add_argument("--edu_dir", default="", help="FineWeb-Edu dir of the documents (default: report.json args)")
    p.add_argument("--keep_dir", default="", help="edu_scores.py OUT: the pool documents' FineWeb-Edu int_score")
    p.add_argument("--heldout", default="fwedu_theldout", help="the held-out split's dir name under --target")
    p.add_argument("--out", required=True)
    p.add_argument("--clean_out", default="", help="write the held-out documents without any copy here (val split)")
    p.add_argument("--workers", type=int, default=32)
    a = p.parse_args()
    t0 = time.time()
    with open(os.path.join(a.target, "report.json")) as f:
        rep_t = json.load(f)
    edu_dir = a.edu_dir or rep_t["args"]["edu_dir"]
    d = pq.read_table(os.path.join(a.target, "docs.parquet")).to_pydict()
    n = len(d["id"])
    pos = {i: k for k, i in enumerate(d["id"])}
    text = [None] * n
    paths = {}
    for dump, fn in sorted(set(zip(d["dump"], d["file"]))):
        if dump not in paths:
            paths[dump] = {os.path.basename(f): f for f in dump_files([edu_dir], dump)[1]}
        if fn not in paths[dump]:
            raise FileNotFoundError(f"{fn} of {dump} not found in {edu_dir}")
        t = read_edu_table(paths[dump][fn], ["id", "text"]).to_pydict()
        for i, x in zip(t["id"], t["text"]):
            if i in pos:
                text[pos[i]] = x
    miss = [d["id"][k] for k in range(n) if text[k] is None]
    if miss:
        raise RuntimeError(f"{len(miss)} documents not found in {edu_dir}: {miss[:3]}")
    th = np.array([h64(x) for x in text], dtype=np.uint64)
    uh = np.array([h64(u) for u in d["url"]], dtype=np.uint64)
    with open(os.path.join(a.pool, "manifest.json")) as f:
        files = json.load(f)["files"]
    dumps = [os.path.basename(os.path.dirname(x)) for x in files]
    _G.update(pool=a.pool, th=np.unique(th[th != 0]), uh=np.unique(uh[uh != 0]), keep_dir=a.keep_dir)
    by_t, by_u = {}, {}
    for k in range(n):
        by_t.setdefault(int(th[k]), []).append(k)
        if uh[k]:
            by_u.setdefault(int(uh[k]), []).append(k)
    # per document: copies with the same text; same URL, other text; their dumps and scores
    n_text, n_url = np.zeros(n, np.int64), np.zeros(n, np.int64)
    best_text, best_url = np.full(n, -1, np.int64), np.full(n, -1, np.int64)
    cdumps = [set() for _ in range(n)]
    ctx = mp.get_context("fork")
    with ctx.Pool(a.workers) as pl:
        for s, rows, cth, cuh, sc in pl.imap_unordered(_scan, range(len(files)), chunksize=4):
            for j in range(len(rows)):
                score = int(sc[j]) if sc is not None else -1
                same_text = by_t.get(int(cth[j]), [])
                for k in same_text:
                    n_text[k] += 1
                    best_text[k] = max(best_text[k], score)
                    cdumps[k].add(dumps[s])
                for k in by_u.get(int(cuh[j]), []):
                    if k not in same_text:
                        n_url[k] += 1
                        best_url[k] = max(best_url[k], score)
                        cdumps[k].add(dumps[s])
    roles = np.array(d["role"])
    used = {i for v in rep_t["tasks"].values() for i in v.get("used_ids", [])}
    is_used = np.array([i in used for i in d["id"]])
    groups = {"heldout": roles == "heldout", "target_written": roles != "heldout", "target_used": is_used}
    rep = {"args": vars(a), "edu_dir": edu_dir, "pool_shards": len(files), "groups": {}}
    for g, m in groups.items():
        anyc = (n_text > 0) | (n_url > 0)
        r = {"docs": int(m.sum()), "with_text_copy": int((m & (n_text > 0)).sum()),
             "with_url_copy_only": int((m & (n_text == 0) & (n_url > 0)).sum()), "with_any_copy": int((m & anyc).sum()),
             "text_copies": int(n_text[m].sum()), "url_only_copies": int(n_url[m].sum())}
        if a.keep_dir:
            r["with_text_copy_score_ge3"] = int((m & (best_text >= 3)).sum())
            r["with_any_copy_score_ge3"] = int((m & ((best_text >= 3) | (best_url >= 3))).sum())
        rep["groups"][g] = r
    h = np.flatnonzero(roles == "heldout")
    rep["heldout_docs"] = [{"id": d["id"][k], "text_copies": int(n_text[k]), "url_only_copies": int(n_url[k]),
                            "max_score_text_copy": int(best_text[k]), "max_score_url_copy": int(best_url[k]),
                            "dumps": sorted(cdumps[k])} for k in h]
    if a.clean_out:
        items = load_items(os.path.join(a.target, a.heldout), "val")
        if len(items) != len(h):
            raise RuntimeError(f"{a.heldout}: {len(items)} items, docs.parquet has {len(h)} held-out documents")
        if any(len(items[j]) != d["tokens"][k] for j, k in enumerate(h)):
            raise RuntimeError(f"{a.heldout}: item lengths differ from docs.parquet tokens (order?)")
        from daisy.targets.build_fwedu_target import write_split
        keep = [j for j, k in enumerate(h) if n_text[k] == 0 and n_url[k] == 0]
        tok = write_split(a.clean_out, [items[j] for j in keep], "val")
        back = load_items(a.clean_out, "val")
        if not (len(back) == len(keep) and all(np.array_equal(x, items[j]) for x, j in zip(back, keep))):
            raise RuntimeError(f"{a.clean_out}: documents do not read back identically")
        rep["clean"] = {"dir": a.clean_out, "docs": len(keep), "tokens": int(sum(len(items[j]) for j in keep)),
                        "padded_tokens": tok}
    rep["seconds"] = round(time.time() - t0, 1)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "heldout_docs"}, indent=1))


if __name__ == "__main__":
    main()
