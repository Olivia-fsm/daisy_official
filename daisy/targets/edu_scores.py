#!/usr/bin/env python
"""FineWeb-Edu scores of the pool's documents, for the baseline "uniform sampling on FineWeb-Edu": for every document
of every pool shard, its FineWeb-Edu int_score (the rounded score of the FineWeb-Edu classifier, 0-5), joined by the
document id (the CommonCrawl record id "<urn:uuid:...>", which the pool's sidecar meta and FineWeb-Edu share). A
document that is not in the FineWeb-Edu files gets 0 (its score is below 2, or FineWeb-Edu dropped it).

Inputs:
  --pool      the tokenized pool (daisy/tokenize): POOL/manifest.json ("files": the source file of every shard as
              "<dump>/<name>.parquet"; the directory part is the shard's dump), POOL/tokens/**/*{shard:05d}*.ds with
              its .ds.index, POOL/sidecar/{shard:05d}.meta.parquet (column id);
  --edu_dir   one or more local copies (a comma list) of the FineWeb-Edu score-2 dataset (documents with score >= 2
              only), one directory per dump: DIR/data/<dump>/*.parquet (the layout of the public dataset
              HuggingFaceFW/fineweb-edu-score-2) or DIR/<dump>/*.parquet. Two column layouts are read (read_edu_table):
              the public dataset's flat columns (id, int_score, ...), or a copy that holds every field but id and
              text in a struct column `metadata` (id, metadata.int_score). Each dump is read from the first directory
              that has it, so a copy in which a dump is incomplete must not come before a complete one: the missing
              documents would all get 0.
Output: OUT/{shard:05d}.u8, one byte per document in shard order, the format of --daisy_pool_keep (daisy/data/pool.py):
--daisy_pool_keep OUT --daisy_pool_keep_min 3 trains on the pool's FineWeb-Edu documents (int_score >= 3, the
definition of FineWeb-Edu).

Ids are joined by a 64-bit key from the uuid's 128 bits (both halves of its 32 hex digits, mixed); an id that is not
a uuid gets build_index.h64 of the string. When a dump's FineWeb-Edu files hold an id more than once, the largest
score is kept (the number of such duplicates is reported). Per dump: FineWeb-Edu rows, pool documents, matched
documents, and the documents and tokens per int_score (OUT/report.json, meta_edu_scores.json). Shards are written
atomically; existing files are kept (a rerun does what is left).

  python daisy/targets/edu_scores.py --pool POOL --edu_dir DATAPATH/fineweb-edu-score-2 --out DATAPATH/edu_scores
"""

import argparse
import glob
import json
import os
import sys
import time
import multiprocessing as mp

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from daisy.index.build_index import h64  # noqa: E402
from daisy.runmeta import write_meta  # noqa: E402

_HEX = np.full(256, 255, dtype=np.uint8)
for _i, _c in enumerate(b"0123456789abcdef"):
    _HEX[_c] = _i
for _i, _c in enumerate(b"ABCDEF"):
    _HEX[_c] = 10 + _i
_UUID_POS = np.r_[10:18, 19:23, 24:28, 29:33, 34:46]    # the 32 hex digits of "<urn:uuid:8-4-4-4-12>"
_MIX = np.uint64(0x9E3779B97F4A7C15)


def id_keys(ids):
    """-> uint64 key per id string (uuid: its 128 bits mixed into 64; anything else: h64 of the string)."""
    ids = list(ids)
    out = np.empty(len(ids), dtype=np.uint64)
    b = np.array([s.encode() if isinstance(s, str) else b"" for s in ids], dtype="S47")
    raw = b.view(np.uint8).reshape(len(ids), 47) if len(ids) else np.zeros((0, 47), np.uint8)
    lens = np.array([len(s) if isinstance(s, str) else 0 for s in ids])
    nib = _HEX[raw[:, _UUID_POS]]
    ok = (lens == 47) & (raw[:, 0] == ord("<")) & (raw[:, 46] == ord(">")) & (nib != 255).all(axis=1)
    if ok.any():
        n = nib[ok].astype(np.uint64)
        hi = np.zeros(int(ok.sum()), dtype=np.uint64)
        lo = np.zeros(int(ok.sum()), dtype=np.uint64)
        for j in range(16):
            hi = (hi << np.uint64(4)) | n[:, j]
            lo = (lo << np.uint64(4)) | n[:, 16 + j]
        out[ok] = hi ^ (lo * _MIX)
    for i in np.flatnonzero(~ok):
        out[i] = np.uint64(h64(ids[i]) if isinstance(ids[i], str) else 0)
    return out, int((~ok).sum())


def read_edu_table(path, columns):
    """One FineWeb-Edu parquet file -> a table with the requested fields (id, text, url, dump, score, int_score, ...)
    as top-level columns. Two layouts are read: the public dataset's flat columns, recognized by a top-level int_score
    column; otherwise a copy that holds every field but id and text in a struct column `metadata`."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    columns = list(columns)
    names = pq.read_schema(path).names
    if "int_score" in names:
        return pq.read_table(path, columns=columns)
    top = [c for c in columns if c in names]
    sub = [c for c in columns if c not in names]
    if not sub:
        return pq.read_table(path, columns=top)
    try:
        t = pq.read_table(path, columns=top + [f"metadata.{c}" for c in sub])
        md = t.column("metadata").combine_chunks()
    except Exception:  # noqa: BLE001  (no nested-column selection: read the whole struct)
        t = pq.read_table(path, columns=top + ["metadata"])
        md = t.column("metadata").combine_chunks()
    return pa.table({c: t.column(c) if c in top else pc.struct_field(md, c) for c in columns})


def read_edu_file(path):
    """-> (keys, int_score uint8, non-uuid ids) of one FineWeb-Edu parquet file."""
    t = read_edu_table(path, ["id", "int_score"])
    sc = t.column("int_score").combine_chunks().to_numpy(zero_copy_only=False)
    keys, odd = id_keys(t.column("id").to_pylist())
    return keys, np.clip(np.nan_to_num(sc, nan=0), 0, 255).astype(np.uint8), odd


def dump_files(edu_dirs, dump):
    """-> (dir, files) of the first of edu_dirs that has the dump (as DIR/<dump>/*.parquet or DIR/data/<dump>/*.parquet)."""
    for d in edu_dirs:
        for sub in (dump, os.path.join("data", dump)):
            files = sorted(glob.glob(os.path.join(d, sub, "*.parquet")))
            if files:
                return d, files
    raise FileNotFoundError(f"no FineWeb-Edu files of {dump} in {edu_dirs}")


def edu_table(edu_dirs, dump, workers):
    """-> (sorted unique keys, their int_score, stats) of one dump's FineWeb-Edu files."""
    src, files = dump_files(edu_dirs, dump)
    with mp.get_context("fork").Pool(workers) as p:
        parts = p.map(read_edu_file, files, chunksize=1)
    keys = np.concatenate([k for k, _, _ in parts])
    sc = np.concatenate([s for _, s, _ in parts])
    o = np.lexsort((sc, keys))                     # by key, then score: the last of a key's run is its largest score
    keys, sc = keys[o], sc[o]
    last = np.r_[keys[1:] != keys[:-1], True]
    st = {"dir": src, "files": len(files), "rows": int(len(keys)), "duplicate_keys": int((~last).sum()),
          "non_uuid_ids": int(sum(x for _, _, x in parts))}
    return keys[last], sc[last], st


_G = {}


def score_shard(s):
    """Shard s: the int_score of each document (0 = not in the dump's FineWeb-Edu files) -> OUT/{s:05d}.u8."""
    import pyarrow.parquet as pq

    out = os.path.join(_G["out"], f"{s:05d}.u8")
    meta = os.path.join(_G["pool"], "sidecar", f"{s:05d}.meta.parquet")
    ends = np.fromfile(os.path.join(_G["tokdir"][s] + ".index"), dtype=np.uint64).astype(np.int64)
    lens = np.diff(np.r_[0, ends])
    if os.path.exists(out):
        sc = np.fromfile(out, dtype=np.uint8)
        if len(sc) != len(lens):
            raise ValueError(f"{out}: {len(sc)} scores, shard has {len(lens)} documents")
        odd = -1
    else:
        ids = pq.read_table(meta, columns=["id"]).column("id").to_pylist()
        if len(ids) != len(lens):
            raise ValueError(f"shard {s}: {len(ids)} ids in the sidecar, {len(lens)} documents")
        keys, odd = id_keys(ids)
        K, S = _G["keys"], _G["scores"]
        pos = np.minimum(np.searchsorted(K, keys), len(K) - 1)
        sc = np.where(K[pos] == keys, S[pos], 0).astype(np.uint8)
        sc.tofile(out + ".tmp")
        os.replace(out + ".tmp", out)
    docs = np.bincount(sc, minlength=6)[:6]
    toks = np.bincount(sc, weights=lens, minlength=6)[:6]
    return s, docs.tolist(), toks.tolist(), odd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True, help="the tokenized pool (manifest.json, tokens/, sidecar/)")
    p.add_argument("--edu_dir", required=True,
                   help="FineWeb-Edu score-2 (DIR/data/<dump>/*.parquet or DIR/<dump>/*.parquet; columns id and int_score, "
                        "flat or in a metadata struct); a comma list: each dump comes from the first directory that has it")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--shards", default="", help="comma list of shards; '' = all")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.pool, "manifest.json")) as f:
        files = json.load(f)["files"]
    shards = [int(x) for x in a.shards.split(",") if x.strip()] or list(range(len(files)))
    dump_of = {s: os.path.dirname(files[s]) for s in shards}
    tokdir = {}
    for s in shards:
        hits = glob.glob(os.path.join(a.pool, "tokens", "**", f"*{s:05d}*.ds"), recursive=True)
        if len(hits) != 1:
            raise FileNotFoundError(f"shard {s}: expected one .ds file, found {hits}")
        tokdir[s] = hits[0]
    rep = {"args": vars(a), "dumps": {}}
    for dump in sorted(set(dump_of.values())):
        t0 = time.time()
        mine = [s for s in shards if dump_of[s] == dump]
        todo = [s for s in mine if not os.path.exists(os.path.join(a.out, f"{s:05d}.u8"))]
        if todo:
            keys, scores, est = edu_table(a.edu_dir.split(","), dump, a.workers)
        else:
            keys, scores, est = np.zeros(1, np.uint64), np.zeros(1, np.uint8), {"reused": True}
        _G.update(keys=keys, scores=scores, out=a.out, pool=a.pool, tokdir=tokdir)
        with mp.get_context("fork").Pool(a.workers) as pl:   # forked after _G is set: the workers share the table
            res = pl.map(score_shard, mine, chunksize=1)
        docs = np.sum([r[1] for r in res], axis=0)
        toks = np.sum([r[2] for r in res], axis=0)
        d = {"edu": est, "shards": len(mine), "pool_docs": int(docs.sum()), "matched_docs": int(docs[1:].sum()),
             "docs_by_int_score": docs.astype(int).tolist(), "tokens_by_int_score": toks.astype(int).tolist(),
             "non_uuid_pool_ids": int(sum(r[3] for r in res if r[3] >= 0)),
             "share_docs_ge3": float(docs[3:].sum() / max(1, docs.sum())),
             "share_tokens_ge3": float(toks[3:].sum() / max(1, toks.sum())),
             "share_docs_ge4": float(docs[4:].sum() / max(1, docs.sum())),
             "share_tokens_ge4": float(toks[4:].sum() / max(1, toks.sum())), "seconds": round(time.time() - t0, 1)}
        rep["dumps"][dump] = d
        print(f"[edu] {dump}: {json.dumps(d)}", flush=True)
    tot_d = np.sum([v["docs_by_int_score"] for v in rep["dumps"].values()], axis=0)
    tot_t = np.sum([v["tokens_by_int_score"] for v in rep["dumps"].values()], axis=0)
    rep["total"] = {"docs_by_int_score": tot_d.astype(int).tolist(), "tokens_by_int_score": tot_t.astype(int).tolist(),
                    "tokens_ge3": int(tot_t[3:].sum()), "tokens_ge4": int(tot_t[4:].sum()),
                    "docs_ge3": int(tot_d[3:].sum()), "docs_ge4": int(tot_d[4:].sum())}
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    write_meta(a.out, "edu_scores", vars(a), rep["total"])
    print(json.dumps(rep["total"]))


if __name__ == "__main__":
    main()
