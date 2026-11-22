#!/usr/bin/env python
"""The embedding index of DAiSY: one frozen embedding phi per document, the documents' GPT-2 tokens, topic clusters
and the held-out web-text set.

Input: an "embedded pool" (--pool-dir), a directory POOL/<dump>/<name>.parquet with one subdirectory per crawl dump
(e.g. CC-MAIN-2024-10) of FineWeb documents that carry a precomputed embedding. Every parquet file has either
  * the columns `id` (string), `text` (string) and `metadata`, a struct with the fields `embeddings`, `url` (string),
    `dump` (string), `token_count` (integer) and `language_score` (float); or
  * the same fields as top-level columns: `id`, `text`, `embeddings`, `url`, `dump`, `token_count`, `language_score`
    (this layout is used when the file has a top-level `embeddings` column).
`embeddings` holds the XLM-RoBERTa-base embedding of the document's first 512-token chunk: a list of exactly --dim
(default 768) floats per document, without nulls (any float width; it is read as float32). It may also be a list of
such lists, one vector per chunk; the document's vector is then its first chunk (--policy first, the default, which
is the same space as one vector per document) or the mean over its chunks (--policy mean). `token_count` is the
document's token count under the embedding model's tokenizer; it is only recorded (as xlmr_tokens). Other columns are
ignored. daisy/scorers/similarity.py re-embeds a sample of the index and checks which pooling reproduces the stored
vectors.

Each dimension is standardized with a mean / std estimated once on a sample (XLM-R has large outlier coordinates,
and the linear scorer works on the standardized vectors). No PCA. The text is tokenized with GPT-2 (--tokenizer, a
Hugging Face tokenizer with at most 65,535 tokens; needs transformers) and stored as uint16.

Stages (run in this order; each writes under --out):
  plan       pick pool files round-robin over the dumps until they hold --target-docs documents, split them into
             units of --unit-rowgroups row groups, assign each unit a global row range -> plan.json
  stats      per-dimension mean / std of the raw vectors on a sample of the planned files -> stats.npz (mean, std, n),
             stats_sample_raw.npy, report_stats.json
  shard      per unit (a rerun skips finished units): standardized fp16 rows -> phi.f16, an (N, dim) array; tokens
             with EOS after every document -> units/u*.tok, units/u*.len.npy; units/u*.docmap.parquet (row, id, url,
             dump, file, file_row, xlmr_tokens, language_score, 64-bit text_hash and url_hash)
  tokens     concatenate the unit token files -> tokens.u16 and offsets.i64.npy (N + 1 token offsets); text_hash.npy,
             url_hash.npy (N,); the unit token files are removed unless --keep-unit-tokens
  kmeans     spherical k-means (--D topics) on a sample of phi (on the GPU when there is one); assign every row
             -> cluster.u8.npy, centroids.npy
  heldout    --heldout-per-topic held-out documents per topic -> heldout_rows.npy, heldout_topic.npy;
             train_mask.u8.npy excludes them and every document with the same text or URL
Global document id = row index in phi.f16 / offsets / cluster / train_mask. Every stage also writes meta_<stage>.json
(its arguments) and, except plan, report_<stage>.json.

  python daisy/index/build_index.py plan      --out INDEX --pool-dir POOL [--target-docs N]
  python daisy/index/build_index.py stats     --out INDEX
  python daisy/index/build_index.py shard     --out INDEX [--workers W]
  python daisy/index/build_index.py tokens    --out INDEX
  python daisy/index/build_index.py kmeans    --out INDEX [--D 16]
  python daisy/index/build_index.py heldout   --out INDEX [--heldout-per-topic 2000]
"""

import argparse
import bisect
import glob
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from daisy.runmeta import write_meta  # noqa: E402


# --------------------------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------------------------
def _jload(path):
    with open(path) as f:
        return json.load(f)


def _jsave(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


def _flat_vectors(arr, dim):
    """ListArray of fixed-length vectors -> (n, dim) float32."""
    if arr.null_count:
        raise ValueError(f"{arr.null_count} null embeddings")
    lens = pc.list_value_length(arr).to_numpy(zero_copy_only=False)
    if len(lens) and (lens != dim).any():
        bad = np.unique(lens[lens != dim])[:5]
        raise ValueError(f"embedding length != {dim}: {bad}")
    return arr.flatten().to_numpy(zero_copy_only=False).reshape(-1, dim).astype(np.float32)


def doc_vectors(arr, dim, policy="first"):
    """One vector per document. Flat list -> as stored. Nested list (one vector per chunk) -> the first
    chunk ("first", the same space as one vector per document) or the mean over chunks ("mean")."""
    vt = arr.type.value_type
    if not (pa.types.is_list(vt) or pa.types.is_large_list(vt)):
        return _flat_vectors(arr, dim)
    if policy == "first":
        return _flat_vectors(pc.list_element(arr, 0), dim)
    if policy == "mean":
        counts = pc.list_value_length(arr).to_numpy(zero_copy_only=False)
        chunks = _flat_vectors(arr.flatten(), dim)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        return (np.add.reduceat(chunks, starts, axis=0) / counts[:, None]).astype(np.float32)
    raise ValueError(policy)


def read_batches(path, row_groups, dim, policy, batch_size=8192, with_text=True):
    """Yield dicts with id, text, vec (n, dim) float32 and metadata columns, for either schema: vector and fields
    inside the `metadata` struct, or top-level columns."""
    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    nested = "metadata" in names and "embeddings" not in names
    cols = ["id"] + (["text"] if with_text else []) + (["metadata"] if nested else
                                                      ["embeddings", "url", "dump", "token_count", "language_score"])
    for b in pf.iter_batches(row_groups=row_groups, columns=cols, batch_size=batch_size):
        src = b.column("metadata") if nested else b
        if nested:  # StructArray.flatten() returns children adjusted for the parent's offset
            kids = dict(zip([f.name for f in src.type], src.flatten()))
            get = kids.__getitem__
        else:
            get = b.column
        out = {
            "id": b.column("id").to_pylist(),
            "vec": doc_vectors(get("embeddings"), dim, policy),
            "url": get("url").to_pylist(),
            "dump": get("dump").to_pylist(),
            "xlmr_tokens": get("token_count").to_numpy(zero_copy_only=False).astype(np.int32),
            "language_score": get("language_score").to_numpy(zero_copy_only=False).astype(np.float32),
        }
        if with_text:
            out["text"] = b.column("text").to_pylist()
        yield out


def h64(s):
    """Stable 64-bit hash of a string (0 is reserved for empty)."""
    if not s:
        return 0
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "surrogatepass"), digest_size=8).digest(), "little") or 1


# --------------------------------------------------------------------------------------------
# tokenizer (per process)
# --------------------------------------------------------------------------------------------
_TOK = {}


def get_tokenizer(name):
    """Returns (encode_batch, eos_id). 'bytes' is a test-only stand-in (utf-8 bytes, eos 0)."""
    if name in _TOK:
        return _TOK[name]
    if name == "bytes":
        enc = (lambda texts: [list(t.encode("utf-8"))[:4096] or [32] for t in texts]), 0
    else:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        tok.model_max_length = 10**12
        if tok.vocab_size > 65535:
            raise ValueError("uint16 token store needs vocab <= 65535")
        enc = (lambda texts: tok(texts, add_special_tokens=False)["input_ids"]), tok.eos_token_id
    _TOK[name] = enc
    return enc


# --------------------------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------------------------
def cmd_plan(a):
    if not a.pool_dir:
        raise SystemExit("[plan] needs --pool-dir")
    files = sorted(glob.glob(os.path.join(a.pool_dir, "*", "*.parquet")))
    if not files:
        raise SystemExit(f"no parquet under {a.pool_dir}")
    by_dump = {}
    for f in files:
        by_dump.setdefault(os.path.basename(os.path.dirname(f)), []).append(f)
    rng = np.random.default_rng(a.seed)
    queues = {d: list(rng.permutation(fs)) for d, fs in sorted(by_dump.items())}
    picked, total = [], 0
    while total < a.target_docs and any(queues.values()) and (a.max_files is None or len(picked) < a.max_files):
        for d in sorted(queues):
            if not queues[d] or total >= a.target_docs or (a.max_files is not None and len(picked) >= a.max_files):
                continue
            f = str(queues[d].pop(0))
            md = pq.ParquetFile(f).metadata
            rg_rows = [md.row_group(i).num_rows for i in range(md.num_row_groups)]
            picked.append({"path": f, "dump": d, "rows": int(md.num_rows), "rg_rows": rg_rows})
            total += md.num_rows
    units, row = [], 0
    for fi, f in enumerate(picked):
        rgs = f["rg_rows"]
        file_row = 0
        for s in range(0, len(rgs), a.unit_rowgroups):
            g = list(range(s, min(s + a.unit_rowgroups, len(rgs))))
            n = int(sum(rgs[i] for i in g))
            units.append({"unit": len(units), "file": fi, "row_groups": g, "row_start": row, "n": n,
                          "file_row_start": file_row})
            row += n
            file_row += n
        del f["rg_rows"]
    plan = {"pool_dir": a.pool_dir, "dim": a.dim, "policy": a.policy, "seed": a.seed, "target_docs": a.target_docs,
            "N": row, "files": picked, "units": units,
            "docs_per_dump": {d: sum(f["rows"] for f in picked if f["dump"] == d) for d in sorted(by_dump)},
            "pool_files": len(files)}
    os.makedirs(a.out, exist_ok=True)
    _jsave(plan, os.path.join(a.out, "plan.json"))
    write_meta(a.out, "plan", vars(a), {"N": row, "n_files": len(picked), "n_units": len(units)})
    print(f"[plan] files {len(picked)}/{len(files)}  units {len(units)}  N {row:,}  per dump {plan['docs_per_dump']}")


# --------------------------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------------------------
def _stats_file(job):
    path, rgs, dim, policy, keep = job
    s = np.zeros(dim)
    ss = np.zeros(dim)
    n = 0
    kept = []
    for b in read_batches(path, rgs, dim, policy, with_text=False):
        x = b["vec"].astype(np.float64)
        s += x.sum(0)
        ss += (x * x).sum(0)
        n += len(x)
        kept.append(b["vec"][:keep])
    return s, ss, n, np.concatenate(kept)[:keep]


def _spectrum(x):
    c = np.cov(x, rowvar=False)
    ev = np.sort(np.linalg.eigvalsh(c))[::-1]
    cum = np.cumsum(ev) / ev.sum()
    return {"top1_share": float(ev[0] / ev.sum()), "top5_share": float(cum[4]),
            "n_for_50pct": int(np.searchsorted(cum, 0.5) + 1), "n_for_90pct": int(np.searchsorted(cum, 0.9) + 1),
            "n_for_99pct": int(np.searchsorted(cum, 0.99) + 1),
            "participation_ratio": float(ev.sum() ** 2 / (ev ** 2).sum())}


def cmd_stats(a):
    plan = _jload(os.path.join(a.out, "plan.json"))
    rng = np.random.default_rng(a.seed + 1)
    files = plan["files"]
    pick = rng.choice(len(files), size=min(a.stats_files, len(files)), replace=False)
    jobs = []
    for fi in pick:
        n_rg = pq.ParquetFile(files[fi]["path"]).metadata.num_row_groups
        rgs = sorted(rng.choice(n_rg, size=min(a.stats_rowgroups, n_rg), replace=False).tolist())
        jobs.append((files[fi]["path"], rgs, plan["dim"], plan["policy"], a.stats_keep // len(pick) + 1))
    t0 = time.time()
    with ProcessPoolExecutor(min(a.workers, len(jobs))) as ex:
        res = list(ex.map(_stats_file, jobs))
    s = sum(r[0] for r in res)
    ss = sum(r[1] for r in res)
    n = sum(r[2] for r in res)
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 0.0))
    if (std < 1e-6).any():
        raise SystemExit(f"[stats] {(std < 1e-6).sum()} dims with ~zero variance: refusing to standardize")
    sample = np.concatenate([r[3] for r in res])[: a.stats_keep]
    np.savez(os.path.join(a.out, "stats.npz"), mean=mean, std=std, n=n)
    np.save(os.path.join(a.out, "stats_sample_raw.npy"), sample)
    z = (sample - mean) / std
    ratio = np.abs(mean) / std
    top = np.argsort(-ratio)[:8]
    rep = {"n_sample": int(n), "files": len(jobs), "seconds": time.time() - t0,
           "norm_raw": {"min": float(np.linalg.norm(sample, axis=1).min()),
                        "median": float(np.median(np.linalg.norm(sample, axis=1))),
                        "max": float(np.linalg.norm(sample, axis=1).max())},
           "std": {"min": float(std.min()), "median": float(np.median(std)), "max": float(std.max()),
                   "argmax": int(std.argmax())},
           "top_abs_mean_over_std": [[int(i), float(mean[i]), float(std[i])] for i in top],
           "spectrum_raw": _spectrum(sample.astype(np.float64)),
           "spectrum_standardized": _spectrum(z),
           "z_absmax_sample": float(np.abs(z).max()),
           "z_frac_abs_gt_6": float((np.abs(z) > 6).mean())}
    _jsave(rep, os.path.join(a.out, "report_stats.json"))
    write_meta(a.out, "stats", vars(a), {"n_sample": int(n)})
    print("[stats]", json.dumps(rep, indent=1))


# --------------------------------------------------------------------------------------------
# shard
# --------------------------------------------------------------------------------------------
def _shard_unit(job):
    out, unit, path, N, dim, policy, tokname = job
    udir = os.path.join(out, "units")
    tag = f"u{unit['unit']:05d}"
    done = os.path.join(udir, tag + ".done.json")
    if os.path.exists(done):
        return _jload(done)
    t0 = time.time()
    st = np.load(os.path.join(out, "stats.npz"))
    mean, std = st["mean"].astype(np.float32), st["std"].astype(np.float32)
    # Write rows with pwrite at their byte offsets (no mmap): many processes mapping one large file on a parallel
    # file system can see a stale file size ("mmap length is greater than file size").
    fd = os.open(os.path.join(out, "phi.f16"), os.O_WRONLY)
    enc, eos = get_tokenizer(tokname)
    row = unit["row_start"]
    lens, cols = [], {k: [] for k in ("id", "url", "dump", "xlmr_tokens", "language_score", "text_hash", "url_hash")}
    zmax, n_indoc_eos, t_tok = 0.0, 0, 0.0
    tok_tmp = os.path.join(udir, tag + ".tok.tmp")
    with open(tok_tmp, "wb") as ftok:
        for b in read_batches(path, unit["row_groups"], dim, policy, batch_size=4096):
            n = len(b["id"])
            z = (b["vec"] - mean) / std
            if not np.isfinite(z).all():
                raise ValueError(f"{tag}: non-finite standardized vector")
            zmax = max(zmax, float(np.abs(z).max()))
            buf16 = np.ascontiguousarray(z.astype(np.float16)).tobytes()
            if os.pwrite(fd, buf16, row * dim * 2) != len(buf16):
                raise IOError(f"{tag}: short write at row {row}")
            row += n
            t1 = time.time()
            ids = enc(b["text"])
            t_tok += time.time() - t1
            L = np.fromiter((len(t) + 1 for t in ids), dtype=np.int64, count=n)
            buf = np.empty(int(L.sum()), dtype=np.uint16)
            o = 0
            for t in ids:
                buf[o:o + len(t)] = t
                buf[o + len(t)] = eos
                o += len(t) + 1
            n_indoc_eos += int((buf == eos).sum()) - n
            ftok.write(buf.tobytes())
            lens.append(L.astype(np.int32))
            cols["id"] += b["id"]
            cols["url"] += b["url"]
            cols["dump"] += b["dump"]
            cols["xlmr_tokens"].append(b["xlmr_tokens"])
            cols["language_score"].append(b["language_score"])
            cols["text_hash"].append(np.fromiter((h64(t) for t in b["text"]), dtype=np.uint64, count=n))
            cols["url_hash"].append(np.fromiter((h64(u) for u in b["url"]), dtype=np.uint64, count=n))
    if row - unit["row_start"] != unit["n"]:
        raise ValueError(f"{tag}: wrote {row - unit['row_start']} rows, plan says {unit['n']}")
    os.fsync(fd)
    os.close(fd)
    lens = np.concatenate(lens)
    np.save(os.path.join(udir, tag + ".len.npy"), lens)
    n = unit["n"]
    tbl = pa.table({
        "row": np.arange(unit["row_start"], unit["row_start"] + n, dtype=np.int64),
        "id": cols["id"], "url": cols["url"], "dump": cols["dump"],
        "file": np.full(n, unit["file"], dtype=np.int32),
        "file_row": np.arange(unit["file_row_start"], unit["file_row_start"] + n, dtype=np.int32),
        "xlmr_tokens": np.concatenate(cols["xlmr_tokens"]),
        "language_score": np.concatenate(cols["language_score"]),
        "text_hash": np.concatenate(cols["text_hash"]), "url_hash": np.concatenate(cols["url_hash"]),
    })
    pq.write_table(tbl, os.path.join(udir, tag + ".docmap.parquet"), compression="zstd")
    os.replace(tok_tmp, os.path.join(udir, tag + ".tok"))
    rec = {"unit": unit["unit"], "n": n, "tokens": int(lens.sum()), "seconds": time.time() - t0,
           "tokenize_seconds": t_tok, "z_absmax": zmax, "n_indoc_eos": n_indoc_eos}
    _jsave(rec, done)
    return rec


def cmd_shard(a):
    plan = _jload(os.path.join(a.out, "plan.json"))
    N, dim = plan["N"], plan["dim"]
    os.makedirs(os.path.join(a.out, "units"), exist_ok=True)
    phi_path = os.path.join(a.out, "phi.f16")
    if not os.path.exists(phi_path):
        with open(phi_path, "wb") as f:
            f.truncate(N * dim * 2)
    elif os.path.getsize(phi_path) != N * dim * 2:
        raise SystemExit("phi.f16 exists with the wrong size")
    jobs = [(a.out, u, plan["files"][u["file"]]["path"], N, dim, plan["policy"], a.tokenizer) for u in plan["units"]]
    get_tokenizer(a.tokenizer)  # download once before forking
    t0 = time.time()
    recs = []
    with ProcessPoolExecutor(min(a.workers, len(jobs))) as ex:
        for i, r in enumerate(ex.map(_shard_unit, jobs)):
            recs.append(r)
            if (i + 1) % max(1, len(jobs) // 20) == 0:
                print(f"[shard] {i + 1}/{len(jobs)} units  {time.time() - t0:.0f}s", flush=True)
    wall = time.time() - t0
    rep = {"units": len(recs), "N": N, "tokens": sum(r["tokens"] for r in recs), "wall_seconds": wall,
           "docs_per_second": N / wall, "unit_seconds_median": float(np.median([r["seconds"] for r in recs])),
           "tokenize_share": sum(r["tokenize_seconds"] for r in recs) / sum(r["seconds"] for r in recs),
           "z_absmax": max(r["z_absmax"] for r in recs), "n_indoc_eos": sum(r["n_indoc_eos"] for r in recs),
           "workers": a.workers, "tokenizer": a.tokenizer}
    _jsave(rep, os.path.join(a.out, "report_shard.json"))
    write_meta(a.out, "shard", vars(a), rep)
    print("[shard]", json.dumps(rep, indent=1))


# --------------------------------------------------------------------------------------------
# tokens
# --------------------------------------------------------------------------------------------
def _copy_unit(job):
    src, dst, off = job
    with open(src, "rb") as fi:
        fd = os.open(dst, os.O_WRONLY)
        try:
            pos = off
            while True:
                chunk = fi.read(1 << 26)
                if not chunk:
                    break
                os.pwrite(fd, chunk, pos)
                pos += len(chunk)
        finally:
            os.close(fd)
    return pos - off


def _read_hashes(path):
    t = pq.read_table(path, columns=["text_hash", "url_hash", "xlmr_tokens"])
    return tuple(t.column(c).to_numpy() for c in ("text_hash", "url_hash", "xlmr_tokens"))


def cmd_tokens(a):
    plan = _jload(os.path.join(a.out, "plan.json"))
    udir = os.path.join(a.out, "units")
    tags = [f"u{u['unit']:05d}" for u in plan["units"]]
    missing = [t for t in tags if not os.path.exists(os.path.join(udir, t + ".done.json"))]
    if missing:
        raise SystemExit(f"[tokens] {len(missing)} units not sharded, e.g. {missing[:3]}")
    lens = np.concatenate([np.load(os.path.join(udir, t + ".len.npy")) for t in tags]).astype(np.int64)
    if len(lens) != plan["N"]:
        raise SystemExit("length table does not match N")
    offsets = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    dst = os.path.join(a.out, "tokens.u16")
    with open(dst, "wb") as f:
        f.truncate(int(offsets[-1]) * 2)
    starts = [int(offsets[u["row_start"]]) * 2 for u in plan["units"]]
    t0 = time.time()
    with ProcessPoolExecutor(min(a.workers, len(tags))) as ex:
        nbytes = list(ex.map(_copy_unit, [(os.path.join(udir, t + ".tok"), dst, s) for t, s in zip(tags, starts)]))
    if sum(nbytes) != int(offsets[-1]) * 2:
        raise SystemExit("token bytes copied != expected")
    np.save(os.path.join(a.out, "offsets.i64.npy"), offsets)
    if not a.keep_unit_tokens:  # merged copy verified above; the unit shards would double the disk use
        for t in tags:
            os.remove(os.path.join(udir, t + ".tok"))
    with ProcessPoolExecutor(min(a.workers, len(tags))) as ex:
        hs = list(ex.map(_read_hashes, [os.path.join(udir, t + ".docmap.parquet") for t in tags]))
    th = np.concatenate([h[0] for h in hs])
    uh = np.concatenate([h[1] for h in hs])
    xt = np.concatenate([h[2] for h in hs]).astype(np.int64)
    np.save(os.path.join(a.out, "text_hash.npy"), th)
    np.save(os.path.join(a.out, "url_hash.npy"), uh)
    pct = np.percentile(lens, [1, 10, 50, 90, 99, 99.9]).tolist()
    rep = {"N": len(lens), "tokens": int(offsets[-1]), "copy_seconds": time.time() - t0,
           "doc_tokens_mean": float(lens.mean()), "doc_tokens_pct_1_10_50_90_99_99.9": pct,
           "doc_tokens_max": int(lens.max()),
           "token_share_docs_over_1024": float(lens[lens > 1024].sum() / lens.sum()),
           "gpt2_over_xlmr_token_ratio_median": float(np.median((lens - 1) / np.maximum(xt, 1))),
           "exact_text_duplicates": int(len(th) - len(np.unique(th))),
           "url_duplicates": int((uh != 0).sum() - len(np.unique(uh[uh != 0])))}
    _jsave(rep, os.path.join(a.out, "report_tokens.json"))
    write_meta(a.out, "tokens", vars(a), rep)
    print("[tokens]", json.dumps(rep, indent=1))


# --------------------------------------------------------------------------------------------
# kmeans
# --------------------------------------------------------------------------------------------
def _unit_lookup(plan):
    starts = [u["row_start"] for u in plan["units"]]
    return lambda r: plan["units"][bisect.bisect_right(starts, r) - 1]


def _urls_for_rows(out, plan, rows):
    look = _unit_lookup(plan)
    res = []
    for r in rows:
        u = look(int(r))
        t = pq.read_table(os.path.join(out, "units", f"u{u['unit']:05d}.docmap.parquet"), columns=["url"])
        res.append(t.column("url")[int(r) - u["row_start"]].as_py())
    return res


def cmd_kmeans(a):
    import torch

    plan = _jload(os.path.join(a.out, "plan.json"))
    N, dim = plan["N"], plan["dim"]
    phi = np.memmap(os.path.join(a.out, "phi.f16"), dtype=np.float16, mode="r", shape=(N, dim))
    rng = np.random.default_rng(a.seed + 2)
    if a.kmeans_sample >= N:
        rows = np.arange(N)
    else:  # contiguous blocks keep the memmap reads sequential
        blk = 1000 if N >= 100_000 else 1
        nb = max(1, a.kmeans_sample // blk)
        starts = np.sort(rng.choice(N // blk, size=nb, replace=False)) * blk
        rows = np.concatenate([np.arange(s, s + blk) for s in starts])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.from_numpy(np.asarray(phi[rows], dtype=np.float32)).to(dev)
    X = torch.nn.functional.normalize(X, dim=1)
    torch.manual_seed(a.seed)
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    best = None
    t0 = time.time()
    for init in range(a.kmeans_init):
        # k-means++ on a sub-sample, then Lloyd on the full sample (cosine / spherical)
        sub = X[torch.randperm(len(X), generator=g)[:200_000].to(dev)]
        C = sub[torch.randint(len(sub), (1,), generator=g).to(dev)]
        for _ in range(1, a.D):
            d2 = (1 - (sub @ C.T).max(1).values).clamp_min(0)
            C = torch.cat([C, sub[torch.multinomial(d2 / d2.sum(), 1, generator=None)]])
        prev = None
        for it in range(a.kmeans_iters):
            assign = torch.empty(len(X), dtype=torch.long, device=dev)
            for s in range(0, len(X), 1 << 20):
                assign[s:s + (1 << 20)] = (X[s:s + (1 << 20)] @ C.T).argmax(1)
            Cn = torch.zeros_like(C).index_add_(0, assign, X)
            empty = Cn.norm(dim=1) == 0
            if empty.any():  # re-seed empty clusters with random points
                Cn[empty] = X[torch.randint(len(X), (int(empty.sum()),), generator=g).to(dev)]
            C = torch.nn.functional.normalize(Cn, dim=1)
            if prev is not None and (assign != prev).float().mean().item() < 1e-4:
                break
            prev = assign
        obj = sum(float((X[s:s + (1 << 20)] @ C.T).max(1).values.sum()) for s in range(0, len(X), 1 << 20)) / len(X)
        print(f"[kmeans] init {init}: iters {it + 1}  mean cos {obj:.4f}", flush=True)
        if best is None or obj > best[0]:
            best = (obj, C.clone(), it + 1)
    obj, C, iters = best
    cl = np.lib.format.open_memmap(os.path.join(a.out, "cluster.u8.npy"), mode="w+", dtype=np.uint8, shape=(N,))
    sizes = np.zeros(a.D, dtype=np.int64)
    margin = []
    for s in range(0, N, 1 << 21):
        x = torch.nn.functional.normalize(torch.from_numpy(np.asarray(phi[s:s + (1 << 21)], dtype=np.float32)).to(dev), dim=1)
        sim = x @ C.T
        top2 = sim.topk(2, dim=1).values
        lab = sim.argmax(1).cpu().numpy().astype(np.uint8)
        cl[s:s + len(lab)] = lab
        sizes += np.bincount(lab, minlength=a.D)
        if len(margin) < 50:
            margin.append((top2[:, 0] - top2[:, 1]).cpu().numpy()[:20000])
    cl.flush()
    np.save(os.path.join(a.out, "centroids.npy"), C.cpu().numpy())
    margin = np.concatenate(margin)
    labs = np.asarray(cl)
    examples = {}
    for c in range(a.D):
        idx = np.flatnonzero(labs[rows] == c)
        pick = rows[rng.choice(idx, size=min(6, len(idx)), replace=False)] if len(idx) else []
        examples[c] = _urls_for_rows(a.out, plan, pick)
    cc = (C @ C.T).cpu().numpy()
    rep = {"D": a.D, "sample": int(len(rows)), "device": dev, "iters": iters, "mean_cos_to_centroid": obj,
           "seconds": time.time() - t0, "sizes": sizes.tolist(), "size_min_max_ratio": float(sizes.min() / sizes.max()),
           "centroid_cos_max_offdiag": float((cc - 2 * np.eye(a.D)).max()),
           "assign_margin_median": float(np.median(margin)), "example_urls": examples}
    _jsave(rep, os.path.join(a.out, "report_kmeans.json"))
    write_meta(a.out, "kmeans", vars(a), {"sizes": sizes.tolist()})
    print("[kmeans]", json.dumps(rep, indent=1))


# --------------------------------------------------------------------------------------------
# held-out sets
# --------------------------------------------------------------------------------------------
def cmd_heldout(a):
    plan = _jload(os.path.join(a.out, "plan.json"))
    N = plan["N"]
    cl = np.load(os.path.join(a.out, "cluster.u8.npy"), mmap_mode="r")
    offsets = np.load(os.path.join(a.out, "offsets.i64.npy"))
    th = np.load(os.path.join(a.out, "text_hash.npy"))
    uh = np.load(os.path.join(a.out, "url_hash.npy"))
    rng = np.random.default_rng(a.seed + 3)
    labs = np.asarray(cl)
    D = int(labs.max()) + 1
    held, hcl = [], []
    for c in range(D):
        idx = np.flatnonzero(labs == c)
        k = min(a.heldout_per_topic, len(idx) // 10)
        pick = np.sort(rng.choice(idx, size=k, replace=False))
        held.append(pick)
        hcl.append(np.full(k, c, dtype=np.uint8))
    held = np.concatenate(held)
    hcl = np.concatenate(hcl)
    order = np.argsort(held)
    held, hcl = held[order], hcl[order]
    ht = np.unique(th[held][th[held] != 0])
    hu = np.unique(uh[held][uh[held] != 0])
    excl = np.isin(th, ht) | np.isin(uh, hu)
    excl[held] = True
    mask = (~excl).astype(np.uint8)
    np.save(os.path.join(a.out, "heldout_rows.npy"), held)
    np.save(os.path.join(a.out, "heldout_topic.npy"), hcl)
    np.save(os.path.join(a.out, "train_mask.u8.npy"), mask)
    lens = np.diff(offsets)
    rep = {"D": D, "heldout_docs": int(len(held)),
           "per_topic_docs": np.bincount(hcl, minlength=D).tolist(),
           "per_topic_tokens": [int(lens[held[hcl == c]].sum()) for c in range(D)],
           "excluded_total": int(excl.sum()), "excluded_duplicates_beyond_heldout": int(excl.sum() - len(held)),
           "trainable_docs": int(mask.sum()), "trainable_tokens": int(lens[mask.astype(bool)].sum()), "N": N}
    _jsave(rep, os.path.join(a.out, "report_heldout.json"))
    write_meta(a.out, "heldout", vars(a), rep)
    print("[heldout]", json.dumps(rep, indent=1))


# --------------------------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["plan", "stats", "shard", "tokens", "kmeans", "heldout"])
    p.add_argument("--out", required=True, help="the index directory (all stages)")
    p.add_argument("--pool-dir", default="", help="plan: the embedded pool, POOL/<dump>/*.parquet")
    p.add_argument("--dim", type=int, default=768, help="embedding dimension")
    p.add_argument("--policy", choices=["first", "mean"], default="first",
                   help="pools with one vector per chunk: the first chunk, or the mean over chunks")
    p.add_argument("--seed", type=int, default=20261013)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 8))
    p.add_argument("--tokenizer", default="gpt2")
    # plan
    p.add_argument("--target-docs", type=int, default=100_000_000)
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--unit-rowgroups", type=int, default=64)
    p.add_argument("--keep-unit-tokens", action="store_true", help="keep units/*.tok after the merge")
    # stats
    p.add_argument("--stats-files", type=int, default=64)
    p.add_argument("--stats-rowgroups", type=int, default=32)
    p.add_argument("--stats-keep", type=int, default=100_000)
    # kmeans / heldout
    p.add_argument("--D", type=int, default=16)
    p.add_argument("--kmeans-sample", type=int, default=2_000_000)
    p.add_argument("--kmeans-iters", type=int, default=100)
    p.add_argument("--kmeans-init", type=int, default=3)
    p.add_argument("--heldout-per-topic", type=int, default=2000)
    a = p.parse_args(argv)
    {"plan": cmd_plan, "stats": cmd_stats, "shard": cmd_shard, "tokens": cmd_tokens, "kmeans": cmd_kmeans,
     "heldout": cmd_heldout}[a.stage](a)


if __name__ == "__main__":
    main()
