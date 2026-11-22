#!/usr/bin/env python
"""Tokenize the multilingual FineWeb2 pool: six languages of FineWeb2 (v2.1.1), read from a copy of the dataset that
carries an mmBERT-base embedding of every document, with every document's embedding (DAiSY's phi) and quality scores
aligned with its tokens.

Input (--pool): POOL/<lang>/*.parquet, one directory per language of --langs (FineWeb2's names: fra_Latn, deu_Latn,
cmn_Hani, spa_Latn, por_Latn, ita_Latn), every file with the columns
    text      the document
    id        its FineWeb2 id
    metadata  a struct with the fields: embeddings (768 floats: the mmBERT-base vector of the document = the mean of
              the last hidden states over all tokens, special ones included, up to 8192 tokens; mmbert_embed in
              build_mling_target.py is this procedure), url, dump, language_score, minhash_cluster_size (FineWeb2's own)
The SAME vector feeds DAiSY (phi) and the quality classifier, so selection and quality scoring share one embedding.
--tokenizer: the multilingual tokenizer (a tokenizers-library tokenizer.json with a 200k vocabulary, </s> = 2 as end of
document, no BOS). --classifier: a Hugging Face id or a local directory with the two heads mfwedu.pt and mdclm.pt
(each Linear 768-3072, ReLU, Linear 3072-3072, ReLU, Linear 3072-1 on the raw mmBERT vector).
manifest.json maps rank -> "<lang>/<file>" (one rank per input file); both passes read each file in row order, so
document j of rank r is row j of the input file in every output.

Two passes (independent; they can run at the same time):
  A  sidecar  (GPU): per input file, row group by row group (columns read in bulk, vectors via numpy):
       sidecar/{rank:05d}.phi.f16       standardized mmBERT vector, fp16, 768 per document (--stats mean / std)
       sidecar/{rank:05d}.meta.parquet  id, lang, url, dump, language_score, minhash_cluster_size, 64-bit text / url
                                        hashes (as daisy/index/build_index.py), and the two heads of the classifier
                                        on the same vector: mfwedu (FineWeb-Edu head, raw logit, ~[0, 5]) and mdclm
                                        (DCLM head, sigmoid)
       sidecar/{rank:05d}.count.json    written last (the completion marker; reruns skip completed ranks)
  B  tokenize (CPU only, no torch): datatrove ParquetReader (text + id only) -> DocumentTokenizer with the multilingual
       tokenizer, </s> appended to every document, document index on, document shuffle OFF:
       tokens/{rank:05d}_unshuffled.ds + .ds.index. The vocabulary (200,064) exceeds 65,536, so the tokens are 4 bytes
       (uint32).

Modes:
  stats     per-coordinate mean / std of the raw vectors over the first k row groups of every input file
            (k = max(1, round(rows / 1e6)): languages weigh by their document counts), plus the classifier scores
            of the sample per language -> --stats (npz) and <stats>.json. The tools that read the tokenized pool
            expect the statistics as stats.npz in its directory (--out), so write them there
  sidecar   pass A for this job's ranks (--local-tasks / --local-rank-offset), --procs processes over the GPUs
  tokenize  pass B for this job's ranks, --workers datatrove workers
  verify    per shard: index / sidecar / phi / meta counts agree, the .ds holds index[-1] 4-byte tokens, every
            document ends with </s>; on the first --sample documents of ~--sample-shards shards: ids in input order,
            tokens = re-tokenization of the input text + </s>, phi = the standardized input vector, scores =
            recomputed scores, all ids < vocab
  --limit N keeps the first N documents of each file and --subset N only N evenly spaced files (timing tests).

  P=DATAPATH/fineweb2-embedded; O=DATAPATH/fineweb2-tokenized
  python daisy/multilingual/fw2_tokenize.py --mode stats --pool $P --stats $O/stats.npz
  python daisy/multilingual/fw2_tokenize.py --mode sidecar --pool $P --tokenizer TOKENIZER/tokenizer.json --out $O \\
      --stats $O/stats.npz --procs N [--local-tasks K --local-rank-offset R]
  python daisy/multilingual/fw2_tokenize.py --mode tokenize --pool $P --tokenizer TOKENIZER/tokenizer.json --out $O \\
      --workers W [--local-tasks K --local-rank-offset R]
  python daisy/multilingual/fw2_tokenize.py --mode verify --out $O
"""

import argparse
import inspect
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from daisy.tokenize.datatrove_tokenize import h64  # noqa: E402

LANGS = ("fra_Latn", "deu_Latn", "cmn_Hani", "spa_Latn", "por_Latn", "ita_Latn")
CLASSIFIER = "epfml/FineWeb2-HQ-PlusPlus-Classifier"
EOS = "</s>"
DIM = 768


def list_files(pool, langs):
    return [f"{l}/{f}" for l in langs for f in sorted(os.listdir(os.path.join(pool, l))) if f.endswith(".parquet")]


def load_heads(clf, device):
    """The two heads of the classifier card (Linear 768-3072, ReLU, Linear 3072-3072, ReLU, Linear 3072-1)."""
    import torch
    from huggingface_hub import snapshot_download

    d = clf if os.path.isdir(clf) else snapshot_download(clf)
    heads = {}
    for name in ("mfwedu", "mdclm"):
        sd = torch.load(os.path.join(d, f"{name}.pt"), weights_only=True, map_location="cpu")
        net = torch.nn.Sequential(torch.nn.Linear(DIM, 3072), torch.nn.ReLU(), torch.nn.Linear(3072, 3072),
                                  torch.nn.ReLU(), torch.nn.Linear(3072, 1))
        net.load_state_dict({k.replace("classifier.", "", 1): v for k, v in sd.items()})
        heads[name] = net.eval().to(device)
    return heads


def score(heads, v, device):
    """v: (n, 768) raw float32 vectors -> {mfwedu: raw logit, mdclm: sigmoid} as float32 arrays."""
    import torch

    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)).to(device)
        return {"mfwedu": heads["mfwedu"](x).squeeze(-1).float().cpu().numpy(),
                "mdclm": torch.sigmoid(heads["mdclm"](x)).squeeze(-1).float().cpu().numpy()}


_W = {}


def _sidecar_init(stats_path, classifier):
    import multiprocessing as mp

    import torch

    torch.set_num_threads(2)
    i = (mp.current_process()._identity or (1,))[0] - 1
    dev = f"cuda:{i % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"
    st = np.load(stats_path)
    _W.update(device=dev, heads=load_heads(classifier, dev), mean=st["mean"].astype(np.float32),
              std=st["std"].astype(np.float32))


def _sidecar_file(job):
    """Pass A for one input file (rank): read row groups in order, write phi / meta / count."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    rank, rel, pool, out, limit = job
    sd = os.path.join(out, "sidecar")
    if os.path.exists(os.path.join(sd, f"{rank:05d}.count.json")):
        return rank, -1, 0.0
    t0 = time.time()
    lang = rel.split("/")[0]
    pf = pq.ParquetFile(os.path.join(pool, rel))
    cols = {k: [] for k in ("id", "url", "dump", "language_score", "minhash_cluster_size", "text_hash", "url_hash",
                            "mfwedu", "mdclm")}
    n = 0
    phi_path = os.path.join(sd, f"{rank:05d}.phi.f16")
    with open(phi_path + ".tmp", "wb") as fphi:
        for g in range(pf.metadata.num_row_groups):
            if 0 <= limit <= n:
                break
            t = pf.read_row_group(g, columns=["id", "text", "metadata"])
            m = t.column("metadata").combine_chunks()
            lst = m.field("embeddings")
            if not (pc.list_value_length(lst).to_numpy(zero_copy_only=False) == DIM).all():
                raise ValueError(f"{rel} row group {g}: embedding length != {DIM}")
            v = lst.flatten().to_numpy(zero_copy_only=False).reshape(-1, DIM).astype(np.float32)
            ids = t.column("id").to_pylist()
            texts = t.column("text").to_pylist()
            urls = [u or "" for u in m.field("url").to_pylist()]
            k = len(ids) if limit < 0 else min(len(ids), limit - n)
            v, ids, texts, urls = v[:k], ids[:k], texts[:k], urls[:k]
            fphi.write(((v - _W["mean"]) / _W["std"]).astype(np.float16).tobytes())
            sc = score(_W["heads"], v, _W["device"])
            cols["mfwedu"].append(sc["mfwedu"])
            cols["mdclm"].append(sc["mdclm"])
            cols["id"] += ids
            cols["url"] += urls
            cols["dump"] += m.field("dump").to_pylist()[:k]
            cols["language_score"] += [-1.0 if x is None else float(x) for x in m.field("language_score").to_pylist()[:k]]
            cols["minhash_cluster_size"] += [-1 if x is None else int(x) for x in m.field("minhash_cluster_size").to_pylist()[:k]]
            cols["text_hash"] += [h64(x) for x in texts]
            cols["url_hash"] += [h64(x) for x in urls]
            n += k
    pq.write_table(pa.table({
        "id": cols["id"], "lang": pa.array([lang] * n).dictionary_encode(), "url": cols["url"],
        "dump": pa.array(cols["dump"], type=pa.string()).dictionary_encode(),
        "language_score": np.array(cols["language_score"], dtype=np.float32),
        "minhash_cluster_size": np.array(cols["minhash_cluster_size"], dtype=np.int32),
        "text_hash": np.array(cols["text_hash"], dtype=np.uint64), "url_hash": np.array(cols["url_hash"], dtype=np.uint64),
        "mfwedu": np.concatenate(cols["mfwedu"]) if n else np.zeros(0, np.float32),
        "mdclm": np.concatenate(cols["mdclm"]) if n else np.zeros(0, np.float32)}),
        os.path.join(sd, f"{rank:05d}.meta.parquet"), compression="zstd")
    os.replace(phi_path + ".tmp", phi_path)
    with open(os.path.join(sd, f"{rank:05d}.count.json"), "w") as f:
        json.dump({"rank": rank, "file": rel, "lang": lang, "docs": n, "device": _W["device"],
                   "first_id": cols["id"][0] if n else None, "last_id": cols["id"][-1] if n else None,
                   "seconds": time.time() - t0}, f)
    return rank, n, time.time() - t0



def tokenizer_step(out_dir, tokenizer, eos, batch_size=None):
    """DocumentTokenizer with document shuffling off (the argument name changed across datatrove versions).

    batch_size = documents per tokenizer.encode_batch call. With datatrove's default (10,000) every worker holds the
    Encoding objects and normalization alignments of 10,000 documents, which is too much memory for many workers per
    node on long documents; each document's tokens do not depend on the batch."""
    from datatrove.pipeline.tokens import DocumentTokenizer

    sig = inspect.signature(DocumentTokenizer.__init__).parameters
    kw = dict(output_folder=out_dir, tokenizer_name_or_path=tokenizer, eos_token=eos, save_index=True)
    if "shuffle_documents" in sig:
        kw["shuffle_documents"] = False
    elif "shuffle" in sig:
        kw["shuffle"] = False
    else:
        raise RuntimeError(f"cannot turn off DocumentTokenizer shuffling; signature: {list(sig)}")
    if "max_tokens_per_file" in sig:
        kw["max_tokens_per_file"] = None  # one .ds file per task
    if batch_size:
        if "batch_size" not in sig:
            raise RuntimeError(f"DocumentTokenizer has no batch_size; signature: {list(sig)}")
        kw["batch_size"] = batch_size
    return DocumentTokenizer(**kw), kw


def embeddings_of_row_group(pf, g):
    """(n, 768) float64 vectors of one row group, reading only metadata.embeddings when the reader allows it."""
    import pyarrow.compute as pc

    try:
        t = pf.read_row_group(g, columns=["metadata.embeddings"])
    except Exception:
        t = pf.read_row_group(g, columns=["metadata"])
    col = t.column("metadata").combine_chunks()
    lst = col.field("embeddings")
    lens = pc.list_value_length(lst).to_numpy(zero_copy_only=False)
    if not (lens == DIM).all():
        raise ValueError(f"embedding lengths {np.unique(lens)}")
    return lst.flatten().to_numpy(zero_copy_only=False).reshape(-1, DIM)


def mode_stats(a):
    import pyarrow.parquet as pq
    import torch

    need_pool(a)
    files = list_files(a.pool, a.langs)
    t0 = time.time()

    def one(rel):
        pf = pq.ParquetFile(os.path.join(a.pool, rel))
        k = max(1, int(round(pf.metadata.num_rows / 1e6)))
        return rel.split("/")[0], np.concatenate([embeddings_of_row_group(pf, g) for g in range(min(k, pf.metadata.num_row_groups))])

    s = np.zeros(DIM)
    ss = np.zeros(DIM)
    n = 0
    per_lang = {l: {"docs": 0, "sum": np.zeros(DIM)} for l in a.langs}
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    heads = load_heads(a.classifier, device)
    scores = {l: {"mfwedu": [], "mdclm": []} for l in a.langs}
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (lang, v) in enumerate(ex.map(one, files)):
            s += v.sum(0)
            ss += (v * v).sum(0)
            n += len(v)
            per_lang[lang]["docs"] += len(v)
            per_lang[lang]["sum"] += v.sum(0)
            sc = score(heads, v.astype(np.float32), device)
            for k in sc:
                scores[lang][k].append(sc[k])
            if i % 200 == 0:
                print(f"[fw2 stats] {i + 1}/{len(files)} files, {n:,} docs, {time.time() - t0:.0f} s", flush=True)
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean * mean, 0))
    os.makedirs(os.path.dirname(os.path.abspath(a.stats)), exist_ok=True)
    np.savez(a.stats, mean=mean.astype(np.float32), std=std.astype(np.float32), n=n)
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    rep = {"pool": a.pool, "files": len(files), "docs": n, "seconds": time.time() - t0,
           "std_min": float(std.min()), "std_max": float(std.max()), "langs": {}}
    for l in a.langs:
        m = per_lang[l]["sum"] / max(1, per_lang[l]["docs"])
        rep["langs"][l] = {"docs": per_lang[l]["docs"],
                           "mean_dist_to_global_in_std": float(np.linalg.norm((m - mean) / std) / np.sqrt(DIM)),
                           **{f"{k}_q{int(100 * qq)}": float(np.quantile(np.concatenate(scores[l][k]), qq))
                              for k in ("mfwedu", "mdclm") for qq in q},
                           "mfwedu_ge3_frac": float(np.mean(np.concatenate(scores[l]["mfwedu"]) >= 3.0))}
    with open(os.path.splitext(a.stats)[0] + ".json", "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1), flush=True)


def need_pool(a):
    if not a.pool:
        raise SystemExit(f"--pool is required with --mode {a.mode}")


def need_tokenizer(a):
    if not a.tokenizer:
        raise SystemExit(f"--tokenizer is required with --mode {a.mode}")


def resolve_files(a):
    need_pool(a)
    if a.paths_file:
        with open(a.paths_file) as f:
            files = [l.strip() for l in f if l.strip()]
    else:
        files = list_files(a.pool, a.langs)
        if a.subset > 0:  # evenly spaced files of the full list (timing tests): languages in proportion
            files = files[:: max(1, len(files) // a.subset)][: a.subset]
    if a.tasks < 0:
        a.tasks = len(files)
    if a.tasks != len(files):
        raise SystemExit(f"{len(files)} input files but --tasks {a.tasks}: use one task per file")
    return files


def write_manifest(a, files):
    """Every job of every pass writes the same paths.txt / manifest.json (atomic replace): no job waits for another."""
    import datatrove
    from tokenizers import Tokenizer

    need_tokenizer(a)
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.mode}{a.local_rank_offset}"
    paths_file = os.path.join(a.out, "paths.txt")
    with open(paths_file + f".tmp{tag}", "w") as f:
        f.write("\n".join(files) + "\n")
    os.replace(paths_file + f".tmp{tag}", paths_file)
    tk = Tokenizer.from_file(a.tokenizer)
    man = {"pool": a.pool, "files": files, "langs": sorted({x.split("/")[0] for x in files}),
           "datatrove": getattr(datatrove, "__version__", None), "stats": a.stats, "tokenizer": a.tokenizer,
           "vocab": tk.get_vocab_size(), "eos": EOS, "eos_id": tk.token_to_id(EOS), "token_bytes": 4,
           "classifier": a.classifier, "embedding": "mmBERT-base mean over all tokens, <= 8192 tokens",
           "limit": a.limit, "git": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip()}
    with open(os.path.join(a.out, f"manifest.json.tmp{tag}"), "w") as f:
        json.dump(man, f, indent=1)
    os.replace(os.path.join(a.out, f"manifest.json.tmp{tag}"), os.path.join(a.out, "manifest.json"))
    return paths_file


def my_ranks(a, n):
    lo = a.local_rank_offset
    hi = n if a.local_tasks < 0 else min(n, lo + a.local_tasks)
    return list(range(lo, hi))


def mode_sidecar(a):
    import multiprocessing as mp

    files = resolve_files(a)
    write_manifest(a, files)
    if not a.stats or not os.path.exists(a.stats):
        raise SystemExit("--stats is required (run --mode stats first)")
    os.makedirs(os.path.join(a.out, "sidecar"), exist_ok=True)
    jobs = [(r, files[r], a.pool, a.out, a.limit) for r in my_ranks(a, len(files))]
    t0, docs, done = time.time(), 0, 0
    with mp.get_context("spawn").Pool(a.procs, initializer=_sidecar_init, initargs=(a.stats, a.classifier)) as pool:
        for rank, n, sec in pool.imap_unordered(_sidecar_file, jobs):
            done += 1
            if n >= 0:
                docs += n
            if done % 10 == 0 or done == len(jobs):
                print(f"[fw2 sidecar] {done}/{len(jobs)} files, {docs:,} docs, {time.time() - t0:.0f} s "
                      f"({docs / max(1e-9, time.time() - t0):,.0f} docs/s); last rank {rank}: {n:,} docs in {sec:.0f} s",
                      flush=True)


def mode_tokenize(a):
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import ParquetReader

    files = resolve_files(a)
    paths_file = write_manifest(a, files)
    reader = ParquetReader(a.pool, paths_file=paths_file, text_key="text", id_key="id", read_metadata=False,
                           batch_size=1000, limit=a.limit)
    tok, tok_kw = tokenizer_step(os.path.join(a.out, "tokens"), a.tokenizer, EOS, a.tok_batch)
    print("DocumentTokenizer args:", tok_kw, flush=True)
    ex = LocalPipelineExecutor(pipeline=[reader, tok], tasks=a.tasks, workers=a.workers,
                               logging_dir=os.path.join(a.out, "logs"), skip_completed=True,
                               local_tasks=a.local_tasks, local_rank_offset=a.local_rank_offset)
    ex.run()



def ds_for(tok_dir, rank):
    import glob

    hits = glob.glob(os.path.join(tok_dir, "**", f"*{rank:05d}*.ds"), recursive=True)
    if len(hits) != 1:
        raise RuntimeError(f"rank {rank}: expected one .ds file, found {hits}")
    return hits[0]


def mode_verify(a):
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    man = json.load(open(os.path.join(a.out, "manifest.json")))
    tk = Tokenizer.from_file(man["tokenizer"])
    eos, vocab = man["eos_id"], man["vocab"]
    st = np.load(man["stats"])
    mean, std = st["mean"].astype(np.float32), st["std"].astype(np.float32)
    files = man["files"]
    sample_ranks = set(range(0, len(files), max(1, len(files) // a.sample_shards)))
    heads, device = None, "cpu"
    if a.sample:
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        heads = load_heads(man["classifier"], device)

    def check(rank):
        try:
            ds = ds_for(os.path.join(a.out, "tokens"), rank)
        except RuntimeError as e:
            return {"rank": rank, "error": str(e)}
        idx = np.fromfile(ds + ".index", dtype=np.uint64).astype(np.int64)
        nbytes = os.path.getsize(ds)
        tok = np.memmap(ds, dtype=np.uint32, mode="r")
        cnt = json.load(open(os.path.join(a.out, "sidecar", f"{rank:05d}.count.json")))["docs"]
        phi_n = os.path.getsize(os.path.join(a.out, "sidecar", f"{rank:05d}.phi.f16")) // (DIM * 2)
        meta_n = pq.ParquetFile(os.path.join(a.out, "sidecar", f"{rank:05d}.meta.parquet")).metadata.num_rows
        r = {"rank": rank, "file": files[rank], "docs": int(cnt), "tokens": int(idx[-1]) if len(idx) else 0,
             "counts_ok": len(idx) == cnt == phi_n == meta_n, "bytes_ok": nbytes == 4 * (int(idx[-1]) if len(idx) else 0),
             "eos_ok": bool((tok[idx - 1] == eos).all()) if len(idx) else True,
             # every id < vocab: read all tokens on the sampled shards only (the full pool is several TB)
             "max_id_ok": bool(int(tok.max()) < vocab) if (len(tok) and rank in sample_ranks) else True}
        return r

    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(check, range(len(files))))
    bad = [r for r in res if "error" in r or not (r["counts_ok"] and r["bytes_ok"] and r["eos_ok"] and r["max_id_ok"])]
    rep = {"shards": len(res), "docs": int(sum(r.get("docs", 0) for r in res)),
           "tokens": int(sum(r.get("tokens", 0) for r in res)), "bad": bad[:50], "n_bad": len(bad), "samples": []}
    by_lang = {}
    for r in res:
        if "error" in r:
            continue
        L = by_lang.setdefault(r["file"].split("/")[0], {"shards": 0, "docs": 0, "tokens": 0})
        L["shards"] += 1
        L["docs"] += r["docs"]
        L["tokens"] += r["tokens"]
    rep["by_lang"] = by_lang
    if a.sample:
        for rank in sorted(sample_ranks):
            ds = ds_for(os.path.join(a.out, "tokens"), rank)
            idx = np.concatenate([[0], np.fromfile(ds + ".index", dtype=np.uint64).astype(np.int64)])
            tok = np.memmap(ds, dtype=np.uint32, mode="r")
            phi = np.memmap(os.path.join(a.out, "sidecar", f"{rank:05d}.phi.f16"), dtype=np.float16, mode="r").reshape(-1, DIM)
            meta = pq.read_table(os.path.join(a.out, "sidecar", f"{rank:05d}.meta.parquet"),
                                 columns=["id", "mfwedu", "mdclm"]).to_pydict()
            src = pq.ParquetFile(os.path.join(man["pool"], files[rank]))
            rows = src.read_row_group(0, columns=["id", "text"]).to_pydict()
            n = min(a.sample, len(rows["id"]), len(idx) - 1)
            v = embeddings_of_row_group(src, 0)[:n].astype(np.float32)
            ids_ok = rows["id"][:n] == meta["id"][:n]
            tok_ok = sum(list(tok[idx[i]:idx[i + 1]]) == tk.encode(rows["text"][i], add_special_tokens=False).ids + [eos]
                         for i in range(n))
            phi_err = float(np.max(np.abs(phi[:n].astype(np.float32) - (v - mean) / std)))
            sc = score(heads, v, device)
            rep["samples"].append({"rank": rank, "file": files[rank], "n": n, "ids_in_order": ids_ok,
                                   "tokens_equal_retokenized": int(tok_ok), "phi_max_abs_err_std_units": phi_err,
                                   "mfwedu_max_abs_diff": float(np.max(np.abs(sc["mfwedu"] - np.array(meta["mfwedu"][:n])))),
                                   "mdclm_max_abs_diff": float(np.max(np.abs(sc["mdclm"] - np.array(meta["mdclm"][:n]))))})
            print(f"[fw2 verify] sample {rep['samples'][-1]}", flush=True)
    with open(os.path.join(a.out, "verify.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "samples"}, indent=1), flush=True)
    if bad:
        raise SystemExit(f"[fw2 verify] {len(bad)} bad shards")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("stats", "sidecar", "tokenize", "verify"), required=True)
    p.add_argument("--pool", default="", help="stats / sidecar / tokenize: the embedded FineWeb2 copy, "
                   "POOL/<lang>/*.parquet with text, id and metadata.embeddings (768 floats)")
    p.add_argument("--langs", default=",".join(LANGS))
    p.add_argument("--out", default=None, help="the tokenized pool (sidecar / tokenize / verify)")
    p.add_argument("--stats", default=None, help="npz with the per-coordinate mean / std of the raw mmBERT vectors")
    p.add_argument("--tokenizer", default="", help="sidecar / tokenize: tokenizer.json of the multilingual tokenizer")
    p.add_argument("--classifier", default=CLASSIFIER,
                   help="Hugging Face id or local directory with the quality heads mfwedu.pt and mdclm.pt")
    p.add_argument("--paths-file", default=None, help="input files relative to --pool (default: all of --langs)")
    p.add_argument("--tasks", type=int, default=-1, help="= number of input files (one task per file)")
    p.add_argument("--workers", type=int, default=64, help="tokenize: datatrove workers; stats / verify: threads")
    p.add_argument("--procs", type=int, default=32, help="sidecar: processes (spread over the GPUs)")
    p.add_argument("--local-tasks", type=int, default=-1)
    p.add_argument("--local-rank-offset", type=int, default=0)
    p.add_argument("--limit", type=int, default=-1, help="documents per task (timing tests); -1 = all")
    p.add_argument("--subset", type=int, default=0, help="tokenize: only N evenly spaced input files (timing tests)")
    p.add_argument("--batch", type=int, default=1024, help="documents per classifier batch")
    p.add_argument("--tok-batch", type=int, default=1000,
                   help="tokenize: documents per encode_batch call (datatrove's default of 10,000 needs far more "
                        "memory per worker)")
    p.add_argument("--sample", type=int, default=200, help="verify: documents re-checked per sampled shard")
    p.add_argument("--sample-shards", type=int, default=12, help="verify: about this many shards sampled")
    a = p.parse_args()
    a.langs = a.langs.split(",") if isinstance(a.langs, str) else a.langs
    if a.mode == "stats":
        mode_stats(a)
    elif a.mode == "sidecar":
        mode_sidecar(a)
    elif a.mode == "tokenize":
        mode_tokenize(a)
    else:
        mode_verify(a)


if __name__ == "__main__":
    main()
