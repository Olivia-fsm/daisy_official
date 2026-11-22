#!/usr/bin/env python
"""Tokenize the whole embedded pool with datatrove, keeping every document's embedding aligned with its tokens: the
tokenized pool that training draws from (--daisy_pool).

Input: the embedded pool of daisy/index/build_index.py (--pool: POOL/<dump>/*.parquet with the columns `text`, `id`
and the embedding `embeddings`, 768 floats per document, with `url`, `dump`, `token_count`, at the top level or
inside the `metadata` struct; an embedding with one vector per chunk is reduced to its first chunk), and the index's
stats.npz (--stats: the per-coordinate mean / std that standardize phi, so that the pool's phi is in the index's space).

One datatrove task per input parquet file (the reader shards the sorted file list as files[rank::tasks]), so task
`rank` reads exactly manifest["files"][rank]. Pipeline per task:

  ParquetReader (text, id, metadata incl. the embedding)
    -> PhiSidecar: in document order, writes the sidecar files below and drops the embedding from the document before
       it moves on
    -> DocumentTokenizer: GPT-2, EOS appended to every document, document index on, datatrove's document shuffle OFF,
       so document j of the .ds file is row j of the sidecar. The training loader shuffles instead
       (daisy/data/pool.py).

Output (--out), one shard per input file, `rank` = the file's position in the manifest:
  manifest.json                      {"pool", "files" (relative to the pool, in task order), "datatrove", "stats",
                                     "tokenizer"}; written by the job with --local-rank-offset 0
  tokens/{rank:05d}_unshuffled.ds    the shard's tokens, uint16, documents back to back
  tokens/{rank:05d}_unshuffled.ds.index   uint64 cumulative document end offsets (in tokens)
  sidecar/{rank:05d}.phi.f16         standardized phi, fp16, 768 per document, in document order
  sidecar/{rank:05d}.meta.parquet    id, url, dump, 64-bit text_hash and url_hash, xlmr_tokens (the input's
                                     token_count), n_chunks
  sidecar/{rank:05d}.count.json      {"rank", "docs"}
  logs/                              datatrove's logs and completion markers (a rerun skips completed tasks)
The .ds names are datatrove's; the loaders accept any single tokens/**/*{rank:05d}*.ds per shard.
daisy/tokenize/verify_tokenized.py checks the result. Needs the datatrove package.

  python daisy/tokenize/datatrove_tokenize.py --pool POOL --stats INDEX/stats.npz --out OUT --tasks N --workers W
      [--paths-file F]                              # a list of parquet paths relative to POOL instead of all files
      [--local-tasks K --local-rank-offset O]       # split the tasks over several nodes / jobs
"""

import argparse
import hashlib
import inspect
import json
import os

import numpy as np


def h64(s):
    """Stable 64-bit hash (same as daisy/index/build_index.py); 0 is reserved for empty strings."""
    if not s:
        return 0
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "surrogatepass"), digest_size=8).digest(), "little") or 1


def make_sidecar_step():
    from datatrove.pipeline.base import PipelineStep

    class PhiSidecar(PipelineStep):
        name = "phi sidecar"
        type = "DAiSY"

        def __init__(self, out_dir, stats_path, dim=768):
            super().__init__()
            self.out_dir, self.stats_path, self.dim = out_dir, stats_path, dim

        def run(self, data, rank: int = 0, world_size: int = 1):
            import pyarrow as pa
            import pyarrow.parquet as pq

            st = np.load(self.stats_path)
            mean, std = st["mean"].astype(np.float32), st["std"].astype(np.float32)
            os.makedirs(self.out_dir, exist_ok=True)
            phi_path = os.path.join(self.out_dir, f"{rank:05d}.phi.f16")
            cols = {k: [] for k in ("id", "url", "dump", "text_hash", "url_hash", "xlmr_tokens", "n_chunks")}
            n = 0
            with open(phi_path + ".tmp", "wb") as fphi:
                for doc in data:
                    md = doc.metadata
                    inner = md.get("metadata") if isinstance(md.get("metadata"), dict) else {}
                    emb = md.pop("embeddings", None)
                    if emb is None:
                        emb = inner.pop("embeddings", None)
                    v = np.asarray(emb, dtype=np.float32)
                    n_chunks = 1
                    if v.ndim == 2:  # one vector per 512-token chunk -> the first chunk (same space)
                        n_chunks = v.shape[0]
                        v = v[0]
                    if v.shape != (self.dim,):
                        raise ValueError(f"doc {doc.id}: embedding shape {v.shape}")
                    fphi.write(((v - mean) / std).astype(np.float16).tobytes())
                    url = md.get("url", inner.get("url", ""))
                    cols["id"].append(doc.id)
                    cols["url"].append(url)
                    cols["dump"].append(md.get("dump", inner.get("dump", "")))
                    cols["text_hash"].append(h64(doc.text))
                    cols["url_hash"].append(h64(url))
                    cols["xlmr_tokens"].append(int(md.get("token_count", inner.get("token_count", -1)) or -1))
                    cols["n_chunks"].append(n_chunks)
                    n += 1
                    self.stat_update("docs")
                    yield doc
            pq.write_table(pa.table({
                "id": cols["id"], "url": cols["url"], "dump": cols["dump"],
                "text_hash": np.array(cols["text_hash"], dtype=np.uint64),
                "url_hash": np.array(cols["url_hash"], dtype=np.uint64),
                "xlmr_tokens": np.array(cols["xlmr_tokens"], dtype=np.int32),
                "n_chunks": np.array(cols["n_chunks"], dtype=np.int16)}),
                os.path.join(self.out_dir, f"{rank:05d}.meta.parquet"), compression="zstd")
            os.replace(phi_path + ".tmp", phi_path)
            with open(os.path.join(self.out_dir, f"{rank:05d}.count.json"), "w") as f:
                json.dump({"rank": rank, "docs": n}, f)

    return PhiSidecar


def tokenizer_step(out_dir, tokenizer):
    """DocumentTokenizer with document shuffling off; the argument name changed across datatrove versions."""
    from datatrove.pipeline.tokens import DocumentTokenizer

    sig = inspect.signature(DocumentTokenizer.__init__).parameters
    kw = dict(output_folder=out_dir, tokenizer_name_or_path=tokenizer, eos_token="<|endoftext|>", save_index=True)
    if "shuffle_documents" in sig:
        kw["shuffle_documents"] = False
    elif "shuffle" in sig:
        kw["shuffle"] = False
    else:
        raise RuntimeError(f"cannot turn off DocumentTokenizer shuffling; signature: {list(sig)}")
    if "max_tokens_per_file" in sig:
        kw["max_tokens_per_file"] = None  # one .ds file per task
    return DocumentTokenizer(**kw), kw


def main():
    import datatrove
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import ParquetReader

    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="the tokenized pool")
    p.add_argument("--stats", required=True, help="npz with per-coordinate mean/std of the raw XLM-R vectors "
                   "(INDEX/stats.npz)")
    p.add_argument("--pool", required=True, help="the embedded pool: POOL/<dump>/*.parquet")
    p.add_argument("--paths-file", default=None, help="optional list of parquet paths relative to --pool")
    p.add_argument("--tasks", type=int, required=True, help="= number of input files (one task per file)")
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--local-tasks", type=int, default=-1, help="tasks this job runs (-1 = all)")
    p.add_argument("--local-rank-offset", type=int, default=0, help="the first task this job runs")
    p.add_argument("--tokenizer", default="openai-community/gpt2")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    reader = ParquetReader(a.pool, paths_file=a.paths_file, glob_pattern=None if a.paths_file else "*/*.parquet",
                           recursive=True, text_key="text", id_key="id", read_metadata=True, batch_size=1000)
    # manifest: task rank -> input file, exactly as the reader shards (sorted list, files[rank::tasks])
    if a.local_rank_offset == 0:
        from datatrove.io import get_datafolder

        df = get_datafolder(a.pool)
        if a.paths_file:
            with open(a.paths_file) as f:
                files = [l.strip() for l in f if l.strip()]
        else:
            files = df.list_files(glob_pattern="*/*.parquet", recursive=True)
        if len(files) != a.tasks:
            raise SystemExit(f"{len(files)} input files but --tasks {a.tasks}: use one task per file")
        with open(os.path.join(a.out, "manifest.json"), "w") as f:
            json.dump({"pool": a.pool, "files": files, "datatrove": datatrove.__version__ if hasattr(datatrove, "__version__")
                       else None, "stats": a.stats, "tokenizer": a.tokenizer}, f)
    tok, tok_kw = tokenizer_step(os.path.join(a.out, "tokens"), a.tokenizer)
    print("DocumentTokenizer args:", tok_kw, flush=True)
    sidecar = make_sidecar_step()(os.path.join(a.out, "sidecar"), a.stats)
    ex = LocalPipelineExecutor(pipeline=[reader, sidecar, tok], tasks=a.tasks, workers=a.workers,
                               logging_dir=os.path.join(a.out, "logs"), skip_completed=True,
                               local_tasks=a.local_tasks, local_rank_offset=a.local_rank_offset)
    ex.run()


if __name__ == "__main__":
    main()
