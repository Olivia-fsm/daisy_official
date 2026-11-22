"""Shared fixtures of the tests: synthetic indexes, pools and target tasks, tiny configurations, and runners of the
trainer (in-process, CPU)."""

import contextlib
import io
import json
import os
import sys
from unittest import mock

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from daisy import config  # noqa: E402
from daisy.distributed.single import SingleNodeBackend  # noqa: E402
from daisy.models.llama import Llama  # noqa: E402
from daisy.train.trainer import train_daisy  # noqa: E402

V, T, EOT = 50304, 64, 50256


def make_index(d, rng, n=400, D=3):
    """A synthetic index of n documents with D topics: tokens, offsets, phi, held-out rows and topics, hashes."""
    lens = rng.integers(5, 150, size=n)
    toks = [np.append(rng.integers(1, 1000, size=L - 1), EOT).astype(np.uint16) for L in lens]
    np.concatenate(toks).tofile(os.path.join(d, "tokens.u16"))
    np.save(os.path.join(d, "offsets.i64.npy"), np.concatenate([[0], np.cumsum(lens)]).astype(np.int64))
    rng.standard_normal((n, 768)).astype(np.float16).tofile(os.path.join(d, "phi.f16"))
    held = np.sort(rng.choice(n, size=10 * D, replace=False))
    np.save(os.path.join(d, "heldout_rows.npy"), held)
    np.save(os.path.join(d, "heldout_topic.npy"), (np.arange(len(held)) % D).astype(np.uint8))
    mask = np.ones(n, np.uint8)
    mask[held] = 0
    np.save(os.path.join(d, "train_mask.u8.npy"), mask)
    np.save(os.path.join(d, "text_hash.npy"), rng.integers(1, 2**62, size=n).astype(np.uint64))
    np.save(os.path.join(d, "url_hash.npy"), rng.integers(1, 2**62, size=n).astype(np.uint64))


def make_benchmarks(d, rng, tasks=("taska", "taskb"), pad=64):
    """Synthetic target tasks with a train and a val split: items padded with EOS to a multiple of `pad`."""
    for t in tasks:
        os.makedirs(os.path.join(d, t))
        for split in ("train", "val"):
            items, lens = [], []
            for _ in range(60):
                L = int(rng.integers(5, 40))
                ids = np.append(rng.integers(1, 1000, size=L), EOT)
                P = -(-len(ids) // pad) * pad
                arr = np.full(P, EOT, np.uint16)
                arr[: len(ids)] = ids
                items.append(arr)
                lens.append(P)
            np.concatenate(items).tofile(os.path.join(d, t, f"{split}.bin"))
            with open(os.path.join(d, t, f"{split}.len"), "wb") as f:
                np.save(f, np.array(lens, dtype=np.uint16))


BASE = ["--n_embd", "32", "--n_head", "2", "--n_layer", "2", "--sequence_length", str(T), "--vocab_size", str(V),
        "--dropout", "0.0", "--device", "cpu", "--dtype", "float32", "--batch_size", "2", "--acc_steps", "1",
        "--lr", "3e-3", "--scheduler", "none", "--grad_clip", "0.5", "--log_interval", "4",
        "--latest_ckpt_interval", "10", "--daisy_refresh", "4", "--daisy_val_docs_per_topic", "4",
        "--daisy_val_max_doc_tokens", "64", "--daisy_val_rows_per_batch", "2", "--scheduler_total_steps", "28",
        "--experiment_name", "test"]


def cfg_for(idx, bench, iters, extra=()):
    """A DAiSY run on a compressed schedule: labels from step 8 on every step-th of --daisy_label_every (default 4),
    the scorer on at 16 and refitted every 2 steps, refresh R = 4, two target tasks."""
    return config.parse_args([*BASE, "--iterations", str(iters), "--daisy_index_dir", idx,
                              "--daisy_learn", "--daisy_label_start", "8", "--daisy_scorer_start", "16",
                              "--daisy_scorer_update_every", "2", "--daisy_metrics_every", "4",
                              "--daisy_targets", "taska,taskb", "--daisy_target_dir", bench,
                              "--daisy_target_tokens", "260", *extra])


def plain_cfg(idx, iters, extra=()):
    """Uniform training without a learner, on the same trainer settings as cfg_for."""
    return config.parse_args([*BASE, "--iterations", str(iters), "--daisy_index_dir", idx,
                              "--daisy_targets", "taska,taskb", *extra])


def run(cfg, exp):
    """The trainer on cfg with a freshly initialized tiny model. -> the trained model."""
    torch.manual_seed(0)
    np.random.seed(0)
    model = Llama(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.8, 0.999), weight_decay=0.1)
    train_daisy(model, opt, None, exp, SingleNodeBackend(cfg), cfg)
    return model


def records(exp, name="metrics.jsonl"):
    with open(os.path.join(exp, "daisy", name)) as f:
        return [json.loads(line) for line in f]


def run_main(argv, env=None, record=None):
    """train.py in-process (what `python train.py ARGV` does on one process). env: DAISY_* variables of the run.
    record: a list that receives [loss, x, y, doc_ids] (numpy) of every training forward."""
    import train

    real = Llama.forward

    def recording_forward(self, *a, **kw):
        out = real(self, *a, **kw)
        if record is not None and self.training and out.get("loss") is not None:
            record.append([float(out["loss"].detach()), a[0].detach().cpu().numpy().copy(),
                           kw["targets"].detach().cpu().numpy().copy(), kw["doc_ids"].detach().cpu().numpy().copy()])
        return out

    clean = {k: v for k, v in os.environ.items() if not k.startswith("DAISY_")}
    with mock.patch.dict(os.environ, {**clean, **(env or {})}, clear=True), \
            mock.patch.object(Llama, "forward", recording_forward), contextlib.redirect_stdout(io.StringIO()):
        train.main(config.parse_args(argv))


def make_run_data(root, n_shards=8, per_shard=1200, seed=1234):
    """Everything a run of train.py reads, synthetic: an index (the held-out web-text documents and their hashes), a
    tokenized pool of n_shards shards with clustered phi (pool document g draws its tokens from [1 + 5 g, 6 + 5 g), so
    any token names its document), two target tasks and a unit similarity scorer.
    -> ({"idx", "pool", "bench", "w_static"}, {document serial g: pool key})."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    idx, pool, bench = (os.path.join(root, d) for d in ("idx", "pool", "bench"))
    os.makedirs(idx)
    make_index(idx, rng, n=200)
    held = np.load(os.path.join(idx, "heldout_rows.npy"))
    th_idx, uh_idx = np.load(os.path.join(idx, "text_hash.npy")), np.load(os.path.join(idx, "url_hash.npy"))
    os.makedirs(os.path.join(pool, "tokens"))
    os.makedirs(os.path.join(pool, "sidecar"))
    centers = rng.standard_normal((40, 768))
    keys, g = {}, 0
    for s in range(n_shards):
        lens = rng.integers(3, 60, size=per_shard)
        dtoks = []
        for i, L in enumerate(lens):
            dtoks.append(np.append(rng.integers(1 + 5 * g, 6 + 5 * g, size=L - 1), EOT).astype(np.uint16))
            keys[g] = (s << 32) | i
            g += 1
        np.concatenate(dtoks).tofile(os.path.join(pool, "tokens", f"{s:05d}_unshuffled.ds"))
        np.cumsum(lens).astype(np.uint64).tofile(os.path.join(pool, "tokens", f"{s:05d}_unshuffled.ds.index"))
        phi = centers[rng.integers(0, len(centers), size=per_shard)] + 0.35 * rng.standard_normal((per_shard, 768))
        phi = (phi - phi.mean()) / phi.std()
        phi.astype(np.float16).tofile(os.path.join(pool, "sidecar", f"{s:05d}.phi.f16"))
        th = rng.integers(1, 2**62, size=per_shard, dtype=np.int64).astype(np.uint64)
        uh = rng.integers(1, 2**62, size=per_shard, dtype=np.int64).astype(np.uint64)
        if s == 0:                                     # pool copies of held-out documents: never trained on
            th[:2] = th_idx[held[:2]]
            uh[2] = uh_idx[held[2]]
        pq.write_table(pa.table({"id": [f"{s}-{i}" for i in range(per_shard)], "text_hash": th, "url_hash": uh}),
                       os.path.join(pool, "sidecar", f"{s:05d}.meta.parquet"))
    with open(os.path.join(pool, "manifest.json"), "w") as f:
        json.dump({"files": [f"f{s}.parquet" for s in range(n_shards)], "pool": "synthetic"}, f)
    make_benchmarks(bench, rng)
    w_static = os.path.join(root, "w_static.npy")
    np.save(w_static, np.append(centers[3] / np.linalg.norm(centers[3]), 0.0).astype(np.float32))
    return {"idx": idx, "pool": pool, "bench": bench, "w_static": w_static}, keys


def main_flags(paths, name, root, iters=32, extra=()):
    """The flags of scripts/train.sh with toy sizes and a compressed schedule (labels from step 8 on every 2nd step,
    the scorer on at 16 and refitted every 4 steps, refresh R = 4, two target tasks), over the synthetic pool."""
    return ["--n_embd", "8", "--n_head", "2", "--n_layer", "2", "--sequence_length", "32", "--vocab_size", str(V),
            "--batch_size", "12", "--acc_steps", "2", "--scheduler_total_steps", "48", "--warmup_steps", "4",
            "--lr", "3e-3", "--latest_ckpt_interval", "10", "--permanent_ckpt_interval", "20", "--device", "cpu",
            "--dtype", "float32", "--log_interval", "1", "--daisy_index_dir", paths["idx"],
            "--daisy_pool", paths["pool"], "--daisy_pool_open_shards", "4", "--daisy_max_doc_tokens", "48",
            "--daisy_refresh", "4", "--daisy_val_docs_per_topic", "8", "--daisy_val_max_doc_tokens", "64",
            "--daisy_val_rows_per_batch", "2", "--daisy_label_start", "8", "--daisy_scorer_start", "16",
            "--daisy_scorer_update_every", "4", "--daisy_metrics_every", "8", "--daisy_label_every", "2",
            "--daisy_targets", "taska,taskb", "--daisy_target_dir", paths["bench"], "--daisy_target_tokens", "260",
            "--daisy_eval_dir", paths["bench"], "--daisy_eval_tokens", "130",
            "--iterations", str(iters), "--experiment_name", name, "--results_base_folder", os.path.join(root, "exps"),
            *extra]
