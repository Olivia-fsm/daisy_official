"""The MATES baseline (daisy/mates/, docs/mates_baseline.md) on CPU: tiny Llama, synthetic pool and target tasks.

Units:
  * reserved probe shards are never opened by the pool stream (and reserve 0 changes nothing);
  * StageListStream packs the selected list with carry-over, resumes exactly, ends empty; MatesStream stage 1 is the
    pool stream; the candidate draw comes from the live stream, never twice, never from a reserved shard;
  * Gumbel-top-k: exactly the token budget (the last document cut), inclusion frequencies of Plackett-Luce;
  * probe documents: reserved shards only, held-out hashes excluded, fresh per stage, seeded;
  * the prober leaves model and optimizer bit-identical; with SGD on a toy model its oracle equals the loss change of
    the step computed independently (and its first-order form); the reference is the train-split target rows;
  * (transformers) the influence model's chunking / pooling equals a direct computation; bucketed scoring with the
    all-pad chunk skipped equals MATES's padded-1024 layout; fine-tuning learns and keeps the best validation model;
  * (tokenizers + the local HF cache) GPT-2 decode + WordPiece equals transformers' tokenizers; the worker pool equals
    in-process encoding.
End to end (DaisyTrainer --mates, U = 8 steps, refresh 4): stage 1 is bit-identical to the uniform run;
artifacts and records of every boundary; no candidate twice, trained within selected within candidates, probes
disjoint from candidates, reserved shards never opened, exactly the token budget per stage; exact resume across a
boundary and mid-stage, reuse of a boundary's artifacts; --mates_init_from = the run from scratch; the GPU-hour budget
stop (final checkpoint, evaluation, finished.json; a resubmitted job does nothing); 2 ranks (gloo DDP).
Without transformers / tokenizers the end-to-end runs use a stand-in backbone and encoder with the same interfaces
(the tests of the real BERT model and tokenizers are skipped); without flex_attention (torch < 2.5) document-masked
attention runs as a dense masked SDPA (the same mask).

    cd tests && python3 -W ignore -m unittest test_mates
"""

import contextlib
import glob
import importlib.util
import itertools
import json
import os
import shutil
import socket
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "tests"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from daisy import config  # noqa: E402
from daisy.data.pool import DsPool, ShardDocStream  # noqa: E402
from daisy.data.prefetch import Prefetcher  # noqa: E402
from daisy.mates import influence, text  # noqa: E402
from daisy.mates.probes import Prober, doc_row, per_rank, probe_documents, reference_batches  # noqa: E402
from daisy.mates.select import gumbel_top_k  # noqa: E402
from daisy.mates.streams import MatesStream, StageListStream, enumerate_candidates  # noqa: E402
from helpers import make_benchmarks, make_index  # noqa: E402
from test_pool import make_pool  # noqa: E402

EOS = 50256
HAVE_TF = importlib.util.find_spec("transformers") is not None
HAVE_TOK = importlib.util.find_spec("tokenizers") is not None
try:
    from torch.nn.attention.flex_attention import flex_attention  # noqa: F401
    HAVE_FLEX = True
except ImportError:
    HAVE_FLEX = False
HUB = os.path.expanduser("~/.cache/huggingface/hub")
BERT_VOCAB = sorted(glob.glob(os.path.join(HUB, "models--prajjwal1--bert-tiny", "snapshots", "*", "vocab.txt")))
GPT2_JSON = sorted(glob.glob(os.path.join(HUB, "models--gpt2", "snapshots", "*", "tokenizer.json")) +
                   glob.glob(os.path.join(HUB, "models--openai-community--gpt2", "snapshots", "*", "tokenizer.json")))
REAL_TEXT = HAVE_TOK and bool(BERT_VOCAB) and bool(GPT2_JSON)
T = 64


# ---------------------------------------------------------------------------------------------- stand-ins
class FakeConfig(types.SimpleNamespace):
    def to_json_file(self, path):
        with open(path, "w") as f:
            json.dump(vars(self), f)


class FakeBert(nn.Module):
    """A BertModel stand-in (no transformers): embeddings, a masked context mix, LayerNorm -> last_hidden_state."""

    def __init__(self, vocab=30522, hidden=16):
        super().__init__()
        self.config = FakeConfig(hidden_size=hidden, hidden_dropout_prob=0.1, classifier_dropout=None,
                                 initializer_range=0.02)
        self.emb, self.pos = nn.Embedding(vocab, hidden), nn.Embedding(512, hidden)
        self.lin, self.ln = nn.Linear(hidden, hidden), nn.LayerNorm(hidden)

    def forward(self, input_ids, attention_mask):
        h = self.emb(input_ids) + self.pos(torch.arange(input_ids.shape[1], device=input_ids.device))
        m = attention_mask.unsqueeze(-1).float()
        ctx = (h * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1.0)
        return types.SimpleNamespace(last_hidden_state=self.ln(torch.tanh(self.lin(h + ctx))))


class FakeTextEncoder:
    """A TextEncoder stand-in (no tokenizers): every GPT-2 token becomes 8 WordPiece ids, [CLS] ... [SEP], <= 1024."""

    def __init__(self, gpt2_json, bert_dir, max_len=1024):
        self.max_len = max_len

    def encode(self, docs):
        out = []
        for d in docs:
            d = np.asarray(d, dtype=np.int64)
            body = (1000 + (np.repeat(d[d != EOS], 8) * 7 + np.tile(np.arange(8), (d != EOS).sum())) % 29000)
            out.append(np.concatenate([[101], body[:self.max_len - 2], [102]]).astype(np.uint16))
        return out


def tiny_bert_config():
    from transformers import BertConfig

    return BertConfig(vocab_size=30522, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                      intermediate_size=32, max_position_embeddings=512)


_BUILD_BERT = influence.build_bert


def test_bert(bert_dir=None):
    return _BUILD_BERT(config=tiny_bert_config()) if HAVE_TF else FakeBert()


def _dense_block_mask(doc_ids):
    T_ = doc_ids.shape[1]
    causal = torch.tril(torch.ones(T_, T_, dtype=torch.bool, device=doc_ids.device))
    return ((doc_ids[:, :, None] == doc_ids[:, None, :]) & causal)[:, None]


def _dense_attention(q, k, v, mask):
    return nn.functional.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v, attn_mask=mask)


@contextlib.contextmanager
def use_fakes():
    """No pretrained BERT in tests: a tiny random one (or the stand-in); the stand-in encoder without the HF cache;
    without flex_attention (torch < 2.5) document-masked attention as a dense masked SDPA (same masking)."""
    with contextlib.ExitStack() as st:
        st.enter_context(mock.patch.object(influence, "build_bert", lambda d=None, config=None: test_bert()))
        if not REAL_TEXT:
            st.enter_context(mock.patch.object(text, "TextEncoder", FakeTextEncoder))
        if not HAVE_FLEX:
            import daisy.models.llama as llama_mod

            st.enter_context(mock.patch.object(llama_mod, "doc_causal_block_mask", _dense_block_mask))
            st.enter_context(mock.patch.object(llama_mod, "doc_masked_attention", _dense_attention))
        yield


# ---------------------------------------------------------------------------------------------- the world
def make_world(root, n_shards=12):
    rng = np.random.default_rng(0)
    paths = {k: os.path.join(root, k) for k in ("idx", "bench", "pool", "bert")}
    for k in ("idx", "pool", "bert"):
        os.makedirs(paths[k])
    make_index(paths["idx"], rng)
    make_benchmarks(paths["bench"], rng)
    make_pool(paths["pool"], np.random.default_rng(1), n_shards=n_shards, docs=(80, 140))
    if BERT_VOCAB:
        shutil.copy(BERT_VOCAB[-1], os.path.join(paths["bert"], "vocab.txt"))
    paths["gpt2"] = GPT2_JSON[-1] if GPT2_JSON else "unused"
    return paths


def mates_cfg(paths, iters, extra=(), mates=True):
    """A MATES run (U = 8 steps; mates=False: the uniform run it starts from) over the synthetic pool."""
    flags = [
        "--experiment_name", "test", "--n_embd", "32", "--n_head", "2", "--n_layer", "2", "--sequence_length", str(T),
        "--vocab_size", "50304", "--dropout", "0.0", "--device", "cpu", "--dtype", "float32",
        "--batch_size", "4", "--acc_steps", "1", "--iterations", str(iters), "--lr", "3e-3",
        "--scheduler", "none", "--grad_clip", "0.5", "--log_interval", "4", "--latest_ckpt_interval", "8",
        "--permanent_ckpt_interval", "8",
        "--daisy_index_dir", paths["idx"], "--daisy_refresh", "4",
        "--daisy_val_docs_per_topic", "4", "--daisy_val_max_doc_tokens", "64",
        "--daisy_val_rows_per_batch", "2",
        "--daisy_pool", paths["pool"], "--daisy_pool_open_shards", "4",
        "--daisy_targets", "taska,taskb", "--daisy_target_dir", paths["bench"],
        "--daisy_target_tokens", "260", "--daisy_eval_dir", paths["bench"], "--daisy_eval_tokens", "130"]
    if mates:
        flags += ["--mates", "--mates_stage_steps", "8", "--mates_probes_first", "24", "--mates_probes_later", "30",
                  "--mates_probe_shards", "2", "--mates_ref_rows_per_batch", "2", "--mates_bert_dir", paths["bert"],
                  "--mates_gpt2_tokenizer", paths["gpt2"], "--mates_bert_epochs", "2", "--mates_bert_batch", "8",
                  "--mates_bert_micro_batch", "4", "--mates_bert_eval_every", "2", "--mates_score_bucket", "32",
                  "--mates_score_tokens", "1024", "--mates_tok_workers", "0"]
    return config.parse_args(flags + list(extra))


def run_mates(cfg, exp, backend=None, record=None):
    """DaisyTrainer on cfg (scheduler none) with a freshly initialized tiny model. record: list that receives
    (iter, stage, rows, starts, ends) of every trained micro-batch. -> the trained (unwrapped) model."""
    from daisy.distributed.single import SingleNodeBackend
    from daisy.models.llama import Llama

    from daisy.train import trainer as trainer_mod

    torch.manual_seed(0)
    np.random.seed(0)
    model = Llama(cfg)
    if backend is not None:
        model = backend.transform_model(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.8, 0.999), weight_decay=0.1)
    real = trainer_mod.DaisyTrainer._micro_step

    def rec(self, micro, mb):
        st = self.stream.stream if isinstance(self.stream, Prefetcher) else self.stream
        record.append((self.it, getattr(st, "stage", 1), mb["rows"].copy(), mb["starts"].copy(), mb["ends"].copy()))
        return real(self, micro, mb)

    with (mock.patch.object(trainer_mod.DaisyTrainer, "_micro_step", rec) if record is not None
          else contextlib.nullcontext()):
        trainer_mod.train_daisy(model, opt, None, exp, backend or SingleNodeBackend(cfg), cfg)
    return backend.get_raw_model(model) if backend is not None else model


def same_weights(test, a, b, msg=""):
    for (n, p1), p2 in zip(a.named_parameters(), b.parameters()):
        test.assertTrue(torch.equal(p1, p2), f"{msg} {n}")


def jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------------------------- units: streams
class StreamTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        make_pool(self.root, np.random.default_rng(0), n_shards=10, docs=(30, 60))
        self.pool = DsPool(self.root)

    def tearDown(self):
        shutil.rmtree(self.root)

    def stream(self, reserve=0, rank=0, world=1):
        return ShardDocStream(self.pool, 16, 48, seed=3, rank=rank, world=world, n_open=3, reserve=reserve)

    def test_reserved_shards_never_opened(self):
        a, b = self.stream(), self.stream(reserve=3)
        self.assertEqual(b.reserved, [int(s) for s in b.shards[-3:]])
        for _ in range(20):                                   # the same stream until the reserved shards are reached
            self.assertEqual(a.plan(4), b.plan(4))
        seen = set()
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            while True:
                seen.add(b._next_doc() >> 32)
        self.assertFalse(seen & set(b.reserved))
        self.assertEqual(seen, set(int(s) for s in b.shards[:-3]))
        st = a.state_dict()
        st["next_shard"] = len(a.shards)
        with self.assertRaisesRegex(ValueError, "reserved"):
            self.stream(reserve=3).load_state_dict(st)

    def test_stage_list_stream(self):
        base = self.stream(reserve=2)
        keys = np.array([base._next_doc() for _ in range(40)], dtype=np.int64)
        lens = np.array([base._length(int(k)) for k in keys])
        B, W = 3, 17
        n_mb = int(lens.sum()) // (B * W)
        total = n_mb * B * W                                     # cut the list to whole micro-batches
        cum = np.cumsum(lens)
        k = int(np.searchsorted(cum, total))
        keys, ends = keys[:k + 1], lens[:k + 1].copy()
        ends[-1] = total - (cum[k - 1] if k else 0)
        ls = StageListStream(self.pool, keys, ends, 16, 48)
        got = [ls.next_microbatch(B) for _ in range(n_mb)]
        want = np.concatenate([self.pool.doc(int(kk), 48)[:e] for kk, e in zip(keys, ends)])
        segs = [np.concatenate([self.pool.doc(int(kk), 48)[a:b] for kk, a, b in zip(g["rows"], g["starts"], g["ends"])])
                for g in got]
        self.assertTrue(np.array_equal(np.concatenate(segs), want))   # in list order, carry-over, nothing dropped
        for g, s in zip(got, segs):
            self.assertTrue(np.array_equal(g["x"], s.reshape(B, W)[:, :-1]))
        self.assertEqual(ls.plan(B), [])                        # used up: empty plans
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            ls.next_microbatch(B)
        for cut in (1, n_mb // 2, n_mb - 1):                    # resume mid-list
            a = StageListStream(self.pool, keys, ends, 16, 48)
            for _ in range(cut):
                a.next_microbatch(B)
            b = StageListStream(self.pool, keys, ends, 16, 48)
            b.load_state_dict(json.loads(json.dumps(a.state_dict())))
            for _ in range(n_mb - cut):
                self.assertTrue(np.array_equal(a.next_microbatch(B)["x"], b.next_microbatch(B)["x"]))

    def test_mates_stream_and_prefetcher(self):
        """Stage 1 = the pool stream batch for batch (through the Prefetcher too); the boundary's rewind drops the
        read-ahead; a stage list plays through the Prefetcher and resumes; a bare pool stream's state loads as
        stage 1."""
        lists = {}
        uni = self.stream(reserve=2)
        ms = MatesStream(self.stream(reserve=2), lambda s: lists[s])
        pf = Prefetcher(ms, 3, depth=4, threads=2)
        try:
            for _ in range(6):
                self.assertTrue(np.array_equal(uni.next_microbatch(3)["x"], pf.next_microbatch(3)["x"]))
            pf.load_state_dict(pf.state_dict())                 # rewind: the base is where the trainer is
            self.assertEqual(ms.base.state_dict(), uni.state_dict())
            keys, lens = enumerate_candidates(ms.base, 500)
            self.assertGreaterEqual(int(lens.sum()), 500)
            self.assertLess(int(lens[:-1].sum()), 500)
            g = gumbel_top_k(np.zeros(len(keys)), lens, 3 * 17 * 6, 1.0, np.random.default_rng(0))
            lists[2] = (keys[g["order"]], g["ends"])
            ms.start_stage(2, *lists[2])
            pf.load_state_dict(ms.state_dict())
            a = [pf.next_microbatch(3) for _ in range(2)]
            st = json.loads(json.dumps(pf.state_dict()))
            b = [pf.next_microbatch(3) for _ in range(2)]
            ms2 = MatesStream(self.stream(reserve=2), lambda s: lists[s])
            ms2.load_state_dict(st)
            for x in b:
                self.assertTrue(np.array_equal(ms2.next_microbatch(3)["x"], x["x"]))
            self.assertTrue(set(np.concatenate([x["rows"] for x in a + b])) <= set(keys.tolist()))
            ms3 = MatesStream(self.stream(reserve=2), lambda s: lists[s])
            ms3.load_state_dict(uni.state_dict())
            self.assertEqual(ms3.stage, 1)
        finally:
            pf.close()

    def test_candidates_never_twice(self):
        base = self.stream(reserve=2)
        seen = []
        for _ in range(4):
            keys, _ = enumerate_candidates(base, 300)
            seen.append(keys)
        allk = np.concatenate(seen)
        self.assertEqual(len(allk), len(set(allk.tolist())))
        self.assertFalse(set((allk >> 32).tolist()) & set(base.reserved))


# ---------------------------------------------------------------------------------------------- units: selection
class SelectTest(unittest.TestCase):
    def test_token_budget(self):
        rng = np.random.default_rng(0)
        n = 500
        lens = rng.integers(1, 300, size=n)
        scores = rng.standard_normal(n)
        for budget in (1, 299, int(lens.sum()) // 5, int(lens.sum())):
            g = gumbel_top_k(scores, lens, budget, 1.0, np.random.default_rng(1))
            self.assertEqual(int(g["ends"].sum()), budget)
            self.assertEqual(set(g["order"].tolist()), set(np.flatnonzero(g["selected"]).tolist()))
            full = g["ends"] == lens[g["order"]]
            self.assertGreaterEqual(int(full.sum()), len(full) - 1)          # at most one document cut
            self.assertTrue((g["ends"] > 0).all())
            keyv = scores + g["gumbel"]
            if (~g["selected"]).any():                                        # the selected are the top keys
                self.assertGreaterEqual(keyv[g["selected"]].min(), keyv[~g["selected"]].max())
        a = gumbel_top_k(scores, lens, 5000, 1.0, np.random.default_rng(7))
        b = gumbel_top_k(scores, lens, 5000, 1.0, np.random.default_rng(7))
        self.assertTrue(np.array_equal(a["order"], b["order"]) and np.array_equal(a["ends"], b["ends"]))
        self.assertFalse(np.array_equal(a["order"], np.sort(a["order"])))         # shuffled
        with self.assertRaises(ValueError):
            gumbel_top_k(scores, lens, int(lens.sum()) + 1, 1.0, rng)

    def test_plackett_luce_inclusion(self):
        """Unit lengths, budget k: P(i selected) = Plackett-Luce inclusion with weights exp(s / tau)."""
        s, tau, k, trials = np.array([0.0, 0.5, 1.0, 2.0]), 0.7, 2, 40000
        w = np.exp(s / tau)
        exact = np.zeros(len(s))
        for seq in itertools.permutations(range(len(s)), k):
            p, rest = 1.0, w.sum()
            for i in seq:
                p *= w[i] / rest
                rest -= w[i]
            exact[list(seq)] += p
        rng = np.random.default_rng(0)
        cnt = np.zeros(len(s))
        for _ in range(trials):
            cnt += gumbel_top_k(s, np.ones(len(s), int), k, tau, rng)["selected"]
        freq = cnt / trials
        se = np.sqrt(exact * (1 - exact) / trials)
        self.assertTrue(np.all(np.abs(freq - exact) < 5 * se), (freq, exact))


# ---------------------------------------------------------------------------------------------- units: probes
class ToyLM(nn.Module):
    """Bigram LM with the Llama call signature (x, targets, get_logits, doc_ids): logits[t] = W[x_t] + b."""

    def __init__(self, V=50304, d=8):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.out = nn.Linear(d, V)

    def forward(self, idx, targets=None, get_logits=False, doc_ids=None, gates=None):
        logits = self.out(torch.tanh(self.emb(idx)))
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1),
                                           ignore_index=-1)
        return {"logits": logits if get_logits else None, "loss": loss}


class ProbeTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.truth = make_pool(os.path.join(self.root, "pool"), np.random.default_rng(0), n_shards=8, docs=(40, 60))
        self.pool = DsPool(os.path.join(self.root, "pool"))
        make_benchmarks(os.path.join(self.root, "bench"), np.random.default_rng(1))
        self.dirs = {t: os.path.join(self.root, "bench", t) for t in ("taska", "taskb")}

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_reference_batches(self):
        """The reference = the target tasks' train-split rows (target_batches defaults), one topic each; L_r = the
        mean of the per-task mean token losses."""
        from daisy.labels.gates import validation_pass
        from daisy.labels.targets import target_batches

        ref, n, info = reference_batches(self.dirs, T, 260, rows_per_batch=3)
        self.assertEqual(n, 2)
        rows = {t: target_batches(d, T, 260)[0] for t, d in self.dirs.items()}
        x = torch.cat([b["x"] for t in self.dirs for b in rows[t]])
        self.assertTrue(torch.equal(torch.cat([b["x"] for b in ref]), x))
        self.assertTrue(all(len(b["x"]) <= 3 for b in ref))
        torch.manual_seed(0)
        lm = ToyLM().double()
        opt = torch.optim.SGD(lm.parameters(), lr=0.1)
        pr = Prober(lm, opt, {"out": lm.out}, ref, n, T, 0.0, "fp32", "cpu")
        per = [float(validation_pass(lm, {"out": lm.out}, rows[t], 1, contextlib.nullcontext(), compute_grad=False,
                                     out_dtype=torch.float64)[1][0]) for t in self.dirs]
        self.assertAlmostEqual(pr.reference_loss(), float(np.mean(per)), places=10)

    def test_probe_documents(self):
        shards = [6, 7]
        th = np.array(sorted(self.truth[k][2] for k in self.truth if (k >> 32) == 6)[:5], dtype=np.uint64)
        counts = {2: 20, 3: 10, 4: 10}
        st = {s: probe_documents(self.pool, shards, th, np.zeros(0, np.uint64), 5, s, 0, counts) for s in (2, 3, 4)}
        for s, k in st.items():
            self.assertEqual(len(k), counts[s])
            self.assertTrue(set((k >> 32).tolist()) <= set(shards))
            self.assertFalse({self.truth[int(x)][2] for x in k} & set(th.tolist()))       # held-out excluded
            self.assertTrue(all(len(self.truth[int(x)][0]) >= 2 for x in k))
        allk = np.concatenate(list(st.values()))
        self.assertEqual(len(set(allk.tolist())), len(allk))                              # fresh per stage
        again = probe_documents(self.pool, shards, th, np.zeros(0, np.uint64), 5, 3, 0, counts)
        self.assertTrue(np.array_equal(again, st[3]))                                     # seeded
        other = probe_documents(self.pool, shards, th, np.zeros(0, np.uint64), 5, 3, 1, counts)
        self.assertFalse(np.array_equal(other, st[3]))
        self.assertEqual([per_rank(10, 4, r) for r in range(4)], [3, 3, 2, 2])

    def test_oracle_equals_sgd_loss_change(self):
        """SGD (no momentum, no decay) on a float64 toy LM: the oracle equals L_r(theta) - L_r(theta - lr clip(g))
        computed independently (to 2e-7: validation_pass takes the token losses in fp32), and its first-order form
        lr <grad L_r, clip(g)> to 2e-3 (the curvature term); the gradient norm and the clip as in training."""
        ref, n, _ = reference_batches(self.dirs, T, 260, rows_per_batch=2)
        docs = [self.pool.doc(k, 48) for k in sorted(self.truth)[:6]]

        def ref_loss(m):
            """mean over target tasks of the mean token loss over their rows (a differentiable scalar)"""
            per = []
            for t in range(n):
                tot, cnt = 0.0, 0
                for b in ref:
                    lg = m(b["x"], targets=b["y"], get_logits=True)["logits"]
                    keep = (b["y"] >= 0) & (b["topic"] == t)
                    tot = tot + nn.functional.cross_entropy(lg[keep], b["y"][keep], reduction="sum")
                    cnt += int(keep.sum())
                per.append(tot / cnt)
            return sum(per) / n

        for clip, lr in ((0.0, 0.5), (1e-3, 0.5)):
            torch.manual_seed(0)
            lm = ToyLM().double()
            opt = torch.optim.SGD(lm.parameters(), lr=lr)
            pr = Prober(lm, opt, {"out": lm.out}, ref, n, T, clip, "fp32", "cpu")
            before, res = pr.run(docs)
            exact_before = float(ref_loss(lm))
            self.assertLess(abs(before - exact_before), 2e-7)
            ref_grad = torch.autograd.grad(ref_loss(lm), list(lm.parameters()))
            for d, after, gn, clipped in zip(docs, res["loss_after"], res["grad_norm"], res["clipped"]):
                x, y, seg = (torch.from_numpy(a) for a in doc_row(d, T))
                g = torch.autograd.grad(lm(x, targets=y, doc_ids=seg)["loss"], list(lm.parameters()))
                norm = float(torch.sqrt(sum((gi ** 2).sum() for gi in g)))
                self.assertAlmostEqual(gn, norm, places=10)
                self.assertEqual(bool(clipped), clip > 0 and norm > clip)
                scale = min(1.0, clip / (norm + 1e-6)) if clip > 0 else 1.0
                twin = ToyLM().double()
                twin.load_state_dict(lm.state_dict())
                with torch.no_grad():
                    for p, gi in zip(twin.parameters(), g):
                        p -= lr * scale * gi
                    o, o_exact = before - after, exact_before - float(ref_loss(twin))
                self.assertGreater(abs(o_exact), 1e-6 if clip else 1e-4)
                self.assertLess(abs(o - o_exact), 2e-7)
                lin = lr * scale * float(sum((a * b).sum() for a, b in zip(ref_grad, g)))
                self.assertLess(abs(o_exact / lin - 1.0), 2e-3)
            if clip > 0:
                self.assertTrue(res["clipped"].all())

    def test_probe_restores_model_and_optimizer(self):
        """Tiny Llama + AdamW after a few steps: probing (document-masked, clipped) leaves every parameter and every
        optimizer state tensor bit-identical, grads cleared; the oracle has spread; the clip flag follows the norm."""
        from daisy.models.llama import Llama

        cfg = config.parse_args([
            "--experiment_name", "test", "--daisy_index_dir", "unused", "--n_embd", "32", "--n_head", "2",
            "--n_layer", "2", "--sequence_length", str(T), "--vocab_size", "50304", "--dropout", "0.0"])
        st = contextlib.ExitStack()
        st.enter_context(use_fakes())
        self.addCleanup(st.close)
        torch.manual_seed(0)
        lm = Llama(cfg)
        opt = torch.optim.AdamW(lm.parameters(), lr=3e-3, betas=(0.8, 0.999), weight_decay=0.1)
        rng = np.random.default_rng(0)
        for _ in range(3):
            x = torch.from_numpy(rng.integers(0, 1000, (2, T)))
            lm(x, targets=x, doc_ids=torch.zeros_like(x))["loss"].backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        from daisy.labels.gates import attach_gate_names

        layers = attach_gate_names(lm)
        ref, n, _ = reference_batches(self.dirs, T, 260, rows_per_batch=2)
        before_p = {k: v.clone() for k, v in lm.state_dict().items()}
        before_o = {id(p): {k: v.clone() for k, v in opt.state[p].items()} for p in lm.parameters()}
        pr = Prober(lm, opt, layers, ref, n, T, 0.5, "bf16", "cpu")
        docs = [self.pool.doc(k, T) for k in sorted(self.truth)[:8]]
        L0, res = pr.run(docs)
        for k, v in lm.state_dict().items():
            self.assertTrue(torch.equal(v, before_p[k]), k)
        for p in lm.parameters():
            self.assertIsNone(p.grad)
            for k, v in opt.state[p].items():
                self.assertTrue(torch.equal(v, before_o[id(p)][k]), k)
        self.assertTrue(np.all(np.isfinite(res["loss_after"])))
        self.assertGreater(np.std(L0 - res["loss_after"]), 0)
        self.assertTrue(np.array_equal(res["clipped"], res["grad_norm"] > 0.5))
        self.assertEqual(pr.reference_loss(), L0)                       # the state is the snapshot again


# ---------------------------------------------------------------------------------------------- units: BERT, text
@unittest.skipUnless(HAVE_TF, "needs transformers")
class InfluenceModelTest(unittest.TestCase):
    def model(self):
        torch.manual_seed(0)
        m = influence.InfluenceModel(influence.build_bert(config=tiny_bert_config()))
        m.eval()
        return m

    def docs(self, lens, seed=0, hi=29000):
        rng = np.random.default_rng(seed)
        return [np.concatenate([[101], rng.integers(1000, hi, size=L - 2), [102]]).astype(np.uint16) for L in lens]

    def test_chunking_and_pooling(self):
        """forward on MATES's padded layout == a direct computation: BERT on the first 512 ids and, separately, on the
        rest (positions restarting, no [CLS]), the mean of all non-pad last hidden states, the linear head."""
        m = self.model()
        docs = self.docs([5, 300, 512, 513, 700, 1024])
        ids, mask = influence.padded_layout(docs)
        with torch.no_grad():
            got = m(torch.from_numpy(ids), torch.from_numpy(mask)).numpy()
            want = []
            for d in docs:
                d = torch.from_numpy(d.astype(np.int64))
                hs = [m.bert(input_ids=c[None], attention_mask=torch.ones_like(c)[None]).last_hidden_state[0]
                      for c in (d[:512], d[512:]) if len(c)]
                want.append(float(m.head(torch.cat(hs).mean(0))))
        self.assertTrue(np.allclose(got, want, atol=1e-5), (got, want))

    def test_bucketed_scoring_equals_padded(self):
        m = self.model()
        rng = np.random.default_rng(1)
        docs = self.docs(rng.integers(2, 1025, size=60), seed=2) + self.docs([512, 513, 1024, 2])
        ids, mask = influence.padded_layout(docs)
        with torch.no_grad():
            direct = m(torch.from_numpy(ids), torch.from_numpy(mask)).double().numpy()
        for kw in ({"padded": True, "tokens": 4096}, {"bucket": 64, "tokens": 2048}, {"bucket": 128, "tokens": 700},
                   {"bucket": 512, "tokens": 512}):
            sc = influence.Scorer(m, "cpu", contextlib.nullcontext(), **kw)
            got = influence.score_arrays(sc, docs, block=7)
            self.assertTrue(np.allclose(got, direct, atol=1e-5), kw)
            if not kw.get("padded"):
                self.assertLess(sc.stats["padded_tokens"], len(docs) * 1024)

    def test_finetune_learns_and_keeps_best(self):
        """Labels = the share of 'marked' WordPiece ids: validation Spearman rises, the records come every eval_every
        steps and at the last, and the model ends with the best record's weights."""
        rng = np.random.default_rng(0)
        docs = self.docs(rng.integers(20, 200, size=300), seed=3, hi=1014)       # 14 ids, each seen often
        marked = np.arange(1000, 1014) % 7 == 0
        y = np.array([marked[d[1:-1] - 1000].mean() for d in docs]) + 0.01 * rng.standard_normal(len(docs))
        torch.manual_seed(0)
        m = influence.InfluenceModel(influence.build_bert(config=tiny_bert_config()))
        summary, recs = influence.finetune(m, docs, y, world=1, rank=0, device="cpu", ctx=contextlib.nullcontext(),
                                           scorer_kw={"bucket": 64, "tokens": 4096}, epochs=8, batch=16, micro=8,
                                           lr=5e-3, eval_every=5, log=lambda *a: None)
        steps = [r["step"] for r in recs]
        self.assertEqual(steps, sorted(set(list(range(5, summary["steps"] + 1, 5)) + [summary["steps"]])))
        self.assertEqual(summary["n_val"], 30)
        best = max(recs, key=lambda r: r["val_spearman"])
        self.assertEqual(summary["best"]["step"], best["step"])
        self.assertGreater(best["val_spearman"], 0.6)
        self.assertLess(recs[0]["val_spearman"], 0.3)
        tr, va = influence.split_train_val(len(y), 0.1)
        z = (y - y[tr].mean()) / y[tr].std()
        sc = influence.Scorer(m, "cpu", contextlib.nullcontext(), bucket=64, tokens=4096)
        self.assertAlmostEqual(influence.spearman(influence.score_arrays(sc, [docs[i] for i in va]), z[va]),
                               best["val_spearman"], places=6)

    def test_spearman_matches_average_ranks(self):
        a = np.array([1.0, 2.0, 2.0, 3.0, 5.0, 5.0, 5.0])
        self.assertTrue(np.array_equal(influence.rankdata(a), [1, 2.5, 2.5, 4, 6, 6, 6]))
        self.assertAlmostEqual(influence.spearman(a, -a), -1.0)


@unittest.skipUnless(REAL_TEXT and HAVE_TF, "needs tokenizers, transformers and the local HF cache (gpt2, bert-tiny)")
class TextTest(unittest.TestCase):
    def test_decode_and_wordpiece_match_transformers(self):
        from transformers import BertTokenizerFast, PreTrainedTokenizerFast
        from tokenizers import Tokenizer

        g = Tokenizer.from_file(GPT2_JSON[-1])
        texts = ["Hello world. MATES selects data!", "Ünïcode café — naïve façade 東京 " * 3,
                 "a long document " * 600, ""]
        docs = [np.array(g.encode(t).ids + [EOS], dtype=np.uint16) for t in texts]
        d = tempfile.mkdtemp()
        try:
            shutil.copy(BERT_VOCAB[-1], os.path.join(d, "vocab.txt"))
            enc = text.TextEncoder(GPT2_JSON[-1], d)
            got = enc.encode(docs)
            hf_g = PreTrainedTokenizerFast(tokenizer_file=GPT2_JSON[-1])
            hf_b = BertTokenizerFast(vocab_file=BERT_VOCAB[-1], do_lower_case=True)
            dec = hf_g.batch_decode([x.tolist() for x in docs], skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
            self.assertEqual(dec[:2], texts[:2])
            want = hf_b(dec, max_length=1024, truncation=True)["input_ids"]
            for a, b in zip(got, want):
                self.assertEqual(a.tolist(), b)
                self.assertEqual((a[0], a[-1]), (101, 102))
            self.assertEqual(len(got[2]), 1024)
        finally:
            shutil.rmtree(d)

    def test_worker_pool_equals_in_process(self):
        root = tempfile.mkdtemp()
        try:
            make_pool(os.path.join(root, "pool"), np.random.default_rng(0), n_shards=4, docs=(30, 40))
            os.makedirs(os.path.join(root, "bert"))
            shutil.copy(BERT_VOCAB[-1], os.path.join(root, "bert", "vocab.txt"))
            keys = np.array(sorted({(s << 32) | i for s in range(4) for i in range(25)}), dtype=np.int64)[::-1]
            args = (os.path.join(root, "pool"), 48, GPT2_JSON[-1], os.path.join(root, "bert"))
            a = text.EncodePool(*args, workers=0).encode_keys(keys, chunk=16)
            p = text.EncodePool(*args, workers=2)
            try:
                b = p.encode_keys(keys, chunk=16)
            finally:
                p.close()
            self.assertEqual([x.tolist() for x in a], [x.tolist() for x in b])
            self.assertGreater(p.cpu_s, 0)
        finally:
            shutil.rmtree(root)


# ---------------------------------------------------------------------------------------------- end to end
class MatesRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.paths = make_world(cls.root)
        cls.fakes = use_fakes()
        cls.fakes.__enter__()
        cls.full_rec = []
        cls.full_exp = os.path.join(cls.root, "full")
        cls.full = run_mates(mates_cfg(cls.paths, 26), cls.full_exp, record=cls.full_rec)

    @classmethod
    def tearDownClass(cls):
        cls.fakes.__exit__(None, None, None)
        shutil.rmtree(cls.root)

    def mdir(self, exp=None):
        return os.path.join(exp or self.full_exp, "daisy", "mates")

    def test_stage1_is_the_uniform_run(self):
        """Up to the first boundary the MATES run trains exactly the uniform run's batches: same weights and
        held-out losses at step 8, the same stream position; a MATES run that ends at 8 equals the uniform run."""
        exp = os.path.join(self.root, "uniform")
        rec = []
        uni = run_mates(mates_cfg(self.paths, 8, mates=False), exp, record=rec)
        self.assertEqual([(r[0], r[2].tolist(), r[3].tolist(), r[4].tolist()) for r in rec],
                         [(r[0], r[2].tolist(), r[3].tolist(), r[4].tolist()) for r in self.full_rec if r[0] < 8])
        a = torch.load(os.path.join(exp, "ckpts", "8", "main.pt"), weights_only=False)
        b = torch.load(os.path.join(self.full_exp, "ckpts", "8", "main.pt"), weights_only=False)
        for k, v in a["model"].items():
            self.assertTrue(torch.equal(v, b["model"][k]), k)
        sa = json.load(open(os.path.join(exp, "ckpts", "8", "stream_r0.json")))
        sb = json.load(open(os.path.join(self.full_exp, "ckpts", "8", "stream_r0.json")))
        self.assertEqual(sb["mates_stage"], 1)
        self.assertEqual(sa, sb["base"])
        bv = lambda e: [r for r in jsonl(os.path.join(e, "daisy", "metrics.jsonl")) if r["step"] <= 8]
        self.assertEqual([{k: v for k, v in r.items() if k != "seconds"} for r in bv(exp)],
                         [{k: v for k, v in r.items() if k != "seconds"} for r in bv(self.full_exp)])
        m8 = run_mates(mates_cfg(self.paths, 8), os.path.join(self.root, "mates8"))
        same_weights(self, uni, m8, "MATES to 8 vs uniform")

    def test_boundaries_and_invariants(self):
        md = self.mdir()
        stages = jsonl(os.path.join(md, "stages.jsonl"))
        costs = jsonl(os.path.join(md, "stage_costs.jsonl"))
        self.assertEqual([r["stage"] for r in stages], [2, 3, 4])
        self.assertEqual([r["step"] for r in costs], [8, 16, 24])
        budget = 8 * 4 * (T + 1)
        cand_all, probe_all = [], []
        for s in (2, 3, 4):
            d = os.path.join(md, f"stage_{s}")
            for f in ("probes_r0.npz", "probe_meta.json", "bert/model.pt", "bert/config.json", "bert_log.jsonl",
                      "bert_done.json", "select_r0.npz", "select_r0.json"):
                self.assertTrue(os.path.exists(os.path.join(d, f)), (s, f))
            pz, sz = np.load(os.path.join(d, "probes_r0.npz")), np.load(os.path.join(d, "select_r0.npz"))
            self.assertEqual(len(pz["key"]), 24 if s == 2 else 30)
            sel_keys = sz["cand_key"][sz["selected"]]
            self.assertEqual(set(sel_keys.tolist()), set(sz["sel_key"].tolist()))
            self.assertEqual(int(sz["sel_end"].sum()), budget)                       # exactly the stage budget
            self.assertGreaterEqual(int(sz["cand_len"].sum()), 5 * budget)            # 1 / ratio x the budget
            self.assertLess(int(sz["cand_len"][:-1].sum()), 5 * budget)
            ends = dict(zip(sz["sel_key"].tolist(), sz["sel_end"].tolist()))
            # trained in stage s: exactly the selected list, in order, every selected token once
            trained = [r for r in self.full_rec if r[1] == s]
            self.assertEqual(sorted({r[0] for r in trained}), list(range(8 * (s - 1), min(8 * s, 26))))
            got = {}
            for _, _, rows, starts, seg_ends in trained:
                for k, a, b in zip(rows.tolist(), starts.tolist(), seg_ends.tolist()):
                    self.assertIn(k, ends)
                    self.assertLessEqual(b, ends[k])
                    got[k] = got.get(k, 0) + (b - a)
            if 8 * s <= 26:
                self.assertEqual(got, ends)
                self.assertEqual(sum(got.values()), budget)
            cand_all.append(sz["cand_key"])
            probe_all.append(pz["key"])
            meta = json.load(open(os.path.join(d, "probe_meta.json")))
            self.assertEqual(meta["n"], len(pz["key"]))
            self.assertAlmostEqual(stages[s - 2]["oracle_mean"], float(np.mean(meta["loss_before"] - pz["loss_after"])))
            log = jsonl(os.path.join(d, "bert_log.jsonl"))
            self.assertEqual(log[-1]["step"], json.load(open(os.path.join(d, "bert_done.json")))["steps"])
        cand_all, probe_all = np.concatenate(cand_all), np.concatenate(probe_all)
        self.assertEqual(len(cand_all), len(set(cand_all.tolist())))                  # no candidate twice
        self.assertEqual(len(probe_all), len(set(probe_all.tolist())))                # fresh probes
        self.assertFalse(set(cand_all.tolist()) & set(probe_all.tolist()))
        base = ShardDocStream(DsPool(self.paths["pool"]), T, 2048, seed=1337, n_open=4, reserve=2)
        self.assertTrue(set((probe_all >> 32).tolist()) <= set(base.reserved))
        self.assertFalse(set((cand_all >> 32).tolist()) & set(base.reserved))
        stage1 = {k for r in self.full_rec if r[1] == 1 for k in r[2].tolist()}
        self.assertFalse(stage1 & set(cand_all.tolist()))
        for r in stages:
            self.assertEqual(r["selected_tokens"], budget)
            self.assertAlmostEqual(r["sel_token_share"], budget / r["candidate_tokens"])
        b = json.load(open(os.path.join(md, "budget.json")))
        self.assertEqual(b["step"], 24)
        self.assertAlmostEqual(b["parts_rank0"]["boundaries"], sum(c["total_gpu_s"] for c in costs) / 3600)
        fin = json.load(open(os.path.join(self.full_exp, "daisy", "finished.json")))
        self.assertEqual(fin["iter"], 26)
        self.assertNotIn("reason", fin)

    def test_first_boundary(self):
        """--mates_first_boundary 4: boundaries at 4, 12, 20 (stages 2, 3, 4 of U = 8 steps); stage 1 is the uniform
        run's first 4 steps, and every later stage trains exactly its selected list, one stage budget each."""
        exp = os.path.join(self.root, "first4")
        rec = []
        run_mates(mates_cfg(self.paths, 22, ["--mates_first_boundary", "4"]), exp, record=rec)
        md = self.mdir(exp)
        self.assertEqual([r["step"] for r in jsonl(os.path.join(md, "stage_costs.jsonl"))], [4, 12, 20])
        self.assertEqual([r["stage"] for r in jsonl(os.path.join(md, "stages.jsonl"))], [2, 3, 4])
        self.assertEqual([(r[0], r[1], r[2].tolist()) for r in rec if r[0] < 4],
                         [(r[0], r[1], r[2].tolist()) for r in self.full_rec if r[0] < 4])
        budget = 8 * 4 * (T + 1)
        for s in (2, 3, 4):
            trained = [r for r in rec if r[1] == s]
            self.assertEqual(sorted({r[0] for r in trained}), list(range(4 + 8 * (s - 2), min(4 + 8 * (s - 1), 22))))
            sz = np.load(os.path.join(md, f"stage_{s}", "select_r0.npz"))
            self.assertEqual(int(sz["sel_end"].sum()), budget)
            if s < 4:
                self.assertEqual(sum(b - a for r in trained for a, b in zip(r[3].tolist(), r[4].tolist())), budget)

    def test_resume_across_boundary_and_mid_stage(self):
        """Chained jobs stopping before the boundary at 8 (at 4) and inside stage 2 (at 12) end exactly like the
        uninterrupted run, with the same selections."""
        for stop in (4, 12):
            with self.subTest(stop=stop):
                exp = os.path.join(self.root, f"chain{stop}")
                with mock.patch.dict(os.environ, {"DAISY_STOP_AT": "1", "DAISY_STOP_MIN_ITER": str(stop)}):
                    run_mates(mates_cfg(self.paths, 26), exp)
                self.assertEqual(int(torch.load(os.path.join(exp, "ckpts", "latest", "main.pt"),
                                                weights_only=False)["itr"]), stop)
                m = run_mates(mates_cfg(self.paths, 26, ["--resume_from", os.path.join(exp, "ckpts", "latest")]), exp)
                same_weights(self, self.full, m, f"chain {stop}")
                for s in (2, 3, 4):
                    a = np.load(os.path.join(self.mdir(exp), f"stage_{s}", "select_r0.npz"))
                    b = np.load(os.path.join(self.mdir(), f"stage_{s}", "select_r0.npz"))
                    self.assertTrue(np.array_equal(a["sel_key"], b["sel_key"]))
                self.assertEqual(len(jsonl(os.path.join(self.mdir(exp), "stages.jsonl"))), 3)

    def test_boundary_artifacts_are_reused(self):
        """A job that resumes before a boundary whose artifacts exist (a job cut inside it) reuses them: no probe, no
        fine-tuning, no candidate scoring; the run ends exactly like the uninterrupted one."""
        exp = os.path.join(self.root, "reuse")
        os.makedirs(os.path.join(exp, "ckpts"))
        shutil.copytree(os.path.join(self.full_exp, "ckpts", "16"), os.path.join(exp, "ckpts", "16"))
        shutil.copytree(self.mdir(), self.mdir(exp))
        for f in ("stages.jsonl", "stage_costs.jsonl"):
            os.remove(os.path.join(self.mdir(exp), f))
        with mock.patch.object(Prober, "run", side_effect=AssertionError("probed again")), \
                mock.patch.object(influence, "finetune", side_effect=AssertionError("fine-tuned again")), \
                mock.patch.object(influence.Scorer, "score", side_effect=AssertionError("scored again")):
            m = run_mates(mates_cfg(self.paths, 26, ["--resume_from", os.path.join(exp, "ckpts", "16")]), exp)
        same_weights(self, self.full, m, "reused artifacts")
        got = jsonl(os.path.join(self.mdir(exp), "stage_costs.jsonl"))
        want = {r["stage"]: r for r in jsonl(os.path.join(self.mdir(), "stage_costs.jsonl"))}
        self.assertEqual([r["stage"] for r in got], [3, 4])
        for r in got:
            self.assertEqual(r["reused"], ["probe", "finetune", "select"])
            for k in ("probe", "finetune", "enumerate", "score", "select"):
                self.assertEqual(r["wall_s"][k], want[r["stage"]]["wall_s"][k])

    def test_init_from_the_uniform_run(self):
        """--mates_init_from the uniform run's checkpoint at 8 = the MATES run from scratch; the cost offset is that
        run's steady s/step x 8 steps."""
        uni = os.path.join(self.root, "uniform_init")
        run_mates(mates_cfg(self.paths, 12, mates=False), uni)
        exp = os.path.join(self.root, "init")
        m = run_mates(mates_cfg(self.paths, 26, ["--mates_init_from", os.path.join(uni, "ckpts", "8")]), exp)
        same_weights(self, self.full, m, "init-from")
        b = json.load(open(os.path.join(self.mdir(exp), "budget.json")))
        st = np.load(os.path.join(uni, "daisy", "step_times_r0.npy"))[0]
        warm = min(50, len(st) // 4)
        sps = float(st[warm:][st[warm:] <= 2.0].mean())
        self.assertAlmostEqual(b["parts_rank0"]["offset"], 8 * sps / 3600)
        self.assertEqual(b["parts_rank0"]["steps_this_run"], 24 - 8)
        self.assertFalse(os.path.exists(os.path.join(exp, "ckpts", "4")))
        with self.assertRaisesRegex(ValueError, "differs"):
            run_mates(mates_cfg(self.paths, 26, ["--mates_init_from", os.path.join(uni, "ckpts", "8"),
                                                  "--lr", "1e-3"]), os.path.join(self.root, "init_bad"))

    def test_budget_stop(self):
        """Step times fixed at 1 s and a budget of 6 steps: the run stops at the first refresh step past it (8), before
        the boundary there, like a finished run: final checkpoints, the evaluation at 8, finished.json with the reason;
        a resubmitted job does nothing."""
        from daisy.mates.stage import MatesController

        exp = os.path.join(self.root, "budget")
        cfg = mates_cfg(self.paths, 26, ["--mates_budget", str(6 / 3600), "--mates_cost_offset", "0"])
        real = MatesController.on_step
        with mock.patch.object(MatesController, "on_step", lambda self, dt: real(self, 1.0)):
            run_mates(cfg, exp)
        fin = json.load(open(os.path.join(exp, "daisy", "finished.json")))
        self.assertEqual((fin["iter"], fin["reason"]), (8, "budget"))
        self.assertAlmostEqual(fin["gpu_hours"], 8 / 3600)
        self.assertFalse(os.path.exists(os.path.join(self.mdir(exp), "stage_2")))
        for d in ("8", "latest"):
            self.assertEqual(int(torch.load(os.path.join(exp, "ckpts", d, "main.pt"), weights_only=False)["itr"]), 8)
        bv = [r["step"] for r in jsonl(os.path.join(exp, "daisy", "metrics.jsonl"))]
        self.assertEqual(bv, [0, 4, 8])
        before = sorted(os.listdir(os.path.join(exp, "daisy")))
        cfg2 = mates_cfg(self.paths, 26, ["--mates_budget", str(6 / 3600), "--resume_from",
                                          os.path.join(exp, "ckpts", "latest")])
        from daisy.train.trainer import train_daisy
        from daisy.distributed.single import SingleNodeBackend
        from daisy.models.llama import Llama

        model = Llama(cfg2)
        stats = train_daisy(model, torch.optim.AdamW(model.parameters()), None, exp, SingleNodeBackend(cfg2), cfg2)
        self.assertEqual(stats["already_done"], 8)
        self.assertEqual(sorted(os.listdir(os.path.join(exp, "daisy"))), before)

    def test_boundary_estimate(self):
        """The next boundary's cost: the previous one's GPU-seconds, probing and fine-tuning scaled by the probe counts."""
        from daisy.mates.stage import MatesController

        stub = types.SimpleNamespace(cfg=types.SimpleNamespace(mates_probes_first=80000, mates_probes_later=20000),
                                     boundary_parts={2: {"probe": 800.0, "finetune": 40.0, "gather": 1.0,
                                                         "enumerate": 9.0, "score": 300.0, "select": 2.0},
                                                     3: {"probe": 200.0, "finetune": 10.0, "score": 290.0}})
        self.assertIsNone(MatesController.boundary_estimate(stub, 2))
        self.assertAlmostEqual(MatesController.boundary_estimate(stub, 3), (200 + 10 + 1 + 9 + 300 + 2) / 3600)
        self.assertAlmostEqual(MatesController.boundary_estimate(stub, 4), (200 + 10 + 290) / 3600)

    def test_budget_stop_before_unaffordable_boundary(self):
        """Step times 1 s, budget 30 GPU-s, the second boundary estimated at 100 GPU-s: the first boundary (step 8) runs,
        the run stops at 16 instead of running the second one; finished.json says why."""
        from daisy.mates.stage import MatesController

        exp = os.path.join(self.root, "budget_before")
        cfg = mates_cfg(self.paths, 26, ["--mates_budget", str(30 / 3600), "--mates_cost_offset", "0"])
        real = MatesController.on_step
        with mock.patch.object(MatesController, "on_step", lambda self, dt: real(self, 1.0)), \
                mock.patch.object(MatesController, "boundary_estimate", lambda self, s: None if s == 2 else 100 / 3600):
            run_mates(cfg, exp)
        fin = json.load(open(os.path.join(exp, "daisy", "finished.json")))
        self.assertEqual((fin["iter"], fin["reason"]), (16, "budget"))
        self.assertEqual(fin["stopped_before_boundary"]["stage"], 3)
        self.assertLess(fin["gpu_hours"], 30 / 3600)
        self.assertTrue(os.path.exists(os.path.join(self.mdir(exp), "stage_2")))
        self.assertFalse(os.path.exists(os.path.join(self.mdir(exp), "stage_3")))
        self.assertEqual(int(torch.load(os.path.join(exp, "ckpts", "latest", "main.pt"), weights_only=False)["itr"]), 16)


# ---------------------------------------------------------------------------------------------- two ranks
class _GlooBackend:
    def __init__(self, rank, world):
        self.rank, self.world = rank, world

    def transform_model(self, model):
        from torch.nn.parallel import DistributedDataParallel as DDP

        return DDP(model)

    def get_raw_model(self, model):
        return model.module

    def get_world_size(self):
        return self.world

    def is_master_process(self):
        return self.rank == 0

    @contextlib.contextmanager
    def get_context_for_microstep_forward(self, model, microstep_idx, gradient_accumulation_steps):
        model.require_backward_grad_sync = microstep_idx == gradient_accumulation_steps - 1
        yield


def _two_rank_child(rank, world, port, paths, exp, iters, extra):
    """One rank. A failure's traceback goes to a file: torch.multiprocessing.spawn sends it through a pipe that a long
    traceback fills (macOS: 16 KB) while the parent only reads after the child exits, and both would wait forever."""
    import traceback

    torch.set_num_threads(1)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world)
    try:
        with use_fakes():
            cfg = mates_cfg(paths, iters, extra)
            cfg.seed = cfg.seed + rank                     # as the DDP backend does
            run_mates(cfg, exp, backend=_GlooBackend(rank, world))
    except BaseException:
        path = os.path.join(os.path.dirname(exp), f"traceback_{os.path.basename(exp)}_r{rank}.txt")
        with open(path, "w") as f:
            f.write(traceback.format_exc())
        raise RuntimeError(f"rank {rank} failed: {path}") from None
    finally:
        torch.distributed.destroy_process_group()


class TwoRankTest(unittest.TestCase):
    def test_two_ranks(self):
        """2 gloo ranks: probes split and gathered, DDP fine-tuning, per-rank candidates and selections (disjoint),
        records from rank 0; the budget decision is the same on both ranks (it stops both at the same step)."""
        root = tempfile.mkdtemp()
        try:
            paths = make_world(root, n_shards=16)
            exp = os.path.join(root, "ddp")
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            torch.multiprocessing.spawn(_two_rank_child, args=(2, port, paths, exp, 14, []), nprocs=2, join=True)
            md = os.path.join(exp, "daisy", "mates", "stage_2")
            pz = [np.load(os.path.join(md, f"probes_r{r}.npz")) for r in range(2)]
            self.assertEqual([len(p["key"]) for p in pz], [12, 12])
            self.assertFalse(set(pz[0]["key"].tolist()) & set(pz[1]["key"].tolist()))
            sz = [np.load(os.path.join(md, f"select_r{r}.npz")) for r in range(2)]
            self.assertFalse(set(sz[0]["cand_key"].tolist()) & set(sz[1]["cand_key"].tolist()))
            for z in sz:
                self.assertEqual(int(z["sel_end"].sum()), 8 * 4 * (T + 1))      # per rank: U x 4 rows x (T + 1)
            st = jsonl(os.path.join(exp, "daisy", "mates", "stages.jsonl"))
            self.assertEqual(len(st), 1)
            self.assertEqual(st[0]["n_probes"], 24)
            self.assertEqual(st[0]["n_candidates"], sum(len(z["cand_key"]) for z in sz))
            self.assertEqual(json.load(open(os.path.join(exp, "daisy", "finished.json")))["iter"], 14)
            # budget: rank 0's steps timed at 1 s, rank 1's at 0 s; 10 GPU-s: rank 0 alone reaches it at 8 (16 GPU-s;
            # 8 at step 4), and the max over ranks stops both there
            exp2 = os.path.join(root, "ddp_budget")
            torch.multiprocessing.spawn(_two_rank_budget_child, args=(2, port + 1, paths, exp2), nprocs=2, join=True)
            fin = json.load(open(os.path.join(exp2, "daisy", "finished.json")))
            self.assertEqual((fin["iter"], fin["reason"]), (8, "budget"))
            for r in range(2):
                st = json.load(open(os.path.join(exp2, "ckpts", "latest", f"stream_r{r}.json")))
                self.assertEqual(st["mates_stage"], 1)
        finally:
            shutil.rmtree(root)


def _two_rank_budget_child(rank, world, port, paths, exp):
    from daisy.mates.stage import MatesController

    real = MatesController.on_step
    with mock.patch.object(MatesController, "on_step", lambda self, dt: real(self, 1.0 if self.rank == 0 else 0.0)):
        _two_rank_child(rank, world, port, paths, exp, 26,
                        ["--mates_budget", str(10 / 3600), "--mates_cost_offset", "0"])


if __name__ == "__main__":
    unittest.main()
