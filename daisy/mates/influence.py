"""The data influence model (MATES: modeling_data_influence_model.py), its fine-tuning on the oracle
(train_data_influence_model.py) and the scoring of candidates (predict_data_influence.py).

InfluenceModel: BERT (transformers BertModel, without the unused pooler) over two 512-token chunks of the document's
first 1024 WordPiece tokens ([CLS] ... [SEP]; the second chunk continues the text without [CLS], its positions restart
at 0), the masked mean of the last hidden states over the non-pad tokens of BOTH chunks, dropout (0.1), Linear(768, 1).
The mean and the head run in fp32 (MATES: in bf16).

Scoring (Scorer): the chunks are independent sequences for BERT and the head is linear in the mean, so a document's
score is (w . S_1 + w . S_2) / (n_1 + n_2) + b, with S_c the summed last hidden states of chunk c over its n_c
tokens. Two savings follow that change nothing but floating-point rounding: an all-pad second chunk (documents of at
most 512 WordPiece tokens) is skipped, and each chunk is padded only to its length bucket (multiples of `bucket`, about
`tokens` tokens per batch) instead of 512. padded=True runs MATES's layout (every document as two 512-token chunks;
--mates_score_padded). Batch shapes are fixed per bucket (short batches are filled with one-token dummy rows), so the
compiled model compiles once per bucket; the next batch is assembled on the CPU while the GPU runs the previous one.

Fine-tuning (finetune): MATES's HF Trainer recipe written out: split seeded like datasets.train_test_split(test_size,
seed=1234); labels z-scored with the training split's mean / std (population std, as np.std); MSE; AdamW (betas 0.9 /
0.999, eps 1e-8; weight decay on all parameters but biases and LayerNorm weights), linear decay to 0 without warmup,
gradient norm clipped at 1.0 (the Trainer's default), bf16 autocast; the global batch split over the ranks (DDP on the
run's process group; per-rank micro-batches of fixed shape, accumulated); validation every `eval_every` steps and at
the last step, the model with the best validation Spearman kept (MATES: load_best_model_at_end, metric spearman).
"""

import math
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

CHUNK = 512
MAX_LEN = 2 * CHUNK


# ---------------------------------------------------------------------------------------------- metrics
def rankdata(a):
    """Average ranks (1-based) with ties averaged, as scipy.stats.rankdata(method="average")."""
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), dtype=np.int64)
    inv[order] = np.arange(len(a))
    s = a[order]
    obs = np.r_[True, s[1:] != s[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.flatnonzero(obs), len(obs)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def pearson(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def spearman(a, b):
    return pearson(rankdata(a), rankdata(b)) if len(a) > 2 else float("nan")


def regression_metrics(pred, z):
    pred, z = np.asarray(pred, dtype=np.float64), np.asarray(z, dtype=np.float64)
    return {"mse": float(((pred - z) ** 2).mean()), "mae": float(np.abs(pred - z).mean()), "pearson": pearson(pred, z),
            "spearman": spearman(pred, z)}


# ---------------------------------------------------------------------------------------------- the model
class InfluenceModel(nn.Module):
    def __init__(self, bert, dropout=None):
        super().__init__()
        self.bert = bert
        cfg = bert.config
        p = cfg.classifier_dropout if getattr(cfg, "classifier_dropout", None) is not None else cfg.hidden_dropout_prob
        self.dropout = nn.Dropout(p if dropout is None else dropout)
        self.head = nn.Linear(cfg.hidden_size, 1)
        nn.init.normal_(self.head.weight, std=cfg.initializer_range)       # BertPreTrainedModel._init_weights
        nn.init.zeros_(self.head.bias)

    def chunk_sums(self, ids, mask):
        """ids, mask (n, L): independent sequences. -> (last hidden states summed over the mask (n, H) fp32,
        token counts (n,) fp32)."""
        h = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state
        s = torch.where(mask.bool().unsqueeze(-1), h.float(), torch.zeros((), device=h.device))
        return s.sum(1), mask.sum(1).float()

    def head_fp32(self, pooled):
        with torch.autocast(device_type=pooled.device.type, enabled=False):
            return self.head(self.dropout(pooled.float())).squeeze(-1)

    def forward(self, ids, mask):
        """ids, mask (n, 2, CHUNK): MATES's layout. -> predictions (n,) fp32."""
        n = ids.shape[0]
        s, c = self.chunk_sums(ids.reshape(2 * n, -1), mask.reshape(2 * n, -1))
        pooled = s.view(n, 2, -1).sum(1) / c.view(n, 2).sum(1, keepdim=True)
        return self.head_fp32(pooled)


def build_bert(bert_dir=None, config=None):
    """BertModel without the pooler: from a local directory (bert-base-uncased) or from a BertConfig (tests)."""
    from transformers import BertModel

    if config is not None:
        return BertModel(config, add_pooling_layer=False)
    return BertModel.from_pretrained(bert_dir, add_pooling_layer=False)


def padded_layout(seqs):
    """WordPiece id arrays (<= MAX_LEN) -> ids, mask (n, 2, CHUNK) int64: MATES's padding to 1024 in two chunks."""
    ids = np.zeros((len(seqs), MAX_LEN), dtype=np.int64)
    mask = np.zeros((len(seqs), MAX_LEN), dtype=np.int64)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = s
        mask[i, :len(s)] = 1
    return ids.reshape(-1, 2, CHUNK), mask.reshape(-1, 2, CHUNK)


# ---------------------------------------------------------------------------------------------- scoring
class Scorer:
    """Bucketed (or padded) bf16 inference: score(blocks) -> one score per document."""

    def __init__(self, model, device, ctx, bucket=64, tokens=65536, padded=False, compile=False):
        self.model, self.device, self.ctx, self.padded = model, device, ctx, bool(padded)
        self.edges = [CHUNK] if padded else sorted(set(list(range(bucket, CHUNK, bucket)) + [CHUNK]))
        self.rows = {L: max(1, tokens // L) for L in self.edges}
        self.fn = torch.compile(model.chunk_sums, dynamic=False) if compile else model.chunk_sums
        self.pin = "cuda" in str(device)
        self.stats = {"batches": 0, "rows": 0, "tokens": 0, "padded_tokens": 0, "gpu_wait_s": 0.0}

    def _chunks(self, ids):
        if self.padded:
            return [ids[:CHUNK], ids[CHUNK:MAX_LEN]]
        return [ids[:CHUNK]] + ([ids[CHUNK:MAX_LEN]] if len(ids) > CHUNK else [])

    def _bucket(self, n):
        for L in self.edges:
            if n <= L:
                return L
        raise ValueError(f"chunk of {n} tokens")

    def _launch(self, L, items):
        rows = self.rows[L]
        ids = np.zeros((rows, L), dtype=np.int64)
        mask = np.zeros((rows, L), dtype=np.int64)
        docs = np.empty(len(items), dtype=np.int64)
        for i, (d, c) in enumerate(items):
            ids[i, :len(c)] = c
            mask[i, :len(c)] = 1
            docs[i] = d
        mask[len(items):, 0] = 1                       # dummy rows: one token, outputs dropped
        self.stats["batches"] += 1
        self.stats["rows"] += len(items)
        self.stats["tokens"] += int(mask[:len(items)].sum())
        self.stats["padded_tokens"] += rows * L
        t_ids, t_mask = torch.from_numpy(ids), torch.from_numpy(mask)
        if self.pin:
            t_ids, t_mask = t_ids.pin_memory(), t_mask.pin_memory()
        t_ids, t_mask = t_ids.to(self.device, non_blocking=True), t_mask.to(self.device, non_blocking=True)
        with torch.no_grad():
            with self.ctx:
                s, c = self.fn(t_ids, t_mask)
            proj = s.float() @ self.w                  # fp32, outside the autocast
        return docs, proj[:len(items)], c[:len(items)]

    def _collect(self, pending, proj, cnt):
        if pending is None:
            return
        docs, p, c = pending
        t0 = time.perf_counter()
        p, c = p.double().cpu().numpy(), c.double().cpu().numpy()
        self.stats["gpu_wait_s"] += time.perf_counter() - t0
        np.add.at(proj, docs, p)
        np.add.at(cnt, docs, c)

    def score(self, blocks, n_docs):
        """blocks: iterable of (document indices (m,), list of m WordPiece id arrays). -> scores (n_docs,) float64."""
        self.model.eval()
        self.w = self.model.head.weight.detach().float().reshape(-1)
        proj, cnt = np.zeros(n_docs, dtype=np.float64), np.zeros(n_docs, dtype=np.float64)
        buf = {L: [] for L in self.edges}
        pending = None
        for idx, seqs in blocks:
            for d, ids in zip(idx, seqs):
                for c in self._chunks(ids):
                    L = self._bucket(len(c)) if not self.padded else CHUNK
                    b = buf[L]
                    b.append((int(d), c))
                    if len(b) == self.rows[L]:
                        nxt = self._launch(L, b)
                        self._collect(pending, proj, cnt)        # the previous batch, while this one runs
                        pending = nxt
                        buf[L] = []
        for L in self.edges:
            if buf[L]:
                nxt = self._launch(L, buf[L])
                self._collect(pending, proj, cnt)
                pending = nxt
        self._collect(pending, proj, cnt)
        if (cnt <= 0).any():
            raise RuntimeError(f"{int((cnt <= 0).sum())} documents without tokens")
        return proj / cnt + float(self.model.head.bias.detach().float())


def score_arrays(scorer, seqs, block=4096):
    """Scores of in-memory WordPiece id arrays (validation, tests), in their order."""
    n = len(seqs)
    return scorer.score(((np.arange(i, min(i + block, n)), seqs[i:i + block]) for i in range(0, n, block)), n)


# ---------------------------------------------------------------------------------------------- fine-tuning
def split_train_val(n, val_share, seed=1234):
    """datasets.Dataset.train_test_split(test_size=val_share, seed=seed, shuffle=True): -> (train idx, val idx)."""
    n_val = int(math.ceil(val_share * n))
    perm = np.random.default_rng(seed).permutation(n)
    return perm[n_val:], perm[:n_val]


def decay_groups(model, wd):
    """The HF Trainer's weight-decay split: no decay for biases and LayerNorm weights."""
    no_decay = set()
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            if pn.endswith("bias") or isinstance(m, nn.LayerNorm):
                no_decay.add(id(p))
    params = list(model.parameters())
    return [{"params": [p for p in params if id(p) not in no_decay], "weight_decay": wd},
            {"params": [p for p in params if id(p) in no_decay], "weight_decay": 0.0}]


def _gather(x, world):
    if world == 1:
        return [x]
    out = [None] * world
    dist.all_gather_object(out, x)
    return out


def finetune(model, seqs, oracle, *, world, rank, device, ctx, scorer_kw, epochs=5, batch=256, micro=32, lr=5e-5,
             wd=0.01, val_share=0.1, eval_every=20, clip=1.0, seed=0, ddp=None, compile=False, log=print,
             on_record=None):
    """Fine-tune `model` (InfluenceModel) on (seqs[i], oracle[i]); every rank holds all labels. ddp: model -> wrapped
    model (DDP on the process group) or None (one process). -> (summary dict, records). The model ends with the best
    validation-Spearman weights."""
    t_start = time.perf_counter()
    oracle = np.asarray(oracle, dtype=np.float64)
    tr, va = split_train_val(len(oracle), val_share)
    mu, sd = float(oracle[tr].mean()), float(oracle[tr].std())
    if not sd > 0:
        raise ValueError("the oracle has no spread on the training split")
    z = (oracle - mu) / sd
    model.train()
    opt = torch.optim.AdamW(decay_groups(model, wd), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    steps_per_epoch = int(math.ceil(len(tr) / batch))
    total = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, (total - s) / total))
    wrapped = ddp(model) if ddp is not None else model
    fwd = torch.compile(wrapped, dynamic=False) if compile else wrapped
    scorer = Scorer(model, device, ctx, **scorer_kw)
    va_mine = va[rank::world]
    best = {"spearman": -math.inf, "step": None}
    best_state, records = None, []
    sse, cnt, step = 0.0, 0, 0
    step_s, eval_s, val_docs = [], 0.0, 0
    for epoch in range(epochs):
        order = tr[np.random.default_rng([seed, epoch]).permutation(len(tr))]
        for b in range(steps_per_epoch):
            t_step = time.perf_counter()
            G = order[b * batch:(b + 1) * batch]
            mine = G[rank::world]
            # the same number of fixed-shape micro-batches on every rank (pads: weight 0); every backward is reduced
            # (DDP averages over ranks; the accumulated sum is the gradient of the global batch's MSE)
            n_micro = max(1, int(math.ceil(math.ceil(len(G) / world) / micro)))
            for part in np.array_split(mine, n_micro):
                w = np.zeros(micro, dtype=np.float32)
                w[:len(part)] = 1.0
                rows = np.concatenate([part, np.full(micro - len(part), G[0], dtype=part.dtype)])
                ids, mask = padded_layout([seqs[j] for j in rows])
                ids, mask = torch.from_numpy(ids).to(device), torch.from_numpy(mask).to(device)
                zt = torch.from_numpy(z[rows].astype(np.float32)).to(device)
                wt = torch.from_numpy(w).to(device)
                with ctx:
                    pred = fwd(ids, mask)
                err = (pred.float() - zt) ** 2 * wt
                (err.sum() * world / len(G)).backward()
                sse += float(err.detach().sum())
                cnt += len(part)
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            step_s.append(time.perf_counter() - t_step)
            if step % eval_every == 0 or step == total:
                t_eval = time.perf_counter()
                tr_err = torch.tensor([sse, float(cnt)], dtype=torch.float64, device=device)
                if world > 1:
                    dist.all_reduce(tr_err)
                sse, cnt = 0.0, 0
                pred = score_arrays(scorer, [seqs[j] for j in va_mine])
                val_docs += len(va_mine)
                eval_s += time.perf_counter() - t_eval
                model.train()
                parts = _gather((va_mine, pred), world)
                p_all = np.empty(len(va))
                pos = {int(j): k for k, j in enumerate(va)}
                for idx_r, pred_r in parts:
                    p_all[[pos[int(j)] for j in idx_r]] = pred_r
                m = regression_metrics(p_all, z[va])
                rec = {"step": step, "epoch": epoch + (b + 1) / steps_per_epoch, "lr": opt.param_groups[0]["lr"],
                       "train_mse": float(tr_err[0] / tr_err[1].clamp_min(1)),
                       **{f"val_{k}": v for k, v in m.items()}, "seconds": time.perf_counter() - t_start}
                records.append(rec)
                if on_record is not None:
                    on_record(rec)
                if m["spearman"] > best["spearman"]:
                    best = {"spearman": m["spearman"], "step": step, **{f"val_{k}": v for k, v in m.items()}}
                    best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
                if rank == 0:
                    log(f"[mates] influence model step {step}/{total}: train mse {rec['train_mse']:.4f}, val mse "
                        f"{m['mse']:.4f}, spearman {m['spearman']:.4f} (best {best['spearman']:.4f} @ {best['step']})")
    del fwd, wrapped
    if best_state is not None:
        model.load_state_dict(best_state)
    else:                                  # no finite validation Spearman (e.g. constant predictions): the last model
        best = {"spearman": None, "step": None}
    model.eval()
    summary = {"n": len(oracle), "n_train": len(tr), "n_val": len(va), "mean": mu, "std": sd, "steps": total,
               "steps_per_epoch": steps_per_epoch, "best": best, "seconds": time.perf_counter() - t_start,
               "step_s_median": float(np.median(step_s)), "step_s_mean": float(np.mean(step_s)),
               "eval_s": eval_s, "val_docs": val_docs}       # this rank's: step wall (sync each micro), validations
    return summary, records
