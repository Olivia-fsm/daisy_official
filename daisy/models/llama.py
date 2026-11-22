"""Llama-style decoder (RMSNorm, rotary positions, SwiGLU) that compiles with torch.compile (no complex tensors).

Two additions for DAiSY, both off unless the model is called with the corresponding argument:
  doc_ids   document-masked attention on packed rows (daisy/models/doc_mask.py)
  gates     label gates on the tracked Linear layers (daisy/models/label_gate.py)
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from daisy.models.base import CausalSelfAttention, GPTBase
from daisy.models.doc_mask import doc_causal_block_mask, doc_masked_attention
from daisy.models.label_gate import gated_linear


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    # cos and sin parts stacked in the last dimension stand in for complex numbers
    return torch.stack((torch.cos(freqs), torch.sin(freqs)), dim=-1)


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """freqs_cis: (seq_len, head_dim / 2, 2); x: (bsz, seq_len, n_head, head_dim / 2, 2)"""
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape[:-1] == (x.shape[1], x.shape[-2])
    shape = [1 if i != 1 and i != ndim - 2 else d for i, d in enumerate(x.shape[:-1])] + [2]
    return freqs_cis.view(*shape)


def apply_rotary_emb(q, k, freqs_cis):
    """q, k: (B, T, nh, hs); freqs_cis: (T, hs / 2, 2) -> rotated q, k of the same shape."""
    q = q.float().reshape(*q.shape[:-1], -1, 2)
    k = k.float().reshape(*k.shape[:-1], -1, 2)
    freqs_cis = _reshape_for_broadcast(freqs_cis, q)

    q_cos = q[..., 0] * freqs_cis[..., 0] - q[..., 1] * freqs_cis[..., 1]
    q_sin = q[..., 0] * freqs_cis[..., 1] + q[..., 1] * freqs_cis[..., 0]
    k_cos = k[..., 0] * freqs_cis[..., 0] - k[..., 1] * freqs_cis[..., 1]
    k_sin = k[..., 0] * freqs_cis[..., 1] + k[..., 1] * freqs_cis[..., 0]

    q_out = torch.stack((q_cos, q_sin), dim=-1).reshape(q.shape).flatten(3)
    k_out = torch.stack((k_cos, k_sin), dim=-1).reshape(k.shape).flatten(3)
    return q_out, k_out


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(2 * config.n_embd * 4 / 3)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x, gates=None):
        if gates is not None:
            h = nn.functional.silu(gated_linear(self.w1, x, gates)) * gated_linear(self.w2, x, gates)
            return gated_linear(self.c_proj, h, gates)
        return self.c_proj(nn.functional.silu(self.w1(x)) * self.w2(x))


class LlamaAttention(CausalSelfAttention):
    def forward(self, x, freqs_cis, block_mask=None, gates=None):
        B, T, C = x.size()

        qkv = self.c_attn(x) if gates is None else gated_linear(self.c_attn, x, gates)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head)
        q = q.view(B, T, self.n_head, C // self.n_head)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        q, k = q.transpose(1, 2), k.transpose(1, 2)                           # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if block_mask is not None:
            # packed documents: a token attends only to earlier tokens of its own document
            y = doc_masked_attention(q, k, v, block_mask)
        else:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout,
                                                                 is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y) if gates is None else gated_linear(self.c_proj, y, gates))


class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.attn = LlamaAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.mlp = LlamaMLP(config)

    def forward(self, x, freqs_cis, block_mask=None, gates=None):
        x = x + self.attn(self.ln_1(x), freqs_cis, block_mask=block_mask, gates=gates)
        return x + (self.mlp(self.ln_2(x)) if gates is None else self.mlp(self.ln_2(x), gates=gates))


class Llama(GPTBase):
    def __init__(self, config):
        super().__init__(config)
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config

        self.head_dim = config.n_embd // config.n_head
        self.freqs_cis = precompute_freqs_cis(self.head_dim, config.sequence_length)

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([LlamaBlock(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd, eps=config.rmsnorm_eps),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if not config.untied_embeds:
            self.transformer.wte.weight = self.lm_head.weight          # weight tying

        self.apply(self._init_weights)
        # scaled init of the residual projections, as in GPT-2
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=self.config.init_std / math.sqrt(2 * config.n_layer))

    def forward(self, idx, targets=None, get_logits=False, doc_ids=None, gates=None):
        """idx, targets (B, T) long (targets -1 = no loss term). doc_ids (B, T): document index per position, for
        document-masked attention (None = plain causal attention). gates: the label gates of this forward (None = off).
        -> {"logits": (B, T, V) if get_logits and targets are given, "loss": mean token loss or None}."""
        b, t = idx.size()
        assert t <= self.config.sequence_length, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)

        x = self.transformer.drop(self.transformer.wte(idx))
        freqs_cis = self.freqs_cis.to(x.device)[pos]
        block_mask = doc_causal_block_mask(doc_ids) if doc_ids is not None else None
        for block in self.transformer.h:
            x = block(x, freqs_cis=freqs_cis, block_mask=block_mask, gates=gates)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])     # inference: only the last position
            loss = None
        return {"logits": logits if get_logits else None, "loss": loss}
