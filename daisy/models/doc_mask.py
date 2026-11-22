"""Document-masked causal attention for packed sequences.

A token attends only to earlier tokens of its own document, so each document's loss depends only on its own tokens
and the per-document utility labels are exact. RoPE is relative, so positions need no reset at document starts: within
a document, q·k sees the same relative offsets as if the document started at position 0.

Used only when the model is called with `doc_ids`; without doc_ids the attention is the standard causal one.
"""

import torch

_FLEX = None


def flex_attention_fn():
    """flex_attention, compiled once on GPU (the eager version is a slow reference implementation)."""
    global _FLEX
    if _FLEX is None:
        from torch.nn.attention.flex_attention import flex_attention

        _FLEX = torch.compile(flex_attention, dynamic=False) if torch.cuda.is_available() else flex_attention
    return _FLEX


def doc_causal_block_mask(doc_ids):
    """doc_ids: (B, T) integer tensor; equal values = same document. Returns a flex_attention BlockMask."""
    from torch.nn.attention.flex_attention import create_block_mask

    B, T = doc_ids.shape

    def mask_mod(b, h, q_idx, kv_idx):
        return (doc_ids[b, q_idx] == doc_ids[b, kv_idx]) & (q_idx >= kv_idx)

    if doc_ids.is_cuda:  # compiled mask construction never materializes the dense (B, T, T) mask
        return create_block_mask(mask_mod, B, None, T, T, device=doc_ids.device, _compile=True)
    return create_block_mask(mask_mod, B, None, T, T, device=doc_ids.device)


def doc_masked_attention(q, k, v, block_mask):
    """q, k, v: (B, nh, T, hs). Casts q, k to v's dtype (RoPE upcasts them to fp32; flex has no autocast rule)."""
    return flex_attention_fn()(q.to(v.dtype), k.to(v.dtype), v, block_mask=block_mask)
