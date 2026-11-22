"""The index's token store (DocStore: tokens.u16 + offsets, phi.f16) and the packing of whole documents into
fixed-length rows with document ids (for document-masked attention and per-doc labels).

Documents are taken whole (capped at `max_doc_tokens`, EOS kept as the last token), concatenated, and cut into rows of
seq_len + 1 tokens: x = row[:-1], y = row[1:]. A document may continue in the next row (no padding, no dropped tokens).
Targets that cross a document boundary (the EOS position predicting the next document's first token) are set to -1,
the trainer's ignore_index, so every loss term belongs to exactly one document. `doc` gives, per position, the index
into `docs` (global document ids) of the document the input token belongs to.
"""

import os

import numpy as np


class DocStore:
    """Token store written by daisy/index/build_index.py (tokens.u16 + offsets.i64.npy), or with 4-byte tokens
    (tokens.u32: vocabularies above 65,536, e.g. the multilingual held-out index of
    daisy/multilingual/build_mling_index.py).

    Two read paths with identical results: doc()/phi() through memmaps (simple, used for the one-off held-out set),
    and read_segment()/read_phi() through os.pread on file descriptors (used by the training stream). On a network
    file system a memmap page fault per random document is slow and holds the GIL; pread releases it, so a thread
    pool reads many documents at once.
    The offsets (8 bytes per document) are held in RAM."""

    def __init__(self, index_dir, phi_path=None):
        self.offsets = np.load(f"{index_dir}/offsets.i64.npy")
        tok_path = f"{index_dir}/tokens.u16"
        if not os.path.exists(tok_path) and os.path.exists(f"{index_dir}/tokens.u32"):
            tok_path = f"{index_dir}/tokens.u32"
        self.dtype = np.dtype(np.uint32 if tok_path.endswith(".u32") else np.uint16)
        self.tokens = np.memmap(tok_path, dtype=self.dtype, mode="r")
        self.N = len(self.offsets) - 1
        p = phi_path or f"{index_dir}/phi.f16"     # e.g. a copy on a RAM disk for selection (many random phi reads)
        self.phi_rows = np.memmap(p, dtype=np.float16, mode="r", shape=(self.N, 768)) if os.path.exists(p) else None
        self.has_phi = self.phi_rows is not None
        self._fd_tok = os.open(tok_path, os.O_RDONLY)
        self._fd_phi = os.open(p, os.O_RDONLY) if self.has_phi else None

    def read_segment(self, i, start, end, max_doc_tokens=None):
        """doc(i, max_doc_tokens)[start:end] via pread (reads only those tokens)."""
        s, e = int(self.offsets[i]), int(self.offsets[i + 1])
        return _pread_capped(self._fd_tok, s, e - s, start, end, max_doc_tokens, self.dtype)

    def read_phi(self, i):
        return np.frombuffer(_pread_exact(self._fd_phi, 2 * 768, 2 * 768 * int(i)), dtype=np.float16)

    def doc(self, i, max_doc_tokens=None):
        s, e = int(self.offsets[i]), int(self.offsets[i + 1])
        t = np.asarray(self.tokens[s:e])
        if max_doc_tokens is not None and len(t) > max_doc_tokens:
            t = np.concatenate([t[: max_doc_tokens - 1], t[-1:]])  # keep the trailing EOS
        return t

    def phi(self, rows):
        if self.phi_rows is None:
            raise FileNotFoundError("index has no phi.f16")
        return np.asarray(self.phi_rows[np.asarray(rows)])

    def lengths(self, rows, max_doc_tokens=None):
        L = np.asarray(self.offsets[np.asarray(rows) + 1] - self.offsets[np.asarray(rows)])
        return np.minimum(L, max_doc_tokens) if max_doc_tokens is not None else L


def _pread_exact(fd, n, off):
    buf = os.pread(fd, n, off)
    while len(buf) < n:                     # pread may return short reads; finish them
        more = os.pread(fd, n - len(buf), off + len(buf))
        if not more:
            raise EOFError(f"short read at {off}: {len(buf)} of {n} bytes")
        buf += more
    return buf


def _pread_capped(fd, s, L, start, end, cap, dtype=np.uint16):
    """Tokens [start, end) of the capped document whose tokens (dtype: uint16, or uint32 for vocabularies above
    65,536) occupy [s, s + L) of the file. The capped document is t[:cap - 1] + t[-1:] (EOS kept) when L > cap, else t."""
    dt = np.dtype(dtype)
    b = dt.itemsize
    if end <= start:
        return np.zeros(0, dtype=dt)
    if cap is None or L <= cap:
        return np.frombuffer(_pread_exact(fd, b * (end - start), b * (s + start)), dtype=dt)
    head_end = min(end, cap - 1)
    parts = []
    if start < head_end:
        parts.append(np.frombuffer(_pread_exact(fd, b * (head_end - start), b * (s + start)), dtype=dt))
    if end == cap:                          # the segment includes the kept last token
        parts.append(np.frombuffer(_pread_exact(fd, b, b * (s + L - 1)), dtype=dt))
    return np.concatenate(parts) if len(parts) > 1 else parts[0]


def pack(store, doc_rows, seq_len, max_doc_tokens=None):
    """Pack the given documents (global ids, in order). Returns dict with
    x, y: (R, seq_len) int64; doc: (R, seq_len) int32 index into `docs`; docs: int64 global ids;
    n_rows R = floor(total / (seq_len + 1)); `rest` = number of trailing tokens not placed (carry them over)."""
    parts = [store.doc(int(i), max_doc_tokens) for i in doc_rows]
    lens = np.array([len(p) for p in parts], dtype=np.int64)
    flat = np.concatenate(parts).astype(np.int64)
    owner = np.repeat(np.arange(len(parts), dtype=np.int32), lens)
    W = seq_len + 1
    R = len(flat) // W
    rows = flat[: R * W].reshape(R, W)
    own = owner[: R * W].reshape(R, W)
    x, y = rows[:, :-1].copy(), rows[:, 1:].copy()
    doc = own[:, :-1].copy()
    y[own[:, 1:] != own[:, :-1]] = -1
    return {"x": x, "y": y, "doc": doc, "docs": np.asarray(doc_rows, dtype=np.int64), "n_rows": R,
            "rest": int(len(flat) - R * W)}

