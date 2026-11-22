"""Chat data for instruction tuning (supervised fine-tuning, SFT) of pretrained checkpoints (GPT-2 tokens).

Rendering follows the Tulu chat template, written out as ordinary text (GPT-2 has no chat tokens):
    <|system|>\\n{content}\\n   <|user|>\\n{content}\\n   <|assistant|>\\n{content}<EOT>\\n   (no \\n after the last turn)
Loss mask (as in open-instruct): 1 on each assistant turn's content and its closing EOT, 0 on everything else (role
markers, system and user turns, the newline after EOT). Literal "<|endoftext|>" strings inside a message are rewritten
so they cannot become the EOT id.

Packing: whole conversations, in a seeded order, are packed greedily into rows of seq_len + 1 tokens; a conversation
is never split across rows (the next one starts a new row when it does not fit; the rest is EOT padding, its own
document, never a target). A conversation longer than a row is cut to it and is kept only if at least one assistant
token remains a target. Attention is document-masked, so packed conversations never see each other.
"""

import numpy as np

EOT = 50256
ROLE = {"system": "<|system|>\n", "user": "<|user|>\n", "assistant": "<|assistant|>\n"}


def render(messages, encode):
    """messages: [{"role", "content"}]; encode: str -> list of GPT-2 ids (no special tokens).
    -> tokens (uint16), mask (uint8). Raises ValueError on an unknown role."""
    toks, mask = [], []

    def add(ids, m):
        toks.extend(ids)
        mask.extend([m] * len(ids))

    for i, msg in enumerate(messages):
        role, content = msg["role"], str(msg["content"]).replace("<|endoftext|>", "<endoftext>")
        if role == "assistant":
            add(encode(ROLE["assistant"]), 0)
            add(encode(content), 1)
            add([EOT], 1)
            if i < len(messages) - 1:
                add(encode("\n"), 0)
        elif role in ("system", "user"):
            add(encode(ROLE[role] + content + "\n"), 0)
        else:
            raise ValueError(f"unknown role {role!r}")
    return np.asarray(toks, dtype=np.uint16), np.asarray(mask, dtype=np.uint8)


class ChatSet:
    """One split written by prep_tulu.py: {split}.tok (uint16), {split}.mask (uint8), {split}.idx (int64 offsets,
    n + 1 entries)."""

    def __init__(self, root, split):
        self.tok = np.memmap(f"{root}/{split}.tok", dtype=np.uint16, mode="r")
        self.mask = np.memmap(f"{root}/{split}.mask", dtype=np.uint8, mode="r")
        self.offs = np.fromfile(f"{root}/{split}.idx", dtype=np.int64)
        if self.offs[-1] != len(self.tok) or len(self.tok) != len(self.mask):
            raise ValueError(f"{root}/{split}: index {self.offs[-1]}, tokens {len(self.tok)}, mask {len(self.mask)}")
        self.lens = np.diff(self.offs)

    def __len__(self):
        return len(self.lens)

    def trainable(self, seq_len):
        """Conversations that keep >= 1 target after being cut to one row (a target sits at positions 1..n-1)."""
        W = seq_len + 1
        cm = np.concatenate([[0], np.cumsum(self.mask, dtype=np.int64)])
        n = np.minimum(self.lens, W)
        s = self.offs[:-1]
        return (cm[s + n] - cm[s + 1]) > 0

    def pack(self, seq_len, order):
        """Greedy packing of the conversations in `order` -> list of rows, each a list of (conversation, n_tokens)."""
        W = seq_len + 1
        rows, cur, used = [], [], 0
        for c in order:
            n = int(min(self.lens[c], W))
            if used + n > W:
                rows.append(cur)
                cur, used = [], 0
            cur.append((int(c), n))
            used += n
        if cur:
            rows.append(cur)
        return rows

    def batch(self, rows, seq_len):
        """-> x, y (R, T) int64 and doc (R, T) int32. y is the next token where it is an assistant token of the same
        conversation, else -1."""
        W = seq_len + 1
        buf = np.full((len(rows), W), EOT, dtype=np.int64)
        m = np.zeros((len(rows), W), dtype=np.uint8)
        own = np.zeros((len(rows), W), dtype=np.int32)
        for r, row in enumerate(rows):
            p = 0
            for j, (c, n) in enumerate(row):
                s = self.offs[c]
                buf[r, p:p + n] = self.tok[s:s + n]
                m[r, p:p + n] = self.mask[s:s + n]
                own[r, p:p + n] = j
                p += n
            own[r, p:] = len(row)                       # padding: its own document, never a target
        x, y = buf[:, :-1].copy(), buf[:, 1:].copy()
        keep = (m[:, 1:] == 1) & (own[:, 1:] == own[:, :-1])
        y[~keep] = -1
        return x, y, own[:, :-1].copy()
