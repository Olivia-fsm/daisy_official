"""GPT-2 tokens -> text -> WordPiece ids: the input of the data influence model.

MATES decodes its LM's tokens with special tokens skipped and re-encodes the text with the bert-base-uncased tokenizer,
truncated to 1024 tokens ([CLS] ... [SEP]). Here the documents are read from the pool by key and capped like training
(--daisy_max_doc_tokens, EOS kept), <|endoftext|> is skipped, and both tokenizers are the Rust `tokenizers`
objects that transformers' fast tokenizers wrap: GPT-2 from its tokenizer.json (daisy/tok.py), BERT from the
influence model's directory (tokenizer.json, or vocab.txt as a lowercasing BertWordPieceTokenizer, which is what
bert-base-uncased's tokenizer.json holds).

EncodePool is the CPU worker pool of a rank: `workers` threads, each with its own tokenizers, read documents with
os.pread and decode / encode them in batch calls; all three release the GIL, and TOKENIZERS_PARALLELISM=false keeps
each call on its thread's core, so the threads run in parallel. (Threads, not processes: a spawned process would
re-import the trainer's __main__, train.py, in every worker.) Chunks of keys go out in order with at most
2 x workers in flight and come back in submission order, so the encoding runs ahead of the GPU that consumes it.
workers = 0 encodes in the calling thread.
"""

import collections
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

EOS = 50256
MAX_LEN = 1024
_cpu_clock = getattr(time, "thread_time", time.process_time)     # thread CPU time (Linux); the process's elsewhere


class TextEncoder:
    def __init__(self, gpt2_json, bert_dir, max_len=MAX_LEN):
        from tokenizers import Tokenizer

        self.gpt2 = Tokenizer.from_file(gpt2_json)
        tj, vocab = os.path.join(bert_dir, "tokenizer.json"), os.path.join(bert_dir, "vocab.txt")
        if os.path.exists(tj):
            self.bert = Tokenizer.from_file(tj)
        elif os.path.exists(vocab):
            from tokenizers import BertWordPieceTokenizer

            self.bert = BertWordPieceTokenizer(vocab, lowercase=True)
        else:
            raise FileNotFoundError(f"{bert_dir}: no tokenizer.json or vocab.txt")
        self.bert.no_padding()
        self.bert.enable_truncation(max_length=max_len)
        if self.bert.get_vocab_size() > 65535:
            raise ValueError("WordPiece ids are stored as uint16")

    def texts(self, docs):
        return self.gpt2.decode_batch([np.asarray(d)[np.asarray(d) != EOS].tolist() for d in docs],
                                      skip_special_tokens=True)

    def encode(self, docs):
        """docs: GPT-2 token arrays. -> WordPiece id arrays (uint16, at most max_len, [CLS] first, [SEP] last)."""
        return [np.asarray(e.ids, dtype=np.uint16) for e in self.bert.encode_batch(self.texts(docs))]


def read_doc(pool, key, cap):
    """The capped document (EOS kept, as training) by os.pread (DsPool.read_segment; the GIL is released)."""
    _, ends, _, starts = pool.shard(int(key) >> 32)
    i = int(key) & 0xFFFFFFFF
    n = int(ends[i] - starts[i])
    return pool.read_segment(key, 0, min(n, cap) if cap else n, cap)


def split_flat(flat, lens):
    offs = np.concatenate([[0], np.cumsum(lens)])
    return [flat[offs[i]:offs[i + 1]] for i in range(len(lens))]


class EncodePool:
    def __init__(self, pool_root, cap, gpt2_json, bert_dir, workers=0):
        from daisy.data.pool import DsPool

        os.environ["TOKENIZERS_PARALLELISM"] = "false"      # one core per call; the threads are the parallelism
        self.pool, self.cap, self.args = DsPool(pool_root), cap, (gpt2_json, bert_dir)
        self.workers = int(workers)
        self.local = threading.local()
        self.ex = ThreadPoolExecutor(self.workers, thread_name_prefix="daisy-wp") if self.workers > 0 else None
        self.cpu_s = 0.0           # CPU seconds of the encoding (summed over threads)
        self._lock = threading.Lock()

    def _encoder(self):
        enc = getattr(self.local, "enc", None)
        if enc is None:
            enc = self.local.enc = TextEncoder(*self.args)
        return enc

    def _encode_keys(self, keys):
        """-> (concatenated WordPiece ids uint16, lengths int32)."""
        t0 = _cpu_clock()
        ids = self._encoder().encode([read_doc(self.pool, k, self.cap) for k in keys])
        lens = np.array([len(a) for a in ids], dtype=np.int32)
        flat = np.concatenate(ids) if ids else np.zeros(0, dtype=np.uint16)
        with self._lock:
            self.cpu_s += _cpu_clock() - t0
        return flat, lens

    def map(self, chunks):
        """chunks: iterable of key arrays. Yields (concatenated ids, lengths) per chunk, in order."""
        if self.ex is None:
            for c in chunks:
                yield self._encode_keys(c)
            return
        q, it = collections.deque(), iter(chunks)
        for c in it:
            q.append(self.ex.submit(self._encode_keys, c))
            if len(q) >= 2 * self.workers:
                break
        while q:
            out = q.popleft().result()
            nxt = next(it, None)
            if nxt is not None:
                q.append(self.ex.submit(self._encode_keys, nxt))
            yield out

    def encode_keys(self, keys, chunk=512):
        """All documents of `keys`. -> list of WordPiece id arrays, in the order of keys."""
        keys = np.asarray(keys, dtype=np.int64)
        out = []
        for flat, lens in self.map([keys[i:i + chunk] for i in range(0, len(keys), chunk)]):
            out += split_flat(flat, lens)
        return out

    def close(self):
        if self.ex is not None:
            self.ex.shutdown(wait=True, cancel_futures=True)
            self.ex = None
