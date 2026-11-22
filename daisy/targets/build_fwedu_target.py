#!/usr/bin/env python
"""The FineWeb-Edu target: a selection target made of FineWeb-Edu documents with an educational score of at least
--min_int_score (the int_score of the FineWeb-Edu classifier, 0-5; default 4), in place of the benchmark target or the
FineWeb topic target of build_fineweb_target.py. Everything else about the target is as there: --n_tasks target tasks
(default 16, as the 16 topic tasks) in the task format of daisy/labels/targets.py, each document cut at
--max_doc_tokens with its EOS kept, and the run takes its fixed seeded subset of --daisy_target_tokens per task. The
tasks here are random equal shards of the documents (FineWeb-Edu documents carry no topic of the index), so the
combined target (the mean of the tasks' gradients) is about the mean gradient of all target documents.

--edu_dir is a local copy of the FineWeb-Edu score-2 dataset (documents with score >= 2), one directory per dump:
EDU_DIR/data/<dump>/*.parquet (the layout of the public dataset HuggingFaceFW/fineweb-edu-score-2) or
EDU_DIR/<dump>/*.parquet. Two column layouts are read (daisy/targets/edu_scores.read_edu_table): the public dataset's
flat columns (id, text, url, dump, int_score, score, ...), or a copy that holds every field but id and text in a
struct column `metadata` (id, text, metadata.url, .dump, .int_score, .score).

Where the documents come from, and what is removed (in this order; counts in report.json):
  1. the first --files parquet files (sorted) of dump --dump of --edu_dir; --dump must be a crawl snapshot that the
     pool does not contain, so that no document is the same crawl record as a pool document; rows with
     int_score < --min_int_score are dropped;
  2. documents whose text hash or URL hash (daisy/index/build_index.h64, the index's own hashes) equals that of any
     index document (its training documents and its held-out FineWeb documents): never a target or held-out document
     that a run can also train on verbatim or from the same URL; then repeats among the candidates (same text or URL
     hash; the first one kept);
  3. in a seeded random order (--seed), tokenized with the index's tokenizer (build_index.get_tokenizer; the same
     GPT-2 tokens as the pool) + EOS: documents sharing a --ngram-token n-gram with any evaluation item (the held-out
     benchmark splits val.bin under --bench, written by daisy/targets/prepare_benchmarks.py, and, with --lmeval, the
     splits that only lm-evaluation-harness uses, read from the offline Hugging Face datasets cache: LMEVAL below;
     n-grams in more than --max_items_per_ngram items are dropped as boilerplate), or sharing at least half the
     n-grams of a held-out FineWeb document (the first --val_docs_per_topic per topic, as the held-out FineWeb loss
     reads them: near-duplicates).
The surviving documents, in that random order: the first --val_docs are the held-out FineWeb-Edu set
(OUT/<prefix>heldout/val.bin, val.len: never a target, for a held-out FineWeb-Edu loss, daisy/eval/heldout_ckpt.py),
the next --n_tasks x --docs_per_task the target tasks OUT/<prefix><tt>/train.bin, train.len (task k = documents
k*docs_per_task .. ). docs.parquet lists every written document (id, url, dump, file, int_score, score, tokens, role);
every task is read back (load_items) and checked equal. Near-duplicates of pool documents that are not exact text or
URL repeats are not removed (exact hashes only); daisy/targets/pool_copies.py counts the exact copies in the whole
pool.

  python daisy/targets/build_fwedu_target.py --index INDEX --bench BENCH --edu_dir DATAPATH/fineweb-edu-score-2 \\
      --out TARGETS/fineweb_edu
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from daisy.data.docstore import DocStore  # noqa: E402
from daisy.index.build_index import get_tokenizer, h64  # noqa: E402
from daisy.labels.targets import EOT, TASKS, _target_segments, load_items  # noqa: E402
from daisy.runmeta import write_meta  # noqa: E402
from daisy.targets.build_fineweb_target import PAD  # noqa: E402
from daisy.targets.edu_scores import dump_files, read_edu_table  # noqa: E402

P = np.uint64(1000003)
LMEVAL = {  # task: (HF path, config, split, text of the item with its correct answer)
    "piqa": ("baber/piqa", None, "validation", lambda e: f"{e['goal']} {[e['sol1'], e['sol2']][int(e['label'])]}"),
    "winogrande": ("allenai/winogrande", "winogrande_xl", "validation",
                   lambda e: e["sentence"].replace("_", e["option" + str(e["answer"])])),
    "openbookqa": ("allenai/openbookqa", "main", "test",
                   lambda e: f"{e['question_stem']} {e['choices']['text'][e['choices']['label'].index(e['answerKey'])]}"),
    "lambada_openai": ("EleutherAI/lambada_openai", "default", "test", lambda e: e["text"]),
}


def ngram_hashes(t, n):
    """Polynomial hashes (mod 2^64) of every n-gram of the 1-d token array t."""
    t = np.asarray(t, np.uint64)
    m = len(t) - n + 1
    if m <= 0:
        return np.zeros(0, np.uint64)
    h = np.zeros(m, np.uint64)
    for j in range(n):
        h = h * P + t[j: j + m]
    return h


def hf_items(enc, log):
    """-> {task: list of token arrays}: the items of LMEVAL (the evaluation splits of lm-evaluation-harness that are
    not a val.bin of the benchmark tasks), each the question with its correct answer. A task whose dataset cannot be
    loaded (no `datasets` package, not in the offline cache) is skipped with a message."""
    out = {}
    try:
        from datasets import load_dataset
    except ImportError as ex:
        log(f"[fwedu] no lm-eval items ({ex})")
        return out
    for task, (path, name, split, fmt) in LMEVAL.items():
        try:
            ds = load_dataset(path, name, split=split)
            out[task] = [np.array(enc.encode_ordinary(fmt(e)), np.uint16) for e in ds]
        except Exception as ex:  # noqa: BLE001
            log(f"[fwedu] lm-eval items of {task} unavailable offline ({type(ex).__name__}: {ex}); skipped")
    return out


class Table:
    """n-gram hash -> items, over all groups. groups: [(name, items (token arrays), drop_first)]; with drop_first an
    item's first token is not part of its n-grams (the item is tokenized without the leading space that it has inside
    a web document)."""

    def __init__(self, groups, n, max_items):
        hs, ids = [], []
        self.group_names = [g for g, _, _ in groups]
        self.meta, self.group_of, self.last, self.nng = [], [], [], []
        for gi, (g, items, drop_first) in enumerate(groups):
            for k, t in enumerate(items):
                body = t[1:] if drop_first else t
                h = ngram_hashes(body, n)
                self.meta.append((g, k, body))
                self.group_of.append(gi)
                self.nng.append(len(h))
                self.last.append(h[-1] if len(h) else np.uint64(0))
                if len(h):
                    u = np.unique(h)
                    hs.append(u)
                    ids.append(np.full(len(u), len(self.meta) - 1, np.int64))
        hs, ids = np.concatenate(hs), np.concatenate(ids)
        u, inv, cnt = np.unique(hs, return_inverse=True, return_counts=True)
        keep = cnt[inv] <= max_items
        self.dropped = int((cnt > max_items).sum())
        o = np.argsort(hs[keep], kind="stable")
        self.hs, self.ids = hs[keep][o], ids[keep][o]
        self.group_of, self.nng = np.array(self.group_of), np.array(self.nng)
        self.last = np.array(self.last, dtype=np.uint64)
        self.n_items = len(self.meta)

    def scan(self, docs, n):
        """-> (doc, item, distinct matched n-grams, last n-gram matched) per matching pair, and the token count."""
        lens = np.array([len(d) for d in docs], np.int64)
        allt = np.concatenate(docs).astype(np.uint64)
        doc_of = np.repeat(np.arange(len(docs)), lens)
        h = ngram_hashes(allt, n)
        pos = np.flatnonzero(doc_of[: len(h)] == doc_of[n - 1: n - 1 + len(h)])   # windows inside one document
        h = h[pos]
        lo = np.searchsorted(self.hs, h, "left")
        hit = (lo < len(self.hs)) & (self.hs[np.minimum(lo, len(self.hs) - 1)] == h)
        pos, h, lo = pos[hit], h[hit], lo[hit]
        cnt = np.searchsorted(self.hs, h, "right") - lo
        rep = np.repeat(np.arange(len(lo)), cnt)
        idx = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt) + lo[rep]
        pair = doc_of[pos[rep]] * self.n_items + self.ids[idx]
        hv = self.hs[idx]
        o = np.lexsort((hv, pair))
        pair, hv = pair[o], hv[o]
        new = np.r_[True, (pair[1:] != pair[:-1]) | (hv[1:] != hv[:-1])] if len(pair) else np.zeros(0, bool)
        pair, hv = pair[new], hv[new]
        item = pair % self.n_items
        is_last = (hv == self.last[item]) & (self.nng[item] > 0)
        up, start, k = np.unique(pair, return_index=True, return_counts=True)
        last_any = np.logical_or.reduceat(is_last, start) if len(start) else np.zeros(0, bool)
        return up // self.n_items, up % self.n_items, k, last_any, int(lens.sum())


def write_split(path, docs, split):
    """The task format: <split>.bin (each document padded with EOS to a multiple of PAD) and <split>.len."""
    os.makedirs(path, exist_ok=True)
    lens = np.array([-(-len(d) // PAD) * PAD for d in docs], dtype=np.int64)
    buf = np.full(int(lens.sum()), EOT, dtype=np.uint16)
    o = 0
    for d, n in zip(docs, lens):
        buf[o:o + len(d)] = d
        o += int(n)
    buf.tofile(os.path.join(path, f"{split}.bin"))
    with open(os.path.join(path, f"{split}.len"), "wb") as f:
        np.save(f, lens)
    return int(lens.sum())


def read_candidates(files, min_score):
    """-> dict of lists (id, text, url, dump, file, int_score, score) of the rows with int_score >= min_score, and
    the number of rows read."""
    import pyarrow.compute as pc

    out = {k: [] for k in ("id", "text", "url", "dump", "file", "int_score", "score")}
    n_rows = 0
    for fi, f in enumerate(files):
        t = read_edu_table(f, ["id", "text", "url", "dump", "int_score", "score"])
        n_rows += t.num_rows
        t2 = t.filter(pc.greater_equal(t.column("int_score"), min_score))
        for k in ("id", "text", "url", "dump", "int_score", "score"):
            out[k] += t2.column(k).to_pylist()
        out["file"] += [os.path.basename(f)] * t2.num_rows
        print(f"[fwedu] {f}: {t.num_rows} rows, {t2.num_rows} with int_score >= {min_score}", flush=True)
    return out, n_rows


def cut(t, max_doc_tokens):
    """DocStore.doc's cut: the first max_doc_tokens - 1 tokens and the trailing EOS."""
    return np.concatenate([t[: max_doc_tokens - 1], t[-1:]]) if len(t) > max_doc_tokens else t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True, help="the embedding index (daisy/index/build_index.py)")
    p.add_argument("--bench", required=True, help="dir with <task>/val.bin: the held-out benchmark splits")
    p.add_argument("--out", required=True)
    p.add_argument("--edu_dir", required=True,
                   help="FineWeb-Edu score-2: EDU_DIR/data/<dump>/*.parquet or EDU_DIR/<dump>/*.parquet with columns id, "
                        "text, url, dump, int_score, score (flat, or all but id and text in a metadata struct)")
    p.add_argument("--dump", default="CC-MAIN-2023-06", help="a crawl snapshot that the pool does not contain")
    p.add_argument("--files", type=int, default=2, help="the first N parquet files (sorted) of the dump are read")
    p.add_argument("--min_int_score", type=int, default=4)
    p.add_argument("--n_tasks", type=int, default=16)
    p.add_argument("--docs_per_task", type=int, default=256)
    p.add_argument("--val_docs", type=int, default=1024, help="documents of the held-out FineWeb-Edu set")
    p.add_argument("--max_doc_tokens", type=int, default=1024, help="as the runs' --daisy_val_max_doc_tokens")
    p.add_argument("--val_docs_per_topic", type=int, default=64,
                   help="held-out FineWeb documents per topic, as the runs' --daisy_val_docs_per_topic")
    p.add_argument("--prefix", default="fwedu_t")
    p.add_argument("--tasks", default=",".join(TASKS), help="benchmarks whose val.bin items are screened against")
    p.add_argument("--lmeval", type=int, default=1,
                   help="also screen the splits only lm-evaluation-harness uses (offline Hugging Face cache)")
    p.add_argument("--ngram", type=int, default=13)
    p.add_argument("--max_items_per_ngram", type=int, default=5)
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seq_len", type=int, default=2048, help="for the report: the rows target_batches packs")
    p.add_argument("--target_tokens", type=int, default=32768, help="for the report: --daisy_target_tokens")
    a = p.parse_args()
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    need = a.val_docs + a.n_tasks * a.docs_per_task
    rep = {"args": vars(a), "stages": {}}
    st = rep["stages"]

    # 1. candidates
    files = dump_files([a.edu_dir], a.dump)[1][: a.files]
    if len(files) < a.files:
        raise FileNotFoundError(f"{a.edu_dir}, dump {a.dump}: {len(files)} parquet files, --files {a.files}")
    c, n_rows = read_candidates(files, a.min_int_score)
    st["rows_read"], st["int_score_ok"] = n_rows, len(c["id"])

    # 2. not in the index (text or URL hash), no repeats among the candidates
    th = np.fromiter((h64(t) for t in c["text"]), dtype=np.uint64, count=len(c["text"]))
    uh = np.fromiter((h64(u) for u in c["url"]), dtype=np.uint64, count=len(c["url"]))
    ith = np.load(os.path.join(a.index, "text_hash.npy"))
    iuh = np.load(os.path.join(a.index, "url_hash.npy"))
    in_t = np.isin(th, ith[ith != 0])
    in_u = np.isin(uh, iuh[iuh != 0]) & (uh != 0)
    del ith, iuh
    st["dropped_text_in_index"], st["dropped_url_in_index"] = int(in_t.sum()), int((in_u & ~in_t).sum())
    ok = ~(in_t | in_u)
    order = np.random.default_rng(a.seed).permutation(len(th))
    seen_t, seen_u, cand = set(), set(), []
    n_rep = 0
    for i in order:
        if not ok[i]:
            continue
        if int(th[i]) in seen_t or (int(uh[i]) != 0 and int(uh[i]) in seen_u):
            n_rep += 1
            continue
        seen_t.add(int(th[i]))
        seen_u.add(int(uh[i]))
        cand.append(int(i))
    st["dropped_repeats"], st["candidates"] = n_rep, len(cand)

    # 3. tokenize in the random order, screen against evaluation items, until `need` documents survive
    enc, eos = get_tokenizer(a.tokenizer)
    if a.tokenizer != "bytes" and eos != EOT:
        raise ValueError(f"tokenizer EOS {eos} != EOT {EOT}")
    strip = lambda its: [t[:-1] for t in its]   # noqa: E731  (load_items ends with EOT)
    tasks = [t for t in a.tasks.split(",") if t]
    groups = [(f"val/{t}", strip(load_items(os.path.join(a.bench, t), "val")), True) for t in tasks]
    if a.lmeval:
        import tiktoken
        groups += [(f"lmeval/{t}", v, True) for t, v in hf_items(tiktoken.get_encoding("gpt2"), print).items()]
    store = DocStore(a.index)
    hrows = np.load(os.path.join(a.index, "heldout_rows.npy"))
    htopic = np.load(os.path.join(a.index, "heldout_topic.npy"))
    fw = [np.asarray(store.doc(int(r), a.max_doc_tokens), np.uint16)
          for t in range(int(htopic.max()) + 1) for r in np.sort(hrows[htopic == t])[: a.val_docs_per_topic]]
    groups.append(("fineweb_heldout", fw, False))
    tab = Table(groups, a.ngram, a.max_items_per_ngram)
    gi_fw = tab.group_names.index("fineweb_heldout")
    rep["screen_items"] = {g: int((tab.group_of == i).sum()) for i, g in enumerate(tab.group_names)}
    st["screen_ngrams"], st["screen_ngrams_dropped_as_boilerplate"] = int(len(tab.hs)), int(tab.dropped)
    print(f"[fwedu] screen: {rep['screen_items']}, {len(tab.hs)} n-grams", flush=True)
    kept, docs = [], []
    hits = {g: 0 for g in tab.group_names}
    pos, chunk = 0, max(4096, need // 4)
    while len(kept) < need and pos < len(cand):
        idx = cand[pos: pos + chunk]
        pos += len(idx)
        toks = [cut(np.append(np.asarray(t, np.int64), EOT).astype(np.uint16), a.max_doc_tokens)
                for t in enc([c["text"][i] for i in idx])]
        d, it, k, _, _ = tab.scan(toks, a.ngram)
        g = tab.group_of[it]
        is_bad = (g != gi_fw) | (k * 2 >= np.maximum(1, tab.nng[it]))   # any eval n-gram; >= half a FineWeb doc's
        bad = set(d[is_bad].tolist())
        for i_, n_ in enumerate(tab.group_names):
            hits[n_] += len(np.unique(d[is_bad & (g == i_)]))
        for j, i in enumerate(idx):
            if j in bad:
                continue
            if len(kept) < need:
                kept.append(i)
                docs.append(toks[j])
    st["screened"] = pos
    st["dropped_eval_overlap"] = hits   # screened documents (also past the last kept one) per group
    st["kept"] = len(kept)
    if len(kept) < need:
        raise RuntimeError(f"only {len(kept)} documents survive, {need} needed: raise --files")

    # write: held-out set, then the tasks
    roles = ["heldout"] * a.val_docs + [f"{a.prefix}{k:02d}" for k in range(a.n_tasks) for _ in range(a.docs_per_task)]
    hname = f"{a.prefix}heldout"
    write_split(os.path.join(a.out, hname), docs[: a.val_docs], "val")
    back = load_items(os.path.join(a.out, hname), "val")
    if not (len(back) == a.val_docs and all(np.array_equal(x, y) for x, y in zip(back, docs[: a.val_docs]))):
        raise RuntimeError(f"{hname}: documents do not read back identically")
    rep["heldout"] = {"name": hname, "docs": a.val_docs, "tokens": int(sum(len(x) for x in docs[: a.val_docs]))}
    rep["tasks"], names = {}, []
    for k in range(a.n_tasks):
        name = f"{a.prefix}{k:02d}"
        lo = a.val_docs + k * a.docs_per_task
        dk = docs[lo: lo + a.docs_per_task]
        path = os.path.join(a.out, name)
        padded = write_split(path, dk, "train")
        back = load_items(path, "train")
        if not (len(back) == len(dk) and all(np.array_equal(x, y) for x, y in zip(back, dk))):
            raise RuntimeError(f"{name}: documents do not read back identically")
        _, _, segs = _target_segments(path, a.seq_len, a.target_tokens)   # seed 0, as target_batches
        used = np.sort(np.random.default_rng(0).permutation(len(back))[: len(segs)])
        rep["tasks"][name] = {"docs_written": len(dk), "tokens_written": int(sum(len(x) for x in dk)),
                              "padded_tokens": padded, "docs_used_at_budget": len(used),
                              "tokens_used_at_budget": int(sum(e - s for _, s, e in segs)),
                              "used_ids": [c["id"][kept[lo + int(u)]] for u in used]}
        names.append(name)
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table({
        "id": [c["id"][i] for i in kept], "url": [c["url"][i] for i in kept], "dump": [c["dump"][i] for i in kept],
        "file": [c["file"][i] for i in kept], "int_score": [c["int_score"][i] for i in kept],
        "score": [c["score"][i] for i in kept], "tokens": [len(x) for x in docs], "role": roles}),
        os.path.join(a.out, "docs.parquet"))
    sc = np.array([c["int_score"][i] for i in kept])
    rep["int_score_counts"] = {str(v): int((sc == v).sum()) for v in np.unique(sc)}
    rep["targets_flag"] = ",".join(names)
    rep["docs_used_at_budget"] = sum(v["docs_used_at_budget"] for v in rep["tasks"].values())
    rep["tokens_used_at_budget"] = sum(v["tokens_used_at_budget"] for v in rep["tasks"].values())
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    write_meta(a.out, "build", vars(a), {k: v for k, v in rep.items() if k != "tasks"})
    print(json.dumps({k: v for k, v in rep.items() if k not in ("tasks", "args")}, indent=1))


if __name__ == "__main__":
    main()
