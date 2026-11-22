#!/usr/bin/env python
"""Multilingual FineWeb-Edu target (the counterpart of the English FineWeb-Edu target of
daisy/targets/build_fwedu_target.py): documents of the FineWeb2 pool itself with a high multilingual FineWeb-Edu score
(mfwedu, the bytes of fw2_edu_scores.py: floor(10 * mfwedu)), per language, as 2 target tasks per language (12 tasks,
as many as the multilingual benchmark target has) and a held-out set per language for a held-out FineWeb-Edu loss.

The pool is all of FineWeb2 in these languages, so (unlike English) the documents cannot come from a crawl outside it:
  1. Source shards: the TAIL of every rank's shard list (positions >= --tail_from in perm(seed)[rank::world], the
     stream's shard order, daisy/data/pool.py), and none that a run under --check_runs opened (its checkpoint's stream
     state: the first next_shard shards of its list; runs that started a second pass are reported and skipped). A run
     opens the shards of its list in order, and the number it reaches grows with the window factor lambda and the
     number of steps, so runs that stay below --tail_from never train on a tail shard; --check_runs verifies this for
     runs that exist already. For a quality-filtered uniform run (--daisy_pool_keep at the cutoffs of --keep_check),
     report.json "keep_projection" gives the share of every rank's kept tokens that --iters steps consume.
  2. Candidates: documents of those shards with byte >= --min_byte, per language, in a seeded random order. Taken in
     that order while they pass: text and URL hash not those of an index held-out FineWeb2 document (--index) and not
     those of an already taken document; no token --n-gram shared with any held-out benchmark item (--bench / --tasks:
     the INCLUDE and Global-MMLU val splits; an item's first token, which may tokenize differently inside a web
     document, and its </s> are dropped, and n-grams in more than --max_items_per_ngram items are ignored as stock
     phrases); fewer than --near_dup_frac of its n-grams in the held-out FineWeb2 documents.
  3. The first --heldout_docs taken per language are the held-out set (fw2edu_<lang>_heldout/val.*), the next
     2 x --docs_per_task the two target tasks (fw2edu_<lang>_t0, _t1 /train.*). Documents are cut at --max_doc_tokens
     (</s> kept), as the English target. train.emb.npy = the documents' stored phi de-standardized with the pool's
     stats.npz (similarity_mling.py standardizes it back: the exact vectors the selector scores the pool with).
  4. Copies: every pool shard's sidecar text/URL hashes are scanned for the chosen documents' hashes. exclude/
     text_hash.npy, url_hash.npy (sorted unique; --daisy_pool_exclude) remove every copy from a run's training;
     fw2edu_<lang>_heldout_nocopy = the held-out documents with no text or URL copy anywhere else in the pool (the clean
     set for runs trained without the exclusion).
--prefix (default fw2edu) names the tasks <prefix>_<lang>_<role>. --min_byte 0 takes every document: a target of
unfiltered FineWeb2 documents. --avoid DIR[,DIR]: documents whose text or URL hash is in another target's exclude/ are
not taken (so the two targets' documents are disjoint), and exclude_all/ = this target's exclude/ plus those hashes
(one --daisy_pool_exclude for runs that must skip both).
docs.parquet lists every written document (key, shard, row, lang, role, mfwedu, n_tokens, hashes, copies); every task is
read back with daisy/labels/targets.py load_items and checked equal to what was written.

Inputs: --pool the tokenized multilingual pool (fw2_tokenize.py: uint32 tokens, sidecars with text_hash, url_hash,
mfwedu, url; stats.npz); --scores the output of fw2_edu_scores.py; --index the held-out FineWeb2 index
(build_mling_index.py); --bench the multilingual benchmark targets (build_mling_target.py); --stream_seed, --world the
stream seed and number of GPUs of the runs (they fix the shard lists); --tokenizer (optional) the multilingual
tokenizer's tokenizer.json, used only to decode two example documents per language into report.json.

  python daisy/multilingual/build_fw2edu_target.py --pool DATAPATH/fineweb2-tokenized \\
      --scores DATAPATH/fineweb2-edu-scores --index INDEX --bench TARGETS/mling --out TARGETS/fw2edu \\
      --stream_seed SEED [--check_runs EXPS/RUN_A,EXPS/RUN_B]
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from daisy.data.pool import DIM, DsPool  # noqa: E402
from daisy.labels.targets import load_items, task_format  # noqa: E402
from daisy.runmeta import write_meta  # noqa: E402
from daisy.targets.build_fwedu_target import ngram_hashes  # noqa: E402

CODES = {"fra_Latn": "fr", "deu_Latn": "de", "cmn_Hani": "zh", "spa_Latn": "es", "por_Latn": "pt", "ita_Latn": "it"}
BENCH_TASKS = ",".join(f"{b}_{c}" for b in ("gmmlu", "include") for c in ("fr", "de", "zh", "es", "pt", "it"))


def rank_lists(n_shards, seed, world):
    """Every rank's shard list, as the pool stream computes it (daisy/data/pool.py ShardDocStream)."""
    perm = np.random.default_rng(seed).permutation(n_shards)
    return [perm[r::world] for r in range(world)]


def opened_shards(run, lists):
    """-> (set of shards the run's latest checkpoint has opened, description); None if a rank is past pass 0."""
    ck = os.path.join(run, "ckpts")
    steps = sorted(int(d) for d in os.listdir(ck) if d.isdigit())
    src = os.path.join(ck, "latest") if os.path.exists(os.path.join(ck, "latest", "stream_r0.json")) else \
        os.path.join(ck, str(steps[-1]))
    out, desc = set(), []
    for r, L in enumerate(lists):
        with open(os.path.join(src, f"stream_r{r}.json")) as f:
            st = json.load(f)
        st = st.get("base", st)        # a selecting run's state wraps the pool stream's (daisy/data/select.py)
        if "next_shard" not in st:
            raise ValueError(f"{run} rank {r}: no pool stream state (keys {sorted(st)})")
        if int(st.get("pass", 0)) > 0 or int(st.get("epoch", 0)) > 0:
            return None, f"{run}: rank {r} past pass 0"
        k = int(st["next_shard"])
        op = {int(s) for s, _ in st["open"]}
        if not op <= set(int(x) for x in L[:k]):
            raise ValueError(f"{run} rank {r}: open shards not in the first next_shard={k} of the computed list "
                             f"(wrong seed or world?)")
        out |= set(int(x) for x in L[:k])
        desc.append(k)
    return out, f"{os.path.basename(run)} ({src.split('/')[-1]}): next_shard {desc}"


def bench_table(bench, tasks, n, max_items):
    """Sorted unique n-gram hashes of the held-out (val) items of the benchmark tasks (first token and </s> dropped),
    without the n-grams that more than max_items items share. -> (hashes, n-grams dropped as common, items)."""
    hs = []
    for t in tasks:
        _, eos = task_format(os.path.join(bench, t))
        for it in load_items(os.path.join(bench, t), "val"):
            body = np.asarray(it)
            body = body[1:-1] if len(body) and body[-1] == eos else body[1:]
            h = ngram_hashes(body, n)
            if len(h):
                hs.append(np.unique(h))
    H = np.concatenate(hs)
    keys, cnt = np.unique(H, return_counts=True)
    return keys[cnt <= max_items], int((cnt > max_items).sum()), len(hs)


def isin_sorted(h, keys):
    if not len(h) or not len(keys):
        return np.zeros(len(h), bool)
    i = np.searchsorted(keys, h)
    i[i == len(keys)] = 0
    return keys[i] == h


def cap(t, m):
    return np.concatenate([t[: m - 1], t[-1:]]) if len(t) > m else t


def write_task(d, split, docs, eos, vocab, extra):
    os.makedirs(d, exist_ok=True)
    lens = np.array([len(x) for x in docs], dtype=np.int64)
    np.concatenate([np.asarray(x, dtype=np.uint32) for x in docs]).tofile(os.path.join(d, f"{split}.bin"))
    with open(os.path.join(d, f"{split}.len"), "wb") as f:
        np.save(f, lens)
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"task": os.path.basename(d), "dtype": "uint32", "eos_id": int(eos), "padded": False, "vocab": vocab,
                   **extra}, f, indent=1)
    back = load_items(d, split)
    if len(back) != len(docs) or any(not np.array_equal(a, b) for a, b in zip(back, docs)):
        raise SystemExit(f"{d}/{split}: read-back differs from what was written")
    return int(lens.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True, help="the tokenized multilingual pool (fw2_tokenize.py)")
    p.add_argument("--scores", required=True, help="fw2_edu_scores.py output (one byte per document)")
    p.add_argument("--index", required=True, help="the runs' index (held-out FineWeb2 documents)")
    p.add_argument("--bench", required=True, help="the benchmark target tasks (build_mling_target.py)")
    p.add_argument("--tasks", default=BENCH_TASKS)
    p.add_argument("--out", required=True)
    p.add_argument("--min_byte", type=int, default=30, help="30 = raw mfwedu >= 3.0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stream_seed", type=int, required=True, help="the runs' stream seed (stream_r*.json 'seed')")
    p.add_argument("--world", type=int, default=4)
    p.add_argument("--tail_from", type=int, default=300)
    p.add_argument("--check_runs", default="")
    p.add_argument("--heldout_docs", type=int, default=1024)
    p.add_argument("--docs_per_task", type=int, default=256)
    p.add_argument("--tasks_per_lang", type=int, default=2)
    p.add_argument("--max_doc_tokens", type=int, default=1024)
    p.add_argument("--n", type=int, default=13)
    p.add_argument("--max_items_per_ngram", type=int, default=5)
    p.add_argument("--near_dup_frac", type=float, default=0.5)
    p.add_argument("--threads", type=int, default=48)
    p.add_argument("--keep_check", default="25,30", help="bytes K: capped tokens per rank at byte >= K (report)")
    p.add_argument("--train_doc_cap", type=int, default=2048)
    p.add_argument("--iters", type=int, default=256000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--prefix", default="fw2edu", help="task names <prefix>_<lang>_<role>")
    p.add_argument("--avoid", default="", help="comma list of exclude dirs (text_hash.npy, url_hash.npy) of documents "
                                               "not to take; exclude_all/ = exclude/ plus these")
    p.add_argument("--tokenizer", default="", help="tokenizer.json of the multilingual tokenizer, to decode example "
                                                   "documents into report.json ('' = no examples)")
    a = p.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq

    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    pool = DsPool(a.pool)
    langs = pool.shard_langs()
    eos = int(task_format(os.path.join(a.bench, a.tasks.split(",")[0]))[1])
    st = np.load(os.path.join(a.pool, "stats.npz"))
    mean, std = st["mean"].astype(np.float32), st["std"].astype(np.float32)
    rep = {"args": vars(a), "langs": {}}

    # 1. source shards
    lists = rank_lists(pool.n_shards, a.stream_seed, a.world)
    pos = {int(s): (r, k) for r, L in enumerate(lists) for k, s in enumerate(L)}
    tail = {s for s, (r, k) in pos.items() if k >= a.tail_from}
    opened, checks = set(), []
    for run in [x for x in a.check_runs.split(",") if x]:
        op, desc = opened_shards(run, lists)
        checks.append(desc)
        if op is not None:
            opened |= op
    src = sorted(tail - opened)
    rep["source"] = {"rank_list_lengths": [len(L) for L in lists], "tail_shards": len(tail),
                     "tail_opened_by_checked_runs": len(tail & opened), "source_shards": len(src), "checks": checks}
    print(f"[fw2edu] source: {len(src)} tail shards (positions >= {a.tail_from}); checked runs: {checks}", flush=True)

    # what a filtered-uniform run (--daisy_pool_keep SCORES --daisy_pool_keep_min K) can draw per rank: the
    # capped tokens (documents cut at --train_doc_cap, as the stream does) of its rank's shards with byte >= K, against
    # the tokens one rank's stream consumes in --iters steps (batch x (seq + 1) per step: packed with carry-over)
    keeps = [int(x) for x in a.keep_check.split(",") if x]

    def avail(s):
        _, ends, _, starts = pool.shard(s)
        L = np.minimum(ends - starts, a.train_doc_cap)
        u8 = np.fromfile(os.path.join(a.scores, f"{s:05d}.u8"), dtype=np.uint8)
        return s, {K: (int(L[u8 >= K].sum()), int((u8 >= K).sum())) for K in keeps}
    if keeps:
        per = {}
        with ThreadPoolExecutor(a.threads) as ex:
            for s, d in ex.map(avail, range(pool.n_shards)):
                per[s] = d
        need_rank = a.iters * a.batch * (a.seq + 1)
        rep["keep_projection"] = {"tokens_per_rank_needed": need_rank, "train_doc_cap": a.train_doc_cap}
        for K in keeps:
            av = [sum(per[int(s)][K][0] for s in L) for L in lists]
            rep["keep_projection"][f">={K / 10:.1f}"] = {
                "capped_tokens_per_rank": av, "docs_per_rank": [sum(per[int(s)][K][1] for s in L) for L in lists],
                "fraction_used_per_rank": [round(need_rank / max(v, 1), 4) for v in av]}
            print(f"[fw2edu] keep >= {K / 10:.1f}: capped tokens per rank {[f'{v / 1e9:.2f}B' for v in av]}, "
                  f"needed {need_rank / 1e9:.2f}B -> used {[round(need_rank / max(v, 1), 3) for v in av]}", flush=True)

    # 2. candidates per language (byte >= min_byte), seeded random order
    def cands(s):
        u8 = np.fromfile(os.path.join(a.scores, f"{s:05d}.u8"), dtype=np.uint8)
        if len(u8) != len(pool.shard(s)[1]):
            raise ValueError(f"shard {s}: {len(u8)} score bytes")
        return s, np.flatnonzero(u8 >= a.min_byte)
    by_lang = {}
    with ThreadPoolExecutor(a.threads) as ex:
        for s, rows in ex.map(cands, src):
            by_lang.setdefault(langs[s], []).append((np.uint64(s) << np.uint64(32)) | rows.astype(np.uint64))
    held = np.load(os.path.join(a.index, "heldout_rows.npy"))
    ih_text = np.unique(np.load(os.path.join(a.index, "text_hash.npy"))[held])
    ih_url = np.unique(np.load(os.path.join(a.index, "url_hash.npy"))[held])
    ih_url = ih_url[ih_url != 0]
    av_text, av_url = np.zeros(0, np.uint64), np.zeros(0, np.uint64)
    for xd_ in [x for x in a.avoid.split(",") if x]:
        av_text = np.concatenate([av_text, np.load(os.path.join(xd_, "text_hash.npy")).astype(np.uint64)])
        av_url = np.concatenate([av_url, np.load(os.path.join(xd_, "url_hash.npy")).astype(np.uint64)])
    av_text, av_url = np.unique(av_text), np.unique(av_url)
    av_url = av_url[av_url != 0]
    rep["avoid"] = {"dirs": a.avoid, "text_hashes": len(av_text), "url_hashes": len(av_url)}
    bkeys, bdrop, bitems = bench_table(a.bench, a.tasks.split(","), a.n, a.max_items_per_ngram)
    # held-out FineWeb2 documents' n-grams (near-duplicates)
    itok = np.memmap(os.path.join(a.index, "tokens.u32"), dtype=np.uint32, mode="r")
    ioff = np.load(os.path.join(a.index, "offsets.i64.npy"))
    fkeys = np.unique(np.concatenate([ngram_hashes(itok[ioff[i]:ioff[i + 1]], a.n) for i in held]))
    rep["screen"] = {"bench_items": bitems, "bench_ngrams": len(bkeys), "bench_ngrams_dropped_common": bdrop,
                     "heldout_fineweb2_docs": len(held), "heldout_fineweb2_ngrams": len(fkeys)}
    print(f"[fw2edu] screen: {bitems} benchmark items, {len(bkeys):,} n-grams; {len(fkeys):,} held-out FineWeb2 n-grams",
          flush=True)

    need = a.heldout_docs + a.tasks_per_lang * a.docs_per_task

    def fetch(keys):
        """-> {key: (text_hash, url_hash, mfwedu, url, tokens)}: one sidecar read per shard, only the rows needed."""
        by = {}
        for k in keys:
            by.setdefault(int(k) >> 32, []).append(int(k) & 0xFFFFFFFF)

        def one(s):
            t = pq.read_table(os.path.join(a.pool, "sidecar", f"{s:05d}.meta.parquet"),
                              columns=["text_hash", "url_hash", "mfwedu", "url"])
            ix = np.array(by[s], dtype=np.int64)
            c = {n: t.column(n).take(ix).to_pylist() for n in t.column_names}
            return {(s << 32) | int(i): (int(c["text_hash"][j]), int(c["url_hash"][j]), float(c["mfwedu"][j]),
                                         str(c["url"][j]), np.asarray(pool.doc((s << 32) | int(i))))
                    for j, i in enumerate(ix)}
        out = {}
        with ThreadPoolExecutor(a.threads) as ex:
            for d in ex.map(one, sorted(by)):
                out.update(d)
        return out

    rows, written = [], {}
    for L in sorted(by_lang):
        code = CODES[L]
        keys = np.concatenate(by_lang[L])
        keys = keys[np.random.default_rng([a.seed, sum(map(ord, L))]).permutation(len(keys))]
        cnt = {"candidates": int(len(keys)), "index_heldout_hash": 0, "avoid_hash": 0, "repeat_hash": 0,
               "bench_ngram": 0, "near_dup_heldout_fineweb2": 0}
        took, seen_t, seen_u = [], set(), set()
        done = 0
        while len(took) < need and done < len(keys):
            chunk = keys[done: done + 3 * need]          # in the seeded order; fetched together, screened in order
            done += len(chunk)
            got = fetch(chunk)
            for key in chunk:
                if len(took) >= need:
                    break
                key = int(key)
                s, i = key >> 32, key & 0xFFFFFFFF
                th, uh, sc, url, t = got[key]
                if isin_sorted(np.array([th], np.uint64), ih_text)[0] or \
                        (uh != 0 and isin_sorted(np.array([uh], np.uint64), ih_url)[0]):
                    cnt["index_heldout_hash"] += 1
                    continue
                if len(av_text) and (isin_sorted(np.array([th], np.uint64), av_text)[0] or
                                     (uh != 0 and isin_sorted(np.array([uh], np.uint64), av_url)[0])):
                    cnt["avoid_hash"] += 1
                    continue
                if th in seen_t or (uh != 0 and uh in seen_u):
                    cnt["repeat_hash"] += 1
                    continue
                body = t[1:-1] if len(t) and t[-1] == eos else t[1:]
                if isin_sorted(ngram_hashes(body, a.n), bkeys).any():
                    cnt["bench_ngram"] += 1
                    continue
                h = ngram_hashes(t, a.n)
                if len(h) and isin_sorted(np.unique(h), fkeys).mean() >= a.near_dup_frac:
                    cnt["near_dup_heldout_fineweb2"] += 1
                    continue
                seen_t.add(th)
                if uh != 0:
                    seen_u.add(uh)
                took.append((key, s, i, cap(t, a.max_doc_tokens), th, uh, sc, url))
        if len(took) < need:
            raise SystemExit(f"{L}: only {len(took)} documents pass, {need} needed ({cnt})")
        cnt["screened"] = sum(v for k, v in cnt.items() if k != "candidates") + len(took)
        roles = ["heldout"] * a.heldout_docs + [f"t{j}" for j in range(a.tasks_per_lang) for _ in range(a.docs_per_task)]
        for (key, s, i, t, th, uh, sc, url), role in zip(took, roles):
            rows.append({"key": key, "shard": s, "row": i, "lang": L, "code": code, "role": role, "mfwedu": sc,
                         "n_tokens": len(t), "text_hash": th, "url_hash": uh, "url": url,
                         "rank": pos[s][0], "list_pos": pos[s][1]})
        written[L] = (took, roles)
        rep["langs"][L] = cnt
        print(f"[fw2edu] {L}: {cnt}", flush=True)

    # 4. copies of the chosen documents anywhere in the pool (sidecar hashes of every shard)
    ct = np.array(sorted({r["text_hash"] for r in rows}), dtype=np.uint64)
    cu = np.array(sorted({r["url_hash"] for r in rows if r["url_hash"] != 0}), dtype=np.uint64)
    own = {r["key"] for r in rows}

    def scan(s):
        th, uh = pool.hashes(s)
        ht = np.flatnonzero(isin_sorted(th.astype(np.uint64), ct))
        hu = np.flatnonzero(isin_sorted(uh.astype(np.uint64), cu) & (uh != 0))
        return s, [(int(th[j]), (s << 32) | int(j)) for j in ht], [(int(uh[j]), (s << 32) | int(j)) for j in hu]
    tcopies, ucopies = {}, {}
    with ThreadPoolExecutor(a.threads) as ex:
        for s, tl, ul in ex.map(scan, range(pool.n_shards)):
            for h, k in tl:
                if k not in own:
                    tcopies.setdefault(h, []).append(k)
            for h, k in ul:
                if k not in own:
                    ucopies.setdefault(h, []).append(k)
    for r in rows:
        r["text_copies"] = len(tcopies.get(r["text_hash"], []))
        r["url_copies"] = len(ucopies.get(r["url_hash"], [])) if r["url_hash"] else 0
    copy_keys = sorted({k for v in list(tcopies.values()) + list(ucopies.values()) for k in v})
    rep["copies"] = {"docs_with_text_copy": sum(r["text_copies"] > 0 for r in rows),
                     "docs_with_url_copy_only": sum(r["text_copies"] == 0 and r["url_copies"] > 0 for r in rows),
                     "copy_documents_in_pool": len(copy_keys),
                     "copy_documents_in_opened_shards": sum(1 for k in copy_keys if (k >> 32) in opened)}

    # 5. write
    vocab = 200064      # the multilingual tokenizer's vocabulary size (recorded in every meta.json)
    xd = os.path.join(a.out, "exclude")
    os.makedirs(xd, exist_ok=True)
    np.save(os.path.join(xd, "text_hash.npy"), ct)
    np.save(os.path.join(xd, "url_hash.npy"), cu)
    rep["exclude"] = {"text_hashes": len(ct), "url_hashes": len(cu)}
    if a.avoid:
        xa = os.path.join(a.out, "exclude_all")
        os.makedirs(xa, exist_ok=True)
        ut, uu = np.union1d(ct, av_text).astype(np.uint64), np.union1d(cu, av_url).astype(np.uint64)
        np.save(os.path.join(xa, "text_hash.npy"), ut)
        np.save(os.path.join(xa, "url_hash.npy"), uu)
        rep["exclude_all"] = {"text_hashes": len(ut), "url_hashes": len(uu), "from": ["exclude"] + a.avoid.split(",")}
    bykey = {r["key"]: r for r in rows}
    tasks_out = {}
    for L, (took, roles) in written.items():
        code = CODES[L]
        for role in ["heldout"] + [f"t{j}" for j in range(a.tasks_per_lang)]:
            sel = [x for x, rr in zip(took, roles) if rr == role]
            docs = [x[3] for x in sel]
            name = f"{a.prefix}_{code}_{role}"
            extra = {"lang": L, "role": role, "min_mfwedu": a.min_byte / 10, "max_doc_tokens": a.max_doc_tokens,
                     "source": "build_fw2edu_target.py"}
            if role == "heldout":
                tok = write_task(os.path.join(a.out, name), "val", docs, eos, vocab, extra)
                nc = [x[3] for x in sel if bykey[x[0]]["text_copies"] == 0 and bykey[x[0]]["url_copies"] == 0]
                tok_nc = write_task(os.path.join(a.out, name + "_nocopy"), "val", nc, eos, vocab,
                                    extra | {"nocopy": True})
                tasks_out[name] = {"docs": len(docs), "tokens": tok}
                tasks_out[name + "_nocopy"] = {"docs": len(nc), "tokens": tok_nc}
            else:
                tok = write_task(os.path.join(a.out, name), "train", docs, eos, vocab, extra)
                phi = np.stack([np.asarray(pool.shard(x[1])[2][x[2]], dtype=np.float32) for x in sel])
                np.save(os.path.join(a.out, name, "train.emb.npy"), phi * std + mean)
                tasks_out[name] = {"docs": len(docs), "tokens": tok}
    rep["tasks"] = tasks_out
    rep["target_tasks"] = ",".join(sorted(k for k in tasks_out if "_t" in k[-3:]))
    u64 = {"text_hash", "url_hash"}        # 64-bit hashes: above int64's range
    pq.write_table(pa.table({k: pa.array([r[k] for r in rows], type=pa.uint64() if k in u64 else None) for k in rows[0]}),
                   os.path.join(a.out, "docs.parquet"))
    try:
        if not a.tokenizer:
            raise ValueError("no --tokenizer")
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(a.tokenizer)
        rep["examples"] = {L: [tk.decode([int(v) for v in x[3][:120]]) for x in took[a.heldout_docs:a.heldout_docs + 2]]
                           for L, (took, _) in written.items()}
    except Exception as e:      # examples are for reading only
        rep["examples"] = f"not decoded: {e!r}"
    rep["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    write_meta(a.out, "build_fw2edu_target", vars(a))
    print(json.dumps({k: rep[k] for k in ("source", "screen", "copies", "exclude", "tasks", "target_tasks")}, indent=1),
          flush=True)
    print(f"[fw2edu] done in {rep['seconds']} s -> {a.out}")


if __name__ == "__main__":
    main()
