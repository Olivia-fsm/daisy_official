#!/usr/bin/env python
"""The FineWeb topic target: a selection target made of held-out FineWeb documents, one target task per topic, so that
the combined target (the mean of the tasks' gradients, daisy/labels/targets.py) weights the D topics equally, as the
held-out FineWeb loss does (the same number of documents per topic).

Where the documents come from. The heldout stage of daisy/index/build_index.py sets aside heldout_per_topic random
documents per topic, and every run skips all of them by text and URL hash (daisy/train/trainer.py). The held-out
FineWeb loss reads only the first --daisy_val_docs_per_topic of each topic in row order (daisy/data/stream.py,
heldout_batches; default 64). The rest of each topic (the reserve) is never trained on and never evaluated: those are
the target documents. Reserve documents whose text or URL hash equals that of an evaluated document (of any topic) are
dropped.

Format: the task format of daisy/labels/targets.py, so the trainer reads it unchanged (--daisy_targets
fineweb_t00,...,fineweb_t15 --daisy_target_dir OUT): OUT/<prefix><tt>/train.bin (uint16 GPT-2 tokens; each document
cut at --max_doc_tokens with its EOS kept, as the held-out FineWeb loss cuts them, then padded with EOS to a multiple of
1024) and train.len (padded lengths, .npy format). The whole reserve is written in row order; target_batches takes its
fixed seeded subset of --daisy_target_tokens per task from it (the same rule as for the benchmark target).
train_rows.npy holds the index row of every written document. The report (OUT/report.json, meta_build.json) gives per
topic the documents and tokens written, the duplicates dropped, and the documents target_batches uses at the given
sequence length and token budget; "targets_flag" is the value of --daisy_targets. The builder checks that every written
document reads back identically and is not evaluated.

  python daisy/targets/build_fineweb_target.py --index INDEX --out TARGETS/fineweb_topics
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from daisy.data.docstore import DocStore  # noqa: E402
from daisy.labels.targets import EOT, _target_segments, load_items  # noqa: E402
from daisy.runmeta import write_meta  # noqa: E402

PAD = 1024


def split_heldout(held, topic, val_docs_per_topic):
    """-> (evaluated rows, {topic: reserve rows}): per topic the rows in row order, the first val_docs_per_topic
    evaluated (exactly the documents stream.heldout_batches picks), the rest the reserve."""
    D = int(topic.max()) + 1
    ev, res = [], {}
    for t in range(D):
        r = np.sort(held[topic == t])
        ev.append(r[:val_docs_per_topic])
        res[t] = r[val_docs_per_topic:]
    return np.concatenate(ev), res


def write_task(path, docs):
    """The task format: train.bin (each document padded with EOS to a multiple of PAD) and train.len (padded lengths)."""
    os.makedirs(path, exist_ok=True)
    lens = np.array([-(-len(d) // PAD) * PAD for d in docs], dtype=np.int64)
    buf = np.full(int(lens.sum()), EOT, dtype=np.uint16)
    o = 0
    for d, n in zip(docs, lens):
        buf[o:o + len(d)] = d
        o += int(n)
    buf.tofile(os.path.join(path, "train.bin"))
    with open(os.path.join(path, "train.len"), "wb") as f:
        np.save(f, lens)
    return int(lens.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True, help="the embedding index (daisy/index/build_index.py)")
    p.add_argument("--out", required=True, help="one directory per topic is written below it")
    p.add_argument("--val_docs_per_topic", type=int, default=64, help="as the runs' --daisy_val_docs_per_topic")
    p.add_argument("--max_doc_tokens", type=int, default=1024, help="as the runs' --daisy_val_max_doc_tokens")
    p.add_argument("--prefix", default="fineweb_t", help="task names are <prefix><topic, two digits>")
    p.add_argument("--seq_len", type=int, default=2048, help="for the report: the rows target_batches packs")
    p.add_argument("--target_tokens", type=int, default=32768, help="for the report: --daisy_target_tokens")
    a = p.parse_args()

    held = np.load(os.path.join(a.index, "heldout_rows.npy"))
    topic = np.load(os.path.join(a.index, "heldout_topic.npy"))
    th = np.load(os.path.join(a.index, "text_hash.npy"), mmap_mode="r")
    uh = np.load(os.path.join(a.index, "url_hash.npy"), mmap_mode="r")
    store = DocStore(a.index)
    ev, res = split_heldout(held, topic, a.val_docs_per_topic)
    ev_th = set(np.asarray(th[ev]).tolist()) - {0}
    ev_uh = set(np.asarray(uh[ev]).tolist()) - {0}
    ev_set = set(ev.tolist())
    rep = {"index": a.index, "out": a.out, "D": len(res), "evaluated_docs": int(len(ev)), "topics": {}}
    names = []
    for t, rows in res.items():
        assert not (set(rows.tolist()) & ev_set)
        dup = np.array([(int(th[r]) in ev_th) or (int(uh[r]) != 0 and int(uh[r]) in ev_uh) for r in rows], dtype=bool)
        keep = rows[~dup]
        docs = [store.doc(int(r), a.max_doc_tokens) for r in keep]
        name = f"{a.prefix}{t:02d}"
        path = os.path.join(a.out, name)
        padded = write_task(path, docs)
        np.save(os.path.join(path, "train_rows.npy"), keep)
        back = load_items(path, "train")   # read back exactly as the trainer does
        same = len(back) == len(docs) and all(np.array_equal(x, y) for x, y in zip(back, docs))
        if not same:
            raise RuntimeError(f"{name}: documents do not read back identically")
        _, _, segs = _target_segments(path, a.seq_len, a.target_tokens)   # seed 0, as target_batches
        used = np.sort(np.random.default_rng(0).permutation(len(back))[:len(segs)])   # its items, in its order rule
        rep["topics"][name] = {
            "topic": t, "reserve_docs": int(len(rows)), "dropped_duplicates_of_evaluated": int(dup.sum()),
            "docs_written": int(len(keep)), "tokens_written": int(sum(len(d) for d in docs)), "padded_tokens": padded,
            "docs_used_at_budget": len(used), "tokens_used_at_budget": int(sum(e - s for _, s, e in segs)),
            "used_rows": keep[used].tolist()}
        names.append(name)
        print(f"{name}: {len(rows)} reserve docs, {int(dup.sum())} duplicates of evaluated docs dropped, {len(keep)} "
              f"written ({rep['topics'][name]['tokens_written']} tokens); at seq {a.seq_len} and {a.target_tokens} "
              f"target tokens the run uses {len(used)} docs", flush=True)
    rep["targets_flag"] = ",".join(names)
    rep["docs_used_at_budget"] = sum(v["docs_used_at_budget"] for v in rep["topics"].values())
    rep["tokens_used_at_budget"] = sum(v["tokens_used_at_budget"] for v in rep["topics"].values())
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    write_meta(a.out, "build", vars(a), {k: v for k, v in rep.items() if k != "topics"})
    print(json.dumps({k: v for k, v in rep.items() if k != "topics"}, indent=1))


if __name__ == "__main__":
    main()
