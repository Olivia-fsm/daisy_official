#!/usr/bin/env python
"""Zero-shot accuracy of a multilingual checkpoint (a FineWeb2 run on the multilingual tokenizer) on the HELD-OUT items
of the 12 multilingual targets: INCLUDE and Global-MMLU x {fr, de, zh, es, pt, it}, with lm-evaluation-harness.

This file reuses the English evaluator (daisy/eval/lm_eval_run.py): its DaisyLM (logits of the trained Llama,
loglikelihood requests only) and load_run, and adds only what differs here.

  tokenizer  --tokenizer: the multilingual tokenizer, the pool's (a tokenizers-library tokenizer.json with a 200k
             vocabulary, </s> = 2 as end of document, no BOS).
  items      ONLY the held-out split of daisy/multilingual/build_mling_target.py (--targets: its output directory,
             TARGETS/<task>/val.items.parquet): Global-MMLU test items by sample_id, INCLUDE test items by language +
             question hash. Target items (DAiSY's labels, the similarity scorer's centroids) are never scored. The rows
             come from the raw parquet files the targets were built from (CohereLabs/Global-MMLU,
             CohereLabs/include-base-44; the Hugging Face hub cache), with the build's own validity check and same-text
             drop; the matched rows must equal the held-out items one to one (id and correct-answer text), else the
             task fails.
  formats    mc = the harness' own task template, read from the installed lm_eval at run time
                  (global_mmlu/full/<cfg>/_<cfg>_template_yaml, include/default/<Language>/_<language>_template_yaml):
                  question, options A-D, "Answer:", scored on the letter (acc). INCLUDE's per-domain description ("The
                  following is multiple-choice question about <domain>.") as in its per-domain YAMLs. One task per
                  benchmark and language over the held-out items in the harness' domains/subjects = the harness group's
                  size-weighted mean on those items.
             cf = cloze, our own variant (not a harness task): the same prompt without the option lines, scored on each
                  option's text (acc, acc_norm), as the English tasks (ARC, PIQA, ...) are; letter-choice formats of
                  MMLU-style tasks tend to sit near chance for small models.
  extra      per task and format, from the logged samples: gold and predicted option counts, acc (and acc_norm) per gold
             option and their mean over options (acc_balanced): see letter_stats.

  python daisy/multilingual/lm_eval_mling.py --run EXPS/RUN --out OUT.json --targets TARGETS/mling \\
      --tokenizer TOKENIZER/tokenizer.json [--ckpt latest|STEP] [--formats mc,cf] [--tasks ...]
  python daisy/multilingual/lm_eval_mling.py --dry --targets TARGETS/mling --tokenizer TOKENIZER/tokenizer.json [--tasks ...]
      items, checks and one rendered request per task; no model
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from daisy.multilingual.build_mling_target import LANGS, OPT, norm  # noqa: E402  (the targets' own split rules)

TASKS = ("gmmlu_fr,gmmlu_de,gmmlu_zh,gmmlu_es,gmmlu_pt,gmmlu_it,"
         "include_fr,include_de,include_zh,include_es,include_pt,include_it")
CODE = {v[0]: (k,) + v[1:] for k, v in LANGS.items()}   # fr -> (fra_Latn, French, fr)
INC_DESC = re.compile(r"^The following is multiple-choice question about (.+)\.$")


class MultilingualTokenizer:
    """The multilingual tokenizer (tokenizer.json at `path`) with the three things DaisyLM uses: encode without
    special tokens, decode, eos_token_id."""

    def __init__(self, path):
        from tokenizers import Tokenizer

        self.tk = Tokenizer.from_file(path)
        self.eos_token_id = self.tk.token_to_id("</s>")
        self.vocab_size = self.tk.get_vocab_size()

    def encode(self, s, add_special_tokens=False):
        return self.tk.encode(s, add_special_tokens=bool(add_special_tokens)).ids

    def decode(self, ids, skip_special_tokens=False):
        return self.tk.decode(list(ids), skip_special_tokens=skip_special_tokens)


def mling_lm(base):
    """DaisyLM for a 200k vocabulary: the same scores, but batches are filled up to a token budget (padded length x
    rows) instead of a fixed row count, and the log-softmax is taken only at the continuation positions (the base class
    takes it over every position in float32: [rows, length, 200,064] floats). The model still gets targets, which is
    what makes it return the logits of every position."""
    import torch
    import torch.nn.functional as F

    class MlingLM(base):
        def __init__(self, *args, tokens_per_batch=16384, **kw):
            super().__init__(*args, **kw)
            self.tokens_per_batch = int(tokens_per_batch)

        @torch.no_grad()
        def _score(self, batch):
            full = [c + k for c, k in batch]
            L = max(len(f) - 1 for f in full)
            inp = torch.zeros((len(full), L), dtype=torch.long, device=self._device)
            tgt = torch.full((len(full), L), -1, dtype=torch.long, device=self._device)
            for i, f in enumerate(full):
                ids = torch.tensor(f, dtype=torch.long, device=self._device)
                inp[i, : len(f) - 1], tgt[i, : len(f) - 1] = ids[:-1], ids[1:]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self._device.type == "cuda"):
                logits = self._net(inp, targets=tgt.clamp(min=0), get_logits=True)["logits"]
            res = []
            for i, (ctx, cont) in enumerate(batch):
                s, e = len(ctx) - 1, len(ctx) - 1 + len(cont)
                lp, gold = F.log_softmax(logits[i, s:e].float(), dim=-1), tgt[i, s:e]
                res.append((float(lp.gather(1, gold[:, None]).sum()), bool((lp.argmax(-1) == gold).all())))
            return res

        def _loglikelihood_tokens(self, requests, disable_tqdm=False, **kwargs):
            from daisy.eval.lm_eval_run import left_truncate

            reqs = list(requests)
            prep = []
            for i, (key, ctx, cont) in enumerate(reqs):
                ctx, cont = left_truncate(ctx, cont, self.max_length)
                if not ctx:
                    ctx, cont = left_truncate([self.eot_token_id], cont, self.max_length)
                prep.append((i, key, ctx, cont))
            prep.sort(key=lambda t: -(len(t[2]) + len(t[3])))   # longest first: a batch's first row sets its length
            out = [None] * len(reqs)
            b = 0
            while b < len(prep):
                n = max(1, min(self.batch_size, self.tokens_per_batch // max(1, len(prep[b][2]) + len(prep[b][3]))))
                chunk = prep[b:b + n]
                for (i, key, _, _), v in zip(chunk, self._score([(c, k) for _, _, c, k in chunk])):
                    out[i] = v
                    if key is not None and self.cache_hook is not None:
                        self.cache_hook.add_partial("loglikelihood", key, v)
                b += n
            return out

    return MlingLM


def _yaml(path):
    import yaml

    class L(yaml.SafeLoader):
        pass

    L.add_constructor("!function", lambda loader, node: loader.construct_scalar(node))
    with open(path) as f:
        return yaml.load(f, Loader=L)


def harness_task(bench, code):
    """The harness template of this benchmark and language, and the domains (INCLUDE) / subjects (Global-MMLU) its
    group covers."""
    import lm_eval

    lang, inc_cfg, gm_cfg = CODE[code]
    root = os.path.join(os.path.dirname(lm_eval.__file__), "tasks")
    if bench == "gmmlu":
        d = os.path.join(root, "global_mmlu", "full", gm_cfg)
        tmpl = _yaml(os.path.join(d, f"_{gm_cfg}_template_yaml"))
        subs = [_yaml(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.startswith(f"global_mmlu_full_{gm_cfg}_")]
        if any(s.get("description") for s in subs):
            raise ValueError(f"{d}: per-subject descriptions, not handled")
        cover = {s["task"][len(f"global_mmlu_full_{gm_cfg}_"):] for s in subs}
    else:
        d = os.path.join(root, "include", "default", inc_cfg)
        tmpl = _yaml(os.path.join(d, f"_{inc_cfg.lower()}_template_yaml"))
        subs = [_yaml(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.startswith(f"include_base_44_{inc_cfg.lower()}_")]
        cover = set()
        for s in subs:
            m = INC_DESC.match(s.get("description", ""))
            if not m:
                raise ValueError(f"{d}: unexpected description {s.get('description')!r}")
            cover.add(m.group(1))
    return tmpl, cover, d


def raw_rows(repo, path):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    return pq.read_table(hf_hub_download(repo, path, repo_type="dataset")).to_pylist()


def heldout_docs(task, targets):
    """The raw rows of the task's held-out items, with the build's checks; returns (docs, report)."""
    import pyarrow.parquet as pq

    bench, code = task.split("_")
    lang, inc_cfg, gm_cfg = CODE[code]
    d = os.path.join(targets, task)
    val = pq.read_table(os.path.join(d, "val.items.parquet"), columns=["item_id", "answer"]).to_pylist()
    seen = {norm(t) for t in pq.read_table(os.path.join(d, "train.items.parquet"), columns=["text"]).column("text").to_pylist()}
    want = Counter((r["item_id"], r["answer"]) for r in val)
    ids = {r["item_id"] for r in val}
    docs, dropped = [], 0
    if bench == "gmmlu":
        rows = raw_rows("CohereLabs/Global-MMLU", f"{gm_cfg}/test-00000-of-00001.parquet")
    else:
        rows = raw_rows("CohereLabs/include-base-44", f"{inc_cfg}/test-00000-of-00001.parquet")
    for r in rows:
        if bench == "gmmlu":   # build_mling_target.gmmlu_items
            a = str(r.get("answer", "")).strip().upper()
            if a not in ("A", "B", "C", "D") or not r.get("question") or not r.get(OPT["ABCD".index(a)]):
                continue
            iid, k = r["sample_id"], "ABCD".index(a)
        else:                  # build_mling_target.include_items
            a = r.get("answer")
            if not isinstance(a, int) or not 0 <= a <= 3 or not r.get("question") or not r.get(OPT[a]):
                continue
            iid = lang + ":" + hashlib.blake2b(norm(r["question"].strip()).encode("utf-8"), digest_size=8).hexdigest()
            k = a
        if iid not in ids:
            continue
        q, ans = r["question"].strip(), str(r[OPT[k]]).strip()
        if norm(f"{q} {ans}") in seen:   # the build's drop of held-out items with a target item's text
            dropped += 1
            continue
        r = dict(r, _item_id=iid, _answer=ans, _gold=k)
        if bench == "gmmlu":
            r["answer"] = "ABCD"[k]   # the letter as the build read it (stripped, upper case)
        docs.append(r)
    got = Counter((x["_item_id"], x["_answer"]) for x in docs)
    if got != want:
        raise ValueError(f"{task}: raw rows do not match the held-out items: {sum((want - got).values())} missing, "
                         f"{sum((got - want).values())} extra")
    rep = {"heldout_items": len(val), "dropped_same_text": dropped}
    return docs, rep


def build_tasks(task, formats, targets):
    """ConfigurableTask objects (one per format) over the task's held-out items in the harness group's coverage."""
    import datasets
    from lm_eval.api.task import ConfigurableTask

    bench, code = task.split("_")
    tmpl, cover, tdir = harness_task(bench, code)
    docs, rep = heldout_docs(task, targets)
    key = "subject" if bench == "gmmlu" else "domain"
    inside = [x for x in docs if x.get(key) in cover]
    rep.update({"harness_dir": tdir, "items": len(inside), "outside_harness": dict(Counter(x.get(key) for x in docs
                                                                                          if x.get(key) not in cover)),
                "empty_option_items": sum(any(not str(x.get(o) or "").strip() for o in OPT) for x in inside)})
    ds = datasets.Dataset.from_list(inside)
    load = lambda **kw: datasets.DatasetDict({"test": ds})   # noqa: E731
    base = {"test_split": "test", "custom_dataset": load, "output_type": tmpl["output_type"], "num_fewshot": 0,
            "metadata": {"version": 0.0}}
    if bench == "include":
        base["description"] = "The following is multiple-choice question about {{domain}}."
    out = {}
    for f in formats:
        if f == "mc":
            cfg = dict(base, task=f"mling_mc_{task}", doc_to_text=tmpl["doc_to_text"], doc_to_choice=tmpl["doc_to_choice"],
                       doc_to_target=tmpl["doc_to_target"], metric_list=tmpl["metric_list"])
        elif f == "cf":
            cfg = dict(base, task=f"mling_cf_{task}", doc_to_text="{{question.strip()}}\nAnswer:",
                       doc_to_choice=lambda doc: [str(doc[o]).strip() for o in OPT],
                       doc_to_target=lambda doc: doc["_gold"],
                       metric_list=[{"metric": m, "aggregation": "mean", "higher_is_better": True} for m in ("acc", "acc_norm")])
        else:
            raise ValueError(f"unknown format {f}")
        out[f] = ConfigurableTask(config=cfg)
    return out, rep


def letter_stats(samples):
    """Extra fields from the harness' logged samples (gold letters are unbalanced, so a letter preference can move
    acc): gold and predicted option counts (A-D order; prediction = argmax of the summed log-likelihoods, the harness'
    acc rule), acc per gold option and its unweighted mean over the options present (acc_balanced); the same for
    acc_norm when the task has it. pred_check_mismatch counts samples whose acc disagrees with argmax == gold
    (expected 0)."""
    import numpy as np

    gold = np.array([int(s["doc"]["_gold"]) for s in samples])
    pred = np.array([int(np.argmax([float(x[0]) for x in s["filtered_resps"]])) for s in samples])
    out = {"gold_counts": np.bincount(gold, minlength=4).tolist(), "pred_counts": np.bincount(pred, minlength=4).tolist(),
           "pred_check_mismatch": int(sum(float(s["acc"]) != float(p == g) for s, p, g in zip(samples, pred, gold)))}
    for k in ("acc", "acc_norm"):
        if k not in samples[0]:
            continue
        v = np.array([float(s[k]) for s in samples])
        by = [float(v[gold == j].mean()) if (gold == j).any() else None for j in range(4)]
        out[f"{k}_by_gold"] = by
        out[f"{k}_balanced"] = float(np.mean([x for x in by if x is not None]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run")
    p.add_argument("--out")
    p.add_argument("--ckpt", default="latest", help="checkpoint directory under EXP_DIR/ckpts (latest or a step)")
    p.add_argument("--tasks", default=TASKS)
    p.add_argument("--formats", default="mc,cf")
    p.add_argument("--targets", required=True, help="output directory of build_mling_target.py")
    p.add_argument("--tokenizer", required=True, help="tokenizer.json of the multilingual tokenizer")
    p.add_argument("--batch_size", type=int, default=64, help="most rows per forward")
    p.add_argument("--tokens_per_batch", type=int, default=16384, help="most padded tokens per forward")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dry", action="store_true", help="items, checks and one rendered request per task; no model")
    a = p.parse_args()
    import lm_eval

    tok = MultilingualTokenizer(a.tokenizer)
    formats = [f for f in a.formats.split(",") if f]
    tasks = [t for t in a.tasks.split(",") if t.strip()]
    git = os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip()
    if a.dry:
        for task in tasks:
            objs, rep = build_tasks(task, formats, a.targets)
            print(f"[mling eval] {task}: {json.dumps(rep, ensure_ascii=False)}", flush=True)
            for f, t in objs.items():
                t.build_all_requests(limit=1, rank=0, world_size=1)
                ins = t.instances[: len(t.instances)]
                ctx = ins[0].args[0]
                print(f"  [{f}] {len(ins)} requests for doc 0; context {len(tok.encode(ctx))} tokens; "
                      f"continuations {[i.args[1] for i in ins]}; gold {t.doc_to_target(t.test_docs()[0])}\n"
                      f"  ---\n{ctx}\n  ---", flush=True)
        return
    import torch

    from daisy.eval.lm_eval_run import DaisyLM, load_run

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, cfg, step, meta = load_run(a.run, dev, a.ckpt)
    if tok.vocab_size > cfg.vocab_size or tok.eos_token_id != 2:
        raise ValueError(f"tokenizer vocab {tok.vocab_size} / eos {tok.eos_token_id} vs model vocab {cfg.vocab_size}: "
                         "not a run on the multilingual tokenizer")
    lm = mling_lm(DaisyLM)(net, tok, cfg.sequence_length, a.batch_size, dev, tokens_per_batch=a.tokens_per_batch)
    res = {"_meta": {"run": a.run, "ckpt": a.ckpt, "step": step, "git": meta.get("git", {}).get("commit"), "eval_git": git,
                     "lm_eval": getattr(lm_eval, "__version__", "?"), "limit": a.limit or None, "num_fewshot": 0,
                     "tokenizer": a.tokenizer, "targets": a.targets, "split": "held-out (val.items.parquet)",
                     "batch_size": a.batch_size, "tokens_per_batch": a.tokens_per_batch,
                     "params": sum(x.numel() for x in net.parameters())}}
    print(f"[mling eval] {a.run} step {step}, lm_eval {res['_meta']['lm_eval']}", flush=True)
    for task in tasks:
        objs, rep = build_tasks(task, formats, a.targets)
        for f, t in objs.items():
            name = t.config.task
            t0 = time.time()
            r = lm_eval.simple_evaluate(model=lm, tasks=[t], num_fewshot=0, batch_size=a.batch_size,
                                        limit=a.limit or None, log_samples=True)
            block = r["results"][name]
            m = {k.split(",")[0]: float(v) for k, v in block.items()
                 if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))}
            m["n"] = int(r.get("n-samples", {}).get(name, {}).get("effective", -1))
            m.update(letter_stats(r["samples"][name]))
            m["seconds"] = round(time.time() - t0, 1)
            m["items"] = rep
            res[name] = m
            print(f"[mling eval] {name}: " + "  ".join(f"{k}={v:.4f}" for k, v in m.items() if k in
                                                      ("acc", "acc_norm", "acc_stderr", "acc_norm_stderr")) + f"  n={m['n']}",
                  flush=True)
            with open(a.out, "w") as fh:
                json.dump(res, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
