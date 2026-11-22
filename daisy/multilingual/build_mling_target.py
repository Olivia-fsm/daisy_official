#!/usr/bin/env python
"""Multilingual selection targets: INCLUDE (CohereLabs/include-base-44) and Global-MMLU (CohereLabs/Global-MMLU) in the
six languages of the multilingual pool (French, German, Chinese, Spanish, Portuguese, Italian) -> 12 tasks, one
directory each (OUT/<bench>_<lang>/), in the layout of the target tasks (daisy/labels/targets.py): train.bin /
train.len = the target items, val.bin / val.len = the held-out items.

Item text: "{question} {correct answer}" (as the English benchmark targets), tokenized with the multilingual tokenizer
(--tokenizer: a tokenizers-library tokenizer.json with a 200k vocabulary, </s> = 2 as end of document, no BOS), </s>
appended; tokens uint32, NOT padded (.len = each item's true length, </s> included; meta.json says so).

Splits (no benchmark here has a training split):
  Global-MMLU: the items are translations of the same questions, sharing `sample_id` across languages. The split is
     made on sample_id, the SAME for all languages, so no held-out question has a translation among the target items:
     dev -> target; test -> target if blake2b(sample_id) is even, else held-out.
  INCLUDE: separate exam questions per language. validation -> target; test -> target if blake2b(lang + question) is
     even, else held-out (identical questions land on the same side).
  Then any held-out item whose normalized text equals a target item's of the same task is dropped (counted).
The held-out items are not compared with the training pool here.

Embeddings: mmBERT-base (jhu-clsp/mmBERT-base), bf16, mean of the last hidden states over all tokens (special ones
included) of the item, up to 8192 tokens = the procedure behind the pool's stored vectors (fw2_tokenize.py). Raw
float32 vectors in {split}.emb.npy (standardize with the pool's stats.npz). Needs a GPU.
Also written per task: {split}.items.parquet (item_id, source_split, split, subject, question, answer, text, n_tokens;
read by lm_eval_mling.py) and meta.json; OUT/report.json (counts per task, and the number of Global-MMLU sample_ids
that are on both sides, which must be 0).

  python daisy/multilingual/build_mling_target.py --out TARGETS/mling --tokenizer TOKENIZER/tokenizer.json
"""

import argparse
import hashlib
import json
import os
import sys
import time
import unicodedata

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# pool language -> (short code, INCLUDE config, Global-MMLU config)
LANGS = {"fra_Latn": ("fr", "French", "fr"), "deu_Latn": ("de", "German", "de"), "cmn_Hani": ("zh", "Chinese", "zh"),
         "spa_Latn": ("es", "Spanish", "es"), "por_Latn": ("pt", "Portuguese", "pt"), "ita_Latn": ("it", "Italian", "it")}
OPT = ("option_a", "option_b", "option_c", "option_d")


def even(s):
    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()[0] % 2 == 0


def norm(t):
    return " ".join(unicodedata.normalize("NFKC", t).lower().split())


def read_parquet(repo, path):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    return pq.read_table(hf_hub_download(repo, path, repo_type="dataset")).to_pylist()


def gmmlu_items(cfg):
    out, bad = [], 0
    for src in ("dev", "test"):
        for r in read_parquet("CohereLabs/Global-MMLU", f"{cfg}/{src}-00000-of-00001.parquet"):
            a = str(r.get("answer", "")).strip().upper()
            if a not in ("A", "B", "C", "D") or not r.get("question") or not r.get(OPT["ABCD".index(a)]):
                bad += 1
                continue
            ans = str(r[OPT["ABCD".index(a)]]).strip()
            split = "train" if src == "dev" or even(r["sample_id"]) else "val"
            out.append({"item_id": r["sample_id"], "source_split": src, "split": split, "subject": r.get("subject", ""),
                        "question": r["question"].strip(), "answer": ans, "text": f"{r['question'].strip()} {ans}"})
    return out, bad


def include_items(cfg, lang):
    out, bad = [], 0
    for src in ("validation", "test"):
        for r in read_parquet("CohereLabs/include-base-44", f"{cfg}/{src}-00000-of-00001.parquet"):
            a = r.get("answer")
            if not isinstance(a, int) or not 0 <= a <= 3 or not r.get("question") or not r.get(OPT[a]):
                bad += 1
                continue
            q, ans = r["question"].strip(), str(r[OPT[a]]).strip()
            iid = lang + ":" + hashlib.blake2b(norm(q).encode("utf-8"), digest_size=8).hexdigest()
            split = "train" if src == "validation" or even(lang + norm(q)) else "val"
            out.append({"item_id": iid, "source_split": src, "split": split, "subject": r.get("subject", ""),
                        "question": q, "answer": ans, "text": f"{q} {ans}"})
    return out, bad


def mmbert_embed(texts, tok, model, bs=64):
    import torch

    out = []
    order = np.argsort([-len(t) for t in texts])  # long first: less padding per batch
    for i in range(0, len(texts), bs):
        ix = order[i:i + bs]
        enc = tok([texts[j] for j in ix], truncation=True, max_length=8192, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            h = model(**enc).last_hidden_state.float()
        m = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    emb = np.empty((len(texts), 768), dtype=np.float32)
    emb[order] = np.concatenate(out)
    return emb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", required=True, help="tokenizer.json of the multilingual tokenizer")
    p.add_argument("--langs", default=",".join(LANGS))
    a = p.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoModel, AutoTokenizer

    t0 = time.time()
    tk = Tokenizer.from_file(a.tokenizer)
    eos = tk.token_to_id("</s>")
    mtok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
    mmod = AutoModel.from_pretrained("jhu-clsp/mmBERT-base", dtype=torch.bfloat16).cuda().eval()
    rep = {"tokenizer": a.tokenizer, "vocab": tk.get_vocab_size(), "eos_id": eos, "tasks": {},
           "git": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip()}
    for lang in a.langs.split(","):
        code, inc_cfg, gm_cfg = LANGS[lang]
        for bench, (items, bad) in (("gmmlu", gmmlu_items(gm_cfg)), ("include", include_items(inc_cfg, lang))):
            task = f"{bench}_{code}"
            d = os.path.join(a.out, task)
            os.makedirs(d, exist_ok=True)
            seen = {norm(it["text"]) for it in items if it["split"] == "train"}
            dropped = [it for it in items if it["split"] == "val" and norm(it["text"]) in seen]
            items = [it for it in items if not (it["split"] == "val" and norm(it["text"]) in seen)]
            T = {"lang": lang, "bench": bench, "bad_rows": bad, "val_dropped_same_text_as_target": len(dropped)}
            for split in ("train", "val"):
                its = [it for it in items if it["split"] == split]
                ids = [tk.encode(it["text"], add_special_tokens=False).ids + [eos] for it in its]
                lens = np.array([len(x) for x in ids], dtype=np.int64)
                np.concatenate([np.asarray(x, dtype=np.uint32) for x in ids]).tofile(os.path.join(d, f"{split}.bin"))
                with open(os.path.join(d, f"{split}.len"), "wb") as f:
                    np.save(f, lens)
                emb = mmbert_embed([it["text"] for it in its], mtok, mmod)
                np.save(os.path.join(d, f"{split}.emb.npy"), emb)
                pq.write_table(pa.table({k: [it[k] for it in its] for k in its[0]} | {"n_tokens": lens}),
                               os.path.join(d, f"{split}.items.parquet"))
                T[split] = {"items": len(its), "tokens": int(lens.sum()), "tokens_per_item_median": float(np.median(lens)),
                            "by_source": {s: sum(it["source_split"] == s for it in its) for s in sorted({it["source_split"] for it in its})}}
            if bench == "gmmlu":
                T["val_sample_ids"] = len({it["item_id"] for it in items if it["split"] == "val"})
            T["example"] = items[0]["text"][:300]
            with open(os.path.join(d, "meta.json"), "w") as f:
                json.dump({"task": task, "dtype": "uint32", "eos_id": eos, "padded": False, "vocab": tk.get_vocab_size(),
                           "tokenizer": a.tokenizer, "embedding": "mmBERT-base bf16 mean over all tokens, <= 8192",
                           **{k: v for k, v in T.items() if k != "example"}}, f, indent=1)
            rep["tasks"][task] = T
            print(f"[mling target] {task}: {json.dumps(T, ensure_ascii=False)[:900]}", flush=True)
    # cross-language check for Global-MMLU: the same sample_id must be on the same side in every language
    sides = {}
    for task in rep["tasks"]:
        if task.startswith("gmmlu_"):
            for split in ("train", "val"):
                for iid in pq.read_table(os.path.join(a.out, task, f"{split}.items.parquet"), columns=["item_id"]).column("item_id").to_pylist():
                    sides.setdefault(iid, set()).add(split)
    rep["gmmlu_sample_ids_on_both_sides"] = sum(len(s) > 1 for s in sides.values())
    rep["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    print(f"[mling target] Global-MMLU sample_ids on both sides: {rep['gmmlu_sample_ids_on_both_sides']}; "
          f"done in {rep['seconds']:.0f} s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
