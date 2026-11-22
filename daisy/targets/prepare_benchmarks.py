#!/usr/bin/env python
"""The benchmark target tasks: OUT/<task>/{train,val}.bin and {train,val}.len for ARC-Easy, ARC-Challenge, PIQA, SciQ
and HellaSwag (the directory passed as --daisy_target_dir and --daisy_eval_dir).

Every item is the question followed by its correct answer, tokenized with GPT-2, ended by one EOS and padded with EOS
to a multiple of 1,024 tokens (daisy/labels/targets.py strips the padding again):

  arc_easy, arc_challenge   "{question} {correct choice}"            train = train, val = test
  piqa                      "{goal} {solution}"                      train = train, val = test
  sciq                      "{support}{question}{correct_answer}"    train = train, val = test
  hellaswag                 "{ctx} {correct ending}"                 train = train, val = validation

The train split is the selection target; the val split is used for the held-out downstream loss only and never for
labels. The public test split of PIQA has no labels (label = -1), so its held-out items pair the goal with the second
solution; the held-out loss on PIQA is therefore a loss on in-domain text, not on verified answers.

<split>.bin: uint16 token ids of all items, concatenated; <split>.len: the padded length of every item (numpy .npy
content, uint16).

  python daisy/targets/prepare_benchmarks.py --out BENCH
"""

import argparse
import os

import numpy as np

PAD_TO = 1024


def _arc(ex):
    answer = dict(zip(ex["choices"]["label"], ex["choices"]["text"])).get(ex["answerKey"])
    return f"{ex['question']} {answer}"


def _hellaswag(ex):
    return f"{ex['ctx']} {ex['endings'][int(ex['label'])]}"


def _piqa(ex):
    return f"{ex['goal']} {[ex['sol1'], ex['sol2']][ex['label']]}"


def _sciq(ex):
    return f"{ex['support']}{ex['question']}{ex['correct_answer']}"


# task -> (dataset id, configuration, split of train.bin, split of val.bin, item text)
BENCHMARKS = {
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "train", "test", _arc),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "train", "test", _arc),
    "piqa": ("piqa", None, "train", "test", _piqa),
    "sciq": ("sciq", None, "train", "test", _sciq),
    "hellaswag": ("Rowan/hellaswag", None, "train", "validation", _hellaswag),
}


def tokenize_with_pad(enc, text, pad_to=PAD_TO):
    """GPT-2 ids of `text` + EOS, padded with EOS to the next multiple of pad_to."""
    ids = enc.encode_ordinary(text)
    ids.append(enc.eot_token)
    out = np.full(-(-len(ids) // pad_to) * pad_to, enc.eot_token, dtype=np.uint16)
    out[: len(ids)] = ids
    return out


def write_split(enc, rows, text_of, out_dir, split):
    items = [tokenize_with_pad(enc, text_of(ex)) for ex in rows]
    np.concatenate(items).tofile(os.path.join(out_dir, f"{split}.bin"))
    with open(os.path.join(out_dir, f"{split}.len"), "wb") as f:
        np.save(f, np.array([len(it) for it in items], dtype=np.uint16))
    return len(items)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="BENCH: one directory per task is written below it")
    p.add_argument("--tasks", default=",".join(BENCHMARKS))
    a = p.parse_args()
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    for task in [t for t in a.tasks.split(",") if t.strip()]:
        name, conf, train_split, val_split, text_of = BENCHMARKS[task]
        out_dir = os.path.join(a.out, task)
        os.makedirs(out_dir, exist_ok=True)
        counts = {}
        for split, src in (("train", train_split), ("val", val_split)):
            ds = load_dataset(name, conf, split=src) if conf else load_dataset(name, split=src)
            counts[split] = write_split(enc, ds, text_of, out_dir, split)
        print(f"[benchmarks] {task}: {counts['train']:,} train items, {counts['val']:,} val items -> {out_dir}")


if __name__ == "__main__":
    main()
