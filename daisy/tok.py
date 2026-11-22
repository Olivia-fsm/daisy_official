"""The training data's tokenizer (GPT-2 BPE) for offline jobs.

gpt2_tokenizer loads it with transformers (a hub name, found in the Hugging Face hub cache, or a local directory).
gpt2_tokenizer_json finds a tokenizer.json for the `tokenizers` library, which needs no other file (the MATES baseline
decodes documents with it, daisy/mates/text.py): an explicit path, or the hub cache entry of openai-community/gpt2 or
gpt2 (the same vocabulary and merges). The cache is only read, never written."""

import glob
import os


def gpt2_tokenizer_json(path=""):
    """`path` if given, else the newest tokenizer.json of openai-community/gpt2 or gpt2 in the Hugging Face hub cache
    (HF_HUB_CACHE, HF_HOME/hub, ~/.cache/huggingface/hub)."""
    if path:
        return path
    roots = [os.environ.get("HF_HUB_CACHE", ""),
             os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")]
    for root in [r for r in roots if r]:
        for repo in ("models--openai-community--gpt2", "models--gpt2"):
            hits = sorted(glob.glob(os.path.join(root, repo, "snapshots", "*", "tokenizer.json")))
            if hits:
                return hits[-1]
    raise FileNotFoundError(f"no GPT-2 tokenizer.json under {roots}")


def gpt2_tokenizer(name="gpt2"):
    """The GPT-2 tokenizer of transformers. name: a hub name (read from the hub cache) or a local directory."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.eos_token_id != 50256:
        raise ValueError(f"not the GPT-2 tokenizer: eos {tok.eos_token_id}")
    return tok
