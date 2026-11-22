#!/usr/bin/env python
"""Zero-shot downstream accuracy of a checkpoint with EleutherAI lm-evaluation-harness (lm_eval >= 0.4).

No Hugging Face export: `DaisyLM` (a TemplateLM) feeds GPT-2 token ids straight into the trained model and scores
every request from its logits (multiple choice and LAMBADA; no generation). Prompts, normalization and metrics are the
harness' own task YAMLs, so the numbers are comparable with published zero-shot results. Each prompt is one causal
sequence with no prefix, like the first token of a training document under document-masked packing; contexts longer
than the training length are left-truncated.

  python daisy/eval/lm_eval_run.py --run EXP_DIR --out OUT.json [--ckpt latest|STEP] [--tasks ...] [--batch_size 64]

EXP_DIR is an experiment directory (EXP_DIR/daisy/meta_train.json holds the configuration, EXP_DIR/ckpts/<ckpt>/main.pt
the weights; --ckpt STEP = a permanent checkpoint, for accuracy-over-training curves). OUT.json holds, per task, the
harness' metrics (acc, acc_norm, their standard errors, perplexity), the number of items "n" and the seconds; "_meta"
records the checkpoint step, the run's git commit, the lm_eval version and the parameter count. The file is rewritten
after every task.
"""

import argparse
import json
import math
import os
import sys
import time
from argparse import Namespace

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lm_eval.api.model import TemplateLM  # noqa: E402

TASKS = "arc_easy,arc_challenge,piqa,sciq,hellaswag,winogrande,openbookqa,lambada_openai"


def left_truncate(ctx, cont, max_length):
    """Fit context + continuation into max_length tokens by dropping context from the left (keep >= 1 context token);
    the continuation is cut only if it alone overflows."""
    ctx, cont = list(ctx), list(cont)
    over = len(ctx) + len(cont) - max_length
    if over <= 0:
        return ctx, cont
    drop = min(over, max(0, len(ctx) - 1))
    ctx = ctx[drop:]
    over -= drop
    return ctx, (cont[over:] if over > 0 else cont)


class DaisyLM(TemplateLM):
    """The trained model as an lm_eval model: log-likelihood requests only, GPT-2 tokens, one process."""

    def __init__(self, net, tokenizer, max_length, batch_size, device):
        super().__init__()
        self._net, self.tokenizer = net, tokenizer
        self._max_length, self._batch_size, self._device = int(max_length), int(batch_size), torch.device(device)

    @property
    def eot_token_id(self):
        return int(self.tokenizer.eos_token_id)

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return 0

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return 0

    @property
    def world_size(self):
        return 1

    def tok_encode(self, string, add_special_tokens=None, **kwargs):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens, skip_special_tokens=False):
        return self.tokenizer.decode(list(tokens), skip_special_tokens=skip_special_tokens)

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
            out = self._net(inp, targets=tgt.clamp(min=0), get_logits=True)
        logp = F.log_softmax(out["logits"].float(), dim=-1)
        res = []
        for i, (ctx, cont) in enumerate(batch):
            s, e = len(ctx) - 1, len(ctx) - 1 + len(cont)
            sel, gold = logp[i, s:e], tgt[i, s:e]
            res.append((float(sel.gather(1, gold[:, None]).sum()), bool((sel.argmax(-1) == gold).all())))
        return res

    def _loglikelihood_tokens(self, requests, disable_tqdm=False, **kwargs):
        reqs = list(requests)
        prep = []
        for i, (key, ctx, cont) in enumerate(reqs):
            ctx, cont = left_truncate(ctx, cont, self.max_length)
            if not ctx:
                ctx, cont = left_truncate([self.eot_token_id], cont, self.max_length)
            prep.append((i, key, ctx, cont))
        prep.sort(key=lambda t: -(len(t[2]) + len(t[3])))      # length-sorted batches, order restored below
        out = [None] * len(reqs)
        for b in range(0, len(prep), self.batch_size):
            chunk = prep[b:b + self.batch_size]
            for (i, key, _, _), v in zip(chunk, self._score([(c, k) for _, _, c, k in chunk])):
                out[i] = v
                if key is not None and self.cache_hook is not None:
                    self.cache_hook.add_partial("loglikelihood", key, v)
        return out

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        from lm_eval import utils as U

        res = []
        for (string,) in [r.args for r in requests]:
            wins = [(None,) + w for w in map(U.make_disjoint_window, U.get_rolling_token_windows(
                token_list=self.tok_encode(string), prefix_token=self.prefix_token_id, max_seq_len=self.max_length,
                context_len=1))]
            res.append(float(sum(ll for ll, _ in self._loglikelihood_tokens(wins, disable_tqdm=True))))
        return res

    def generate_until(self, requests, disable_tqdm=False):
        raise NotImplementedError("loglikelihood tasks only")


def load_run(exp_dir, device, ckpt="latest"):
    """The run's model, built from the configuration in EXP/daisy/meta_train.json, with the weights of
    EXP/ckpts/<ckpt>/main.pt (daisy/train/checkpoint.py; also the output of daisy/sft/sft_train.py).
    -> (model in eval mode, configuration, the checkpoint's step, the run's metadata)."""
    from daisy.models import get_model

    meta = json.load(open(os.path.join(exp_dir, "daisy", "meta_train.json")))
    cfg = Namespace(**meta["config"])
    net = get_model(cfg)
    ck = torch.load(os.path.join(exp_dir, "ckpts", str(ckpt), "main.pt"), map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model"], strict=True)
    return net.to(device).eval(), cfg, int(ck["itr"]), meta


def main():
    import lm_eval

    from daisy.tok import gpt2_tokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="EXP_DIR: the experiment directory of the run")
    p.add_argument("--out", required=True, help="the JSON file of the results")
    p.add_argument("--ckpt", default="latest", help="checkpoint directory under EXP_DIR/ckpts (latest or a step)")
    p.add_argument("--tasks", default=TASKS)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--limit", type=int, default=0, help="evaluate only the first N items of every task (0 = all)")
    p.add_argument("--tokenizer", default="gpt2")   # the training data's GPT-2 BPE (daisy/tok.py)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, cfg, step, meta = load_run(a.run, dev, a.ckpt)
    tok = gpt2_tokenizer(a.tokenizer)
    lm = DaisyLM(net, tok, cfg.sequence_length, a.batch_size, dev)
    res = {"_meta": {"run": a.run, "ckpt": a.ckpt, "step": step, "git": meta.get("git", {}).get("commit"),
                     "lm_eval": getattr(lm_eval, "__version__", "?"), "limit": a.limit or None, "num_fewshot": 0,
                     "params": sum(p.numel() for p in net.parameters())}}
    print(f"[lm_eval] {a.run} step {step}, lm_eval {res['_meta']['lm_eval']}", flush=True)
    for task in [t for t in a.tasks.split(",") if t.strip()]:
        t0 = time.time()
        r = lm_eval.simple_evaluate(model=lm, tasks=[task], num_fewshot=0, batch_size=a.batch_size,
                                    limit=a.limit or None, log_samples=False)
        block = r["results"][task]
        m = {k.split(",")[0]: float(v) for k, v in block.items()
             if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))}
        m["n"] = int(r.get("n-samples", {}).get(task, {}).get("effective", -1))
        m["seconds"] = round(time.time() - t0, 1)
        res[task] = m
        print(f"[lm_eval] {task}: " + "  ".join(f"{k}={v:.4f}" for k, v in m.items() if k in
                                                ("acc", "acc_norm", "acc_stderr", "perplexity")) + f"  n={m['n']}",
              flush=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
