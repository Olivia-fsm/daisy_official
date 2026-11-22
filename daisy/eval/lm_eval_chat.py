#!/usr/bin/env python
"""lm-evaluation-harness on a checkpoint in the chat format of the SFT data (daisy/sft), and with greedy generation
(IFEval), for instruction-tuned checkpoints (the output of daisy/sft/sft_train.py).

lm_eval_run.py (the zero-shot numbers of pretrained checkpoints) scores every task as raw text with no generation.
This script subclasses its DaisyLM (same weights loader, same log-likelihood scoring) and adds
  * apply_chat_template: the Tulu template exactly as daisy/sft/data.py renders the SFT data (GPT-2 has no chat
    tokens, so the markers are ordinary text):  <|user|>\\n{prompt}\\n<|assistant|>\\n  (+ <|system|> turns, prior
    replies closed by <|endoftext|>\\n). With the chat template (the default; --no_chat turns it off) lm_eval
    (0.4.13) wraps each prompt as a user turn; for multiple-choice tasks it drops the target delimiter, so each
    choice is scored as the assistant's reply right after "<|assistant|>\\n", which is how the SFT data continues.
  * generate_until: greedy decoding (do_sample=False only), stopping at <|endoftext|>, at a task's "until" strings or
    at max_gen_toks, with a KV cache built here from the model's modules (left-padded prompts, per-row RoPE positions;
    the model's own forward has no cache). --no_kv_cache recomputes the whole prefix for every token instead (slow;
    the check of the cache). Requests are sorted by prompt length; the prompt is left-truncated so that prompt +
    max_gen_toks fits the training length. Generative tasks also write their outputs (OUT.<task>.samples.jsonl: doc
    id, prompt (the full context), response).
  * --reply_only (LAMBADA with the chat template): lm_eval moves a context's trailing whitespace into the
    continuation, so the newline of "<|assistant|>\\n" is scored as the first token of every answer. That is harmless
    for multiple choice (the same term in every option, so the flag is off by default) but not for LAMBADA, whose acc
    needs every scored token to be the greedy one: with --reply_only a context that ends in the assistant marker keeps
    its newline, and only the reply is scored. Task lambada_openai_chat (daisy/eval/tasks/, registered with
    include_path) = lambada_openai with the last word as the reply, without the leading space.

  python daisy/eval/lm_eval_chat.py --run EXP_DIR --out OUT.json [--ckpt latest|STEP] [--tasks ...] [--no_chat]
  python daisy/eval/lm_eval_chat.py --run EXP_DIR --out OUT.json --tasks lambada_openai_chat --reply_only

The JSON is lm_eval_run.py's format plus _meta.chat_template ("tulu-plain" or null), _meta.kv_cache and
_meta.reply_only.
"""

import argparse
import json
import math
import os
import re
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from daisy.eval.lm_eval_run import DaisyLM, load_run  # noqa: E402
from daisy.sft.data import EOT, ROLE  # noqa: E402

EOT_TEXT = "<|endoftext|>"
# pieces of a chat prompt; the assistant marker may lack its newline: lm-eval moves a context's trailing whitespace into the
# continuation (TemplateLM._encode_pair), so a context can end in "<|assistant|>"
_SPLIT = re.compile(r"(<\|endoftext\|>|<\|(?:system|user)\|>\n|<\|assistant\|>\n?)")
# the 8 tasks of lm_eval_run.py except LAMBADA (as a chat turn it would be "the assistant says the passage's last word")
MC_TASKS = "arc_easy,arc_challenge,piqa,sciq,hellaswag,winogrande,openbookqa"
TASKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks")      # lambada_openai_chat
ASSISTANT = "<|assistant|>\n"


def render_chat(messages, add_generation_prompt=True):
    """Text form of daisy/sft/data.render: [{"role", "content"}] -> the prompt string. A final assistant turn with
    add_generation_prompt=False is a prefill (left open, no EOT)."""
    out = []
    for i, m in enumerate(messages):
        role, content = m["role"], str(m["content"]).replace(EOT_TEXT, "<endoftext>")
        last = i == len(messages) - 1
        if role in ("system", "user"):
            out.append(ROLE[role] + content + "\n")
        elif role == "assistant":
            if last and not add_generation_prompt:
                out.append(ROLE["assistant"] + content)
            else:
                out.append(ROLE["assistant"] + content + EOT_TEXT + ("" if last else "\n"))
        else:
            raise ValueError(f"unknown role {role!r}")
    if add_generation_prompt:
        if messages and messages[-1]["role"] == "assistant":
            out.append("\n")
        out.append(ROLE["assistant"])
    return "".join(out)


class ChatLM(DaisyLM):
    def __init__(self, *args, gen_batch_size=32, kv_cache=True, reply_only=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._gen_batch_size, self._kv, self._reply_only = int(gen_batch_size), bool(kv_cache), bool(reply_only)

    def _encode_pair(self, context, continuation):
        """TemplateLM._encode_pair; with reply_only, a context ending in the assistant marker keeps the marker's newline,
        so the continuation's tokens are the reply's alone (see the module docstring)."""
        if self._reply_only and context.endswith(ASSISTANT):
            whole, ctx = self.tok_encode(context + continuation), self.tok_encode(context)
            assert whole[:len(ctx)] == ctx, (context[-40:], continuation)
            return ctx, whole[len(ctx):]
        return super()._encode_pair(context, continuation)

    @property
    def tokenizer_name(self):
        return "gpt2-tulu-plain"

    def chat_template(self, chat_template=False):
        return "tulu-plain"

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return render_chat(chat_history, add_generation_prompt)

    def tok_encode(self, string, add_special_tokens=None, **kwargs):
        """Encode a chat prompt piece by piece, as daisy/sft/data.render does (BPE never merges across its pieces): a
        system/user turn (marker + content + "\\n") is one piece, the marker "<|assistant|>\\n" one, the reply after it one,
        the literal <|endoftext|> closing a reply the EOT id, the text after it one. A string without markers is encoded
        whole, as DaisyLM does (prompts without the chat template, --no_chat)."""
        parts = _SPLIT.split(string)
        if len(parts) == 1:
            return super().tok_encode(string)
        ids, pre = [], ""
        for k, part in enumerate(parts):
            if k % 2:                                    # a marker
                if part == EOT_TEXT:
                    ids.append(EOT)
                elif part.startswith("<|assistant|>"):
                    ids.extend(super().tok_encode(part))
                else:
                    pre = part                           # system / user: one piece with the text after it
            else:
                if pre or part:
                    ids.extend(super().tok_encode(pre + part))
                pre = ""
        return ids

    @property
    def max_gen_toks(self):
        return 256

    def _stepper_recompute(self, prompts, maxg):
        """Greedy next tokens without a cache: every call recomputes the batch's prefix (right-padded; causal, so the
        padding after a row's end never reaches it). step(None) -> first next tokens; step(feed) appends feed[i] to row i
        (None: the row is finished and stays as it is) -> next tokens."""
        B, dev, eot = len(prompts), self._device, self.eot_token_id
        buf = torch.full((B, max(len(p) for p in prompts) + maxg), eot, dtype=torch.long, device=dev)
        for i, p in enumerate(prompts):
            buf[i, : len(p)] = torch.tensor(p, dtype=torch.long, device=dev)
        cur = [len(p) for p in prompts]
        rows = torch.arange(B, device=dev)

        def step(feed):
            for i, t in enumerate(feed or []):
                if t is not None:
                    buf[i, cur[i]] = t
                    cur[i] += 1
            inp = buf[:, :max(cur)]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                out = self._net(inp, targets=torch.zeros_like(inp), get_logits=True)   # targets: logits at every position
            return out["logits"][rows, torch.tensor(cur, device=dev) - 1].float().argmax(-1).tolist()
        return step

    def _stepper_kv(self, prompts, maxg):
        """The same with a KV cache, computed from the model's own modules (the forward of daisy/models/llama.py does not
        cache): prompts left-padded to P columns, each row's RoPE positions counted from its first real token (pads
        excluded from attention by a mask; a pad query sees only itself), keys/values of every layer cached, then one
        token per row per step."""
        net, dev, eot = self._net, self._device, self.eot_token_id
        cfg = net.config
        B, P = len(prompts), max(len(p) for p in prompts)
        assert P + maxg <= cfg.sequence_length, (P, maxg)
        pads = torch.tensor([P - len(p) for p in prompts], device=dev)
        buf = torch.full((B, P), eot, dtype=torch.long, device=dev)
        for i, p in enumerate(prompts):
            buf[i, P - len(p):] = torch.tensor(p, dtype=torch.long, device=dev)
        blocks = net.transformer.h
        C, nh = cfg.n_embd, cfg.n_head
        hs = C // nh
        cdt = torch.bfloat16 if dev.type == "cuda" else torch.float32
        kc = [torch.zeros((B, nh, P + maxg, hs), dtype=cdt, device=dev) for _ in blocks]
        vc = [torch.zeros((B, nh, P + maxg, hs), dtype=cdt, device=dev) for _ in blocks]
        freqs = net.freqs_cis.to(dev)
        col = [0]

        def rope(x, cos, sin):                            # = apply_rotary_emb, with per-row positions
            x = x.float().reshape(*x.shape[:-1], -1, 2)
            x0, x1 = x[..., 0], x[..., 1]
            return torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1).flatten(3)

        def run(idx):
            c0, t = col[0], idx.shape[1]
            q_cols = torch.arange(c0, c0 + t, device=dev)
            k_cols = torch.arange(c0 + t, device=dev)
            f = freqs[(q_cols[None, :] - pads[:, None]).clamp(min=0)]                     # (B, t, hs/2, 2)
            cos, sin = f[..., 0].unsqueeze(2), f[..., 1].unsqueeze(2)
            mask = (k_cols[None, None, :] >= pads[:, None, None]) & (k_cols[None, None, :] <= q_cols[None, :, None])
            mask = (mask | (k_cols[None, None, :] == q_cols[None, :, None]))[:, None]      # (B, 1, t, c0 + t)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                x = net.transformer.wte(idx)
                for l, blk in enumerate(blocks):
                    q, k, v = blk.attn.c_attn(blk.ln_1(x)).split(C, dim=2)
                    kc[l][:, :, c0:c0 + t] = rope(k.view(B, t, nh, hs), cos, sin).transpose(1, 2).to(cdt)
                    vc[l][:, :, c0:c0 + t] = v.view(B, t, nh, hs).transpose(1, 2).to(cdt)
                    q = rope(q.view(B, t, nh, hs), cos, sin).transpose(1, 2)
                    y = torch.nn.functional.scaled_dot_product_attention(q, kc[l][:, :, :c0 + t], vc[l][:, :, :c0 + t],
                                                                         attn_mask=mask)
                    x = x + blk.attn.c_proj(y.transpose(1, 2).reshape(B, t, C))
                    x = x + blk.mlp(blk.ln_2(x))
                logits = net.lm_head(net.transformer.ln_f(x[:, -1:]))
            col[0] = c0 + t
            return logits[:, -1].float().argmax(-1).tolist()

        def step(feed):
            if feed is None:
                return run(buf)
            return run(torch.tensor([[eot if t is None else t] for t in feed], dtype=torch.long, device=dev))
        return step

    @torch.no_grad()
    def _generate_ids(self, prompts, max_gens, untils):
        """Greedy continuation of each prompt (lists of token ids) -> the generated ids per prompt (no EOT)."""
        if self._device.type == "cuda":
            # the default attention kernel builds an execution plan for every new sequence length (CPU time on every call),
            # and decoding grows the length by one each step: generation uses the memory-efficient kernel
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                return self._generate_ids_loop(prompts, max_gens, untils)
        return self._generate_ids_loop(prompts, max_gens, untils)

    def _generate_ids_loop(self, prompts, max_gens, untils):
        B, eot = len(prompts), self.eot_token_id
        step = (self._stepper_kv if self._kv else self._stepper_recompute)(prompts, max(max_gens))
        gen = [[] for _ in range(B)]
        done = [False] * B
        nl = step(None)
        while True:
            feed = [None] * B                           # this step's token per row (None: finished earlier)
            for i in range(B):
                if done[i]:
                    continue
                t = nl[i]
                if t == eot:
                    done[i] = True
                    continue
                gen[i].append(t)
                feed[i] = t
                if len(gen[i]) >= max_gens[i]:
                    done[i] = True
                elif untils[i] and any(u in self.tok_decode(gen[i][-24:]) for u in untils[i]):
                    done[i] = True
            if all(done):
                break
            nl = step(feed)
        return gen

    def _generate(self, prompts, max_gens, untils):
        """Greedy continuation of each prompt -> list of strings, cut at the first until string."""
        texts = []
        for i, g in enumerate(self._generate_ids(prompts, max_gens, untils)):
            s = self.tok_decode(g)
            for u in untils[i]:
                if u and u in s:
                    s = s.split(u)[0]
            texts.append(s)
        return texts

    def generate_until(self, requests, disable_tqdm=False):
        reqs = [r.args for r in requests]
        prep = []
        for i, (ctx, kw) in enumerate(reqs):
            kw = dict(kw or {})
            if kw.get("do_sample") or float(kw.get("temperature", 0.0) or 0.0) > 0:
                raise ValueError(f"greedy decoding only, got {kw}")
            until = kw.get("until") or []
            until = [until] if isinstance(until, str) else [u for u in until if u and u != EOT_TEXT]
            g = int(kw.get("max_gen_toks", self.max_gen_toks))
            ids = self.tok_encode(ctx)
            keep = self.max_length - g
            if len(ids) > keep:
                ids = ids[-keep:]
            prep.append((i, ids, g, until))
        prep.sort(key=lambda t: -len(t[1]))
        out = [None] * len(reqs)
        t0 = time.time()
        for b in range(0, len(prep), self._gen_batch_size):
            chunk = prep[b:b + self._gen_batch_size]
            texts = self._generate([c[1] for c in chunk], [c[2] for c in chunk], [c[3] for c in chunk])
            for (i, _, _, _), s in zip(chunk, texts):
                out[i] = s
                if self.cache_hook is not None:
                    self.cache_hook.add_partial("generate_until", reqs[i], s)
            print(f"[lm_eval_chat] generated {min(b + len(chunk), len(prep))}/{len(prep)} ({time.time() - t0:.0f}s)", flush=True)
        return out


def main():
    import lm_eval
    import lm_eval.tasks

    from daisy.tok import gpt2_tokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="EXP_DIR: the experiment directory of the run")
    p.add_argument("--out", required=True, help="the JSON file of the results")
    p.add_argument("--ckpt", default="latest", help="checkpoint directory under EXP_DIR/ckpts (latest or a step)")
    p.add_argument("--tasks", default=MC_TASKS)
    p.add_argument("--no_chat", action="store_true", help="no chat template, e.g. IFEval on a pretrained checkpoint")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gen_batch_size", type=int, default=32)
    p.add_argument("--no_kv_cache", action="store_true", help="recompute the prefix for every new token (the check of the cache)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--show", type=int, default=0, help="print the first N requests' contexts (sanity check)")
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--reply_only", action="store_true", help="score only the reply, not the assistant marker's newline (LAMBADA)")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, cfg, step, meta = load_run(a.run, dev, a.ckpt)
    tok = gpt2_tokenizer(a.tokenizer)
    lm = ChatLM(net, tok, cfg.sequence_length, a.batch_size, dev, gen_batch_size=a.gen_batch_size, kv_cache=not a.no_kv_cache,
                reply_only=a.reply_only)
    tm = lm_eval.tasks.TaskManager(include_path=TASKS_DIR)
    chat = not a.no_chat
    res = {"_meta": {"run": a.run, "ckpt": a.ckpt, "step": step, "git": meta.get("git", {}).get("commit"),
                     "lm_eval": getattr(lm_eval, "__version__", "?"), "limit": a.limit or None, "num_fewshot": 0,
                     "chat_template": "tulu-plain" if chat else None,
                     "kv_cache": not a.no_kv_cache, "reply_only": a.reply_only,
                     "params": sum(p.numel() for p in net.parameters())}}
    print(f"[lm_eval_chat] {a.run} step {step}, lm_eval {res['_meta']['lm_eval']}, chat {chat}", flush=True)
    for task in [t for t in a.tasks.split(",") if t.strip()]:
        t0 = time.time()
        r = lm_eval.simple_evaluate(model=lm, tasks=[task], num_fewshot=0, batch_size=a.batch_size, limit=a.limit or None,
                                    apply_chat_template=chat, fewshot_as_multiturn=False, log_samples=True, task_manager=tm)
        block = r["results"][task]
        m = {k.split(",")[0]: float(v) for k, v in block.items()
             if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))}
        m["n"] = int(r.get("n-samples", {}).get(task, {}).get("effective", -1))
        m["seconds"] = round(time.time() - t0, 1)
        res[task] = m
        samples = r.get("samples", {}).get(task, [])
        for s in samples[:a.show]:
            print(f"[lm_eval_chat] {task} doc {s['doc_id']} context:\n{s['arguments'][0][0]!r}\n  -> {s['arguments'][0][1]!r}",
                  flush=True)
        if samples and samples[0].get("filtered_resps") and isinstance(samples[0]["filtered_resps"][0], str):
            with open(a.out.replace(".json", "") + f".{task}.samples.jsonl", "w") as f:
                for s in samples:
                    f.write(json.dumps({"doc_id": s["doc_id"], "prompt": s["arguments"][0][0],
                                        "response": s["filtered_resps"][0]}) + "\n")
        print(f"[lm_eval_chat] {task}: " + "  ".join(f"{k}={v:.4f}" for k, v in m.items() if k not in ("n", "seconds"))
              + f"  n={m['n']} ({m['seconds']}s)", flush=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
