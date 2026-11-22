"""daisy/eval/lm_eval_chat.py: the chat prompts it builds tokenize exactly as the SFT data (daisy/sft/data.render),
so a choice or a generation scored after "<|assistant|>\\n" sees the tokens the SFT trained on, and lm_eval's split of
context and continuation (TemplateLM._encode_pair: trailing whitespace of the context moves into the continuation) keeps
the context's tokens a prefix of the whole; greedy decoding with the KV cache gives the tokens of the model's own
forward.

The tokenization tests need the GPT-2 tokenizer (transformers) and are skipped without it; the test of
TemplateLM._encode_pair needs lm_eval. The other tests run without either.

    cd tests && python3 -m unittest test_lm_eval_chat
"""

import os
import sys
import types
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
try:
    from lm_eval.api.model import TemplateLM
except ImportError:  # lm_eval is an optional dependency: a stand-in base class for the tests that do not call it
    for name in ("lm_eval", "lm_eval.api"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["lm_eval.api.model"] = types.SimpleNamespace(TemplateLM=object)
    TemplateLM = object
HAVE_LM_EVAL = TemplateLM is not object
from daisy import config  # noqa: E402
from daisy.eval.lm_eval_chat import ChatLM, render_chat  # noqa: E402
from daisy.models import get_model  # noqa: E402
from daisy.sft.data import EOT, render  # noqa: E402

CASES = [
    [{"role": "user", "content": "Question: Which gas do plants take in?\nAnswer:"}],
    [{"role": "user", "content": "Write a haiku about rain. Do not use any commas."}],
    [{"role": "system", "content": "You are a helpful assistant."},
     {"role": "user", "content": "  leading spaces and a trailing newline\n"}],
    [{"role": "user", "content": "First question?"}, {"role": "assistant", "content": "First answer."},
     {"role": "user", "content": "Second question, with <|endoftext|> inside."}],
]
REPLIES = ["carbon dioxide", " Soft rain on the roof", "Sure.\n\nTwo paragraphs.", "Second answer"]


def encode_pair(lm, context, continuation):
    """TemplateLM._encode_pair of lm_eval 0.4.13."""
    n = len(context) - len(context.rstrip())
    if n > 0:
        continuation, context = context[-n:] + continuation, context[:-n]
    whole, ctx = lm.tok_encode(context + continuation), lm.tok_encode(context)
    return ctx, whole[len(ctx):], whole


class TestRenderChat(unittest.TestCase):
    def test_prefill_and_markers(self):
        s = render_chat([{"role": "user", "content": "Hi"}])
        self.assertEqual(s, "<|user|>\nHi\n<|assistant|>\n")
        s = render_chat([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "The best answer is"}],
                        add_generation_prompt=False)
        self.assertEqual(s, "<|user|>\nHi\n<|assistant|>\nThe best answer is")
        s = render_chat(CASES[3])                    # a prior reply is closed by the EOT text; a literal one is rewritten
        self.assertEqual(s.count("<|endoftext|>"), 1)
        self.assertIn("with <endoftext> inside.", s)
        with self.assertRaises(ValueError):
            render_chat([{"role": "tool", "content": "x"}])


class TestChatTemplate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from daisy.tok import gpt2_tokenizer

            cls.tok = gpt2_tokenizer()
        except (ImportError, OSError) as ex:
            raise unittest.SkipTest(f"needs the GPT-2 tokenizer (transformers): {ex}")
        cls.lm = ChatLM(None, cls.tok, 2048, 1, "cpu")

    def enc(self, s):
        return self.tok.encode(s, add_special_tokens=False)

    def test_prompt_plus_reply_equals_sft_render(self):
        for msgs, reply in zip(CASES, REPLIES):
            want = [int(t) for t in render(msgs + [{"role": "assistant", "content": reply}], self.enc)[0]]
            prompt = render_chat(msgs)
            self.assertEqual(self.lm.tok_encode(prompt + reply) + [EOT], want, msgs)
            p = self.lm.tok_encode(prompt)                       # the generation prompt
            self.assertEqual(want[:len(p)], p, msgs)
            ctx, cont, whole = encode_pair(self.lm, prompt, reply)
            self.assertEqual(whole[:len(ctx)], ctx, msgs)        # the split is clean
            self.assertEqual(whole + [EOT], want, msgs)

    def test_eot_marker(self):
        ids = self.lm.tok_encode(render_chat(CASES[3]))
        self.assertEqual(ids.count(EOT), 1)          # the prior reply's closing EOT is the EOT id, the literal one is not

    def test_reply_only(self):
        """--reply_only (LAMBADA): the continuation is the reply's tokens alone, the whole is the SFT render; without it the
        marker's newline (one token) leads the continuation, as lm_eval's own split does."""
        lm = ChatLM(None, self.tok, 2048, 1, "cpu", reply_only=True)
        for msgs, reply in zip(CASES, REPLIES):
            want = [int(t) for t in render(msgs + [{"role": "assistant", "content": reply}], self.enc)[0]]
            prompt = render_chat(msgs)
            ctx, cont = lm._encode_pair(prompt, reply)
            self.assertEqual(ctx + cont + [EOT], want, msgs)
            self.assertEqual(ctx, self.lm.tok_encode(prompt), msgs)
            self.assertEqual(cont, self.enc(reply), msgs)
            ctx0, cont0, _ = encode_pair(self.lm, prompt, reply)
            self.assertEqual(cont0, self.enc("\n") + cont, msgs)

    @unittest.skipUnless(HAVE_LM_EVAL, "needs lm_eval (TemplateLM._encode_pair)")
    def test_reply_only_leaves_other_contexts(self):
        """A context that does not end in the assistant marker is split by lm_eval itself, with or without the flag."""
        lm = ChatLM(None, self.tok, 2048, 1, "cpu", reply_only=True)
        self.assertEqual(lm._encode_pair("Question: 2+2?\nAnswer:", " 4"), self.lm._encode_pair("Question: 2+2?\nAnswer:", " 4"))

    def test_prompt_without_markers_unchanged(self):
        s = "Question: What is 2+2?\nAnswer:"
        self.assertEqual(self.lm.tok_encode(s), self.enc(s))


class TestKVCache(unittest.TestCase):
    def test_kv_equals_recompute(self):
        """Greedy decoding with the KV cache (left-padded prompts, per-row positions) gives the tokens of the recompute path
        (right-padded, the model's own forward), in fp32 on CPU, on a small randomly initialized model."""
        cfg = config.parse_args(["--n_layer", "2", "--n_embd", "128", "--n_head", "4", "--sequence_length", "128",
                                 "--device", "cpu", "--dtype", "float32", "--experiment_name", "test",
                                 "--daisy_index_dir", "unused"])
        torch.manual_seed(0)
        net = get_model(cfg).eval()
        tok = types.SimpleNamespace(eos_token_id=EOT)            # decoding only asks the tokenizer for the EOS id
        g = torch.Generator().manual_seed(1)
        prompts = [torch.randint(0, 50000, (n,), generator=g).tolist() for n in (5, 17, 9, 1)]
        out = {}
        for kv in (True, False):
            lm = ChatLM(net, tok, 128, 1, "cpu", kv_cache=kv)
            out[kv] = lm._generate_ids(prompts, [40, 25, 40, 40], [[], [], [], []])
        self.assertEqual(out[True], out[False])
        self.assertEqual([len(g) for g in out[True]], [40, 25, 40, 40])     # random weights: EOT is never the argmax


if __name__ == "__main__":
    unittest.main()
