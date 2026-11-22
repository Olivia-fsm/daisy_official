"""Limited-pool runs (--daisy_pool_subset_tokens: daisy/data/pool.py) on a synthetic pool (CPU):

  * pass 0 is the stream without a subset, document for document, until the subset holds its tokens;
  * every later pass is a permutation of the subset (each subset document exactly once), a different one per pass;
  * resume is exact inside pass 0, across pass boundaries and inside later passes, for the packed stream and for a
    selecting stream over it;
  * a subset stream loaded from the state of a stream without a subset, taken inside the subset, continues exactly as
    the subset stream run from the start; a state past the subset is refused.

    cd tests && python -m unittest test_subset
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from daisy.data.pool import DsPool, ShardDocStream  # noqa: E402
from daisy.data.select import SelectingStream  # noqa: E402
from test_pool import make_pool  # noqa: E402

T, CAP, SUBSET, B = 31, 64, 4000, 8      # 256 tokens per micro-batch


class SubsetStreamTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.truth = make_pool(self.root, np.random.default_rng(0), docs=(10, 30))
        keys = sorted(self.truth)
        self.held_text = [self.truth[k][2] for k in keys[5:7]]
        self.held_url = [self.truth[k][3] for k in keys[7:9]]
        self.pool = DsPool(self.root)

    def tearDown(self):
        shutil.rmtree(self.root)

    def stream(self, subset=SUBSET):
        return ShardDocStream(self.pool, seq_len=T, max_doc_tokens=CAP, seed=7, rank=0, world=1, n_open=4,
                              held_text=self.held_text, held_url=self.held_url, subset_tokens=subset)

    def capped(self, k):
        return min(len(self.truth[k][0]), CAP)

    def passes(self, st, n_passes):
        """Draw documents until n_passes passes are complete -> one key list per pass."""
        out = [[]]
        while True:
            k = st._next_doc()
            if st.epoch == len(out):
                if len(out) == n_passes:
                    return out
                out.append([])
            out[st.epoch].append(k)

    def test_pass0_is_the_stream_without_a_subset_and_later_passes_permute_the_subset(self):
        p = self.passes(self.stream(), 4)
        free = self.stream(subset=0)
        self.assertEqual(p[0], [free._next_doc() for _ in range(len(p[0]))])
        tok = np.cumsum([self.capped(k) for k in p[0]])
        self.assertGreaterEqual(tok[-1], SUBSET)
        self.assertLess(tok[-2], SUBSET)
        self.assertEqual(len(set(p[0])), len(p[0]))
        held = {k for k, v in self.truth.items() if v[2] in self.held_text or v[3] in self.held_url}
        self.assertFalse(held & set(p[0]))
        shards = {k >> 32 for k in p[0]}
        self.assertGreater(len(shards), 4)          # pass 0 used up shards and opened new ones
        for q in p[1:]:
            self.assertEqual(sorted(q), sorted(p[0]))
        self.assertNotEqual(p[1], p[0])
        self.assertNotEqual(p[2], p[1])
        # a later pass mixes its shards from the start (draws proportional to what each shard has left)
        self.assertGreaterEqual(len({k >> 32 for k in p[2][:20]}), 4)

    def test_resume_is_exact_across_passes(self):
        ref_st = self.stream()
        ref, states = [], []
        for _ in range(50):
            states.append(json.loads(json.dumps(ref_st.state_dict())))
            ref.append(ref_st.next_microbatch(B))
        self.assertGreaterEqual(ref_st.epoch, 3)
        cuts = [3] + [i for i in range(1, 50) if states[i]["epoch"] != states[i - 1]["epoch"]] + [47]
        self.assertGreaterEqual(len(cuts), 5)
        for i in cuts:
            st = self.stream()
            st.load_state_dict(states[i])
            for j in range(i, min(i + 6, 50)):
                mb = st.next_microbatch(B)
                for key in ("x", "y", "rows", "starts", "ends", "phi"):
                    np.testing.assert_array_equal(mb[key], ref[j][key], err_msg=f"cut {i}, batch {j}, {key}")

    def test_start_from_a_state_without_a_subset(self):
        sub, free = self.stream(), self.stream(subset=0)
        for _ in range(4):
            sub.next_microbatch(B)
            free.next_microbatch(B)
        self.assertEqual(sub.epoch, 0)
        branched = self.stream()
        branched.load_state_dict(json.loads(json.dumps(free.state_dict())))
        self.assertEqual(branched.drawn_tokens, sub.drawn_tokens)
        for _ in range(40):
            a, b = sub.next_microbatch(B), branched.next_microbatch(B)
            np.testing.assert_array_equal(a["x"], b["x"])
            np.testing.assert_array_equal(a["rows"], b["rows"])
        self.assertGreaterEqual(sub.epoch, 2)
        self.assertEqual(sub.state_dict(), branched.state_dict())
        late = self.stream(subset=0)
        for _ in range(25):                          # 6,400 tokens: past the subset, inside the pool
            late.next_microbatch(B)
        with self.assertRaises(ValueError):
            self.stream().load_state_dict(late.state_dict())
        with self.assertRaises(ValueError):          # a subset state into a run with a different subset
            self.stream(subset=SUBSET + 1).load_state_dict(sub.state_dict())

    def test_selecting_stream_over_the_subset(self):
        w = np.append(np.random.default_rng(2).standard_normal(768), 0.0).astype(np.float32)

        def make():
            s = SelectingStream(self.stream(), mult=4, seed=5)
            s.set_weights(w)
            return s

        st = make()
        ref, states = [], []
        for _ in range(30):
            states.append(json.loads(json.dumps(st.state_dict())))
            ref.append(st.next_microbatch(B))
        self.assertGreaterEqual(st.base.epoch, 3)
        subset = {(s << 32) | int(r) for s, n in st.base.subset
                  for r in st.base._order0(s)[0][:n]}
        self.assertTrue(all(int(k) in subset for mb in ref for k in mb["rows"]))
        for i in (2, 11, 23):
            b = make()
            b.load_state_dict(states[i])
            for j in range(i, i + 5):
                mb = b.next_microbatch(B)
                for key in ("x", "rows", "phi"):
                    np.testing.assert_array_equal(mb[key], ref[j][key], err_msg=f"cut {i}, batch {j}, {key}")


if __name__ == "__main__":
    unittest.main()
