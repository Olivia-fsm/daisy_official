"""The selecting stream (daisy/data/select.py), CPU.

  * every micro-batch holds exactly B x (T+1) tokens; before the scorer has weights it is a random subset of its window;
  * with weights: every selected document scores at least as high as every candidate of the window that was left out
    (top scores), for the linear score, the cosine score and the maximum over K scorers;
  * the candidate window holds lambda x the micro-batch's tokens, and unchosen candidates are consumed;
  * a micro-batch is planned with the weight version current at its index, and a stream restored from state_dict
    plans exactly the micro-batches of the uninterrupted one, also over the tokenized pool.

    cd tests && python -m unittest test_select
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
from daisy.data.docstore import DocStore  # noqa: E402
from daisy.data.pool import DsPool, ShardDocStream  # noqa: E402
from daisy.data.select import SelectingStream  # noqa: E402
from daisy.data.stream import UniformDocStream  # noqa: E402
from helpers import make_run_data  # noqa: E402

EOT, T = 50256, 63


class SelectTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        rng = np.random.default_rng(0)
        n = 3000
        self.lens = rng.integers(5, 120, size=n)
        np.concatenate([np.append(rng.integers(1, 1000, size=L - 1), EOT) for L in self.lens]).astype(np.uint16).tofile(
            os.path.join(self.root, "tokens.u16"))
        np.save(os.path.join(self.root, "offsets.i64.npy"), np.concatenate([[0], np.cumsum(self.lens)]).astype(np.int64))
        self.phi = rng.standard_normal((n, 768)).astype(np.float16)
        self.phi.tofile(os.path.join(self.root, "phi.f16"))
        self.mask = np.ones(n, np.uint8)
        self.w = np.append(rng.standard_normal(768), 0.0).astype(np.float32)

    def tearDown(self):
        shutil.rmtree(self.root)

    def stream(self, mult=4, cosine=False):
        base = UniformDocStream(DocStore(self.root), self.mask, T, 64, seed=3)
        return SelectingStream(base, mult=mult, seed=3, cosine=cosine)

    def check_top_scores(self, st, score_of, B=8, n=60):
        for _ in range(n):
            pos0 = st.base.pos
            segs, flags = st.plan(B)
            cands = list(st.base.order[pos0:st.base.pos])
            self.assertEqual(sum(e - s for _, s, e in segs), B * (T + 1))
            self.assertTrue((flags == 0).all())                       # every document was picked by score
            score = score_of(np.array(cands))
            chosen = {r for r, _, _ in segs}
            sel = [score[cands.index(r)] for r in chosen]
            rest = [score[i] for i, c in enumerate(cands) if c not in chosen]
            if rest:
                self.assertGreaterEqual(min(sel), max(rest) - 1e-4)
            # only one document is cut: the lowest-scored selected one
            cut = [r for r, s, e in segs if e - s < min(self.lens[r], 64)]
            self.assertLessEqual(len(cut), 1)
            if cut:
                self.assertAlmostEqual(float(score[cands.index(cut[0])]), float(min(sel)), places=4)

    def test_window_and_random_fill_before_the_scorer(self):
        st = self.stream()
        pos0 = st.base.pos
        segs, flags = st.plan(2)
        need = 2 * (T + 1)
        self.assertEqual(sum(e - s for _, s, e in segs), need)
        self.assertTrue((flags == 1).all())                           # no weights yet: a random subset of the window
        window = st.base.order[pos0:st.base.pos]
        self.assertGreaterEqual(int(np.minimum(self.lens[window], 64).sum()), 4 * need)
        self.assertLess(int(np.minimum(self.lens[window[:-1]], 64).sum()), 4 * need)   # and no document more
        self.assertTrue({r for r, _, _ in segs} <= set(window.tolist()))
        segs2, _ = st.plan(2)                                         # unchosen candidates are not revisited
        self.assertFalse({r for r, _, _ in segs2} & set(window.tolist()))
        with self.assertRaises(ValueError):
            self.stream(mult=0.5)

    def test_top_scores(self):
        st = self.stream()
        st.set_weights(self.w)
        self.check_top_scores(st, lambda c: self.phi[c].astype(np.float32) @ self.w[:-1])

    def test_cosine_scores(self):
        st = self.stream(cosine=True)
        w = self.w.copy()
        w[:-1] /= np.linalg.norm(w[:-1])
        st.set_weights(w)

        def cos(c):
            p = self.phi[c].astype(np.float32)
            return p @ w[:-1] / np.linalg.norm(p, axis=1)
        self.check_top_scores(st, cos, n=30)

    def test_max_over_scorers(self):
        rng = np.random.default_rng(1)
        W = np.concatenate([rng.standard_normal((3, 768)), np.zeros((3, 1))], 1).astype(np.float32)
        st = self.stream()
        st.set_weights(W)
        self.check_top_scores(st, lambda c: (self.phi[c].astype(np.float32) @ W[:, :-1].T).max(1), n=30)

    def test_weight_versions_and_exact_resume(self):
        """Plans made before a weight version arrived keep the earlier version; a restored stream re-plans the same
        micro-batches (the JSON round trip of a checkpoint)."""
        a = self.stream()
        plans = [a.plan(4) for _ in range(3)]                         # scorer off
        a.set_weights(self.w)
        state = json.loads(json.dumps(a.state_dict()))
        plans += [a.plan(4) for _ in range(5)]
        a.set_weights(-self.w)
        plans += [a.plan(4) for _ in range(5)]
        b = self.stream()
        b.load_state_dict(state)
        self.assertEqual(b.n_planned, 3)
        again = [b.plan(4) for _ in range(5)]
        b.set_weights(-self.w)
        again += [b.plan(4) for _ in range(5)]
        for (s1, f1), (s2, f2) in zip(plans[3:], again):
            self.assertEqual(s1, s2)
            self.assertEqual(f1.tolist(), f2.tolist())
        self.assertTrue(all((f == 1).all() for _, f in plans[:3]) and all((f == 0).all() for _, f in plans[3:]))

    def test_another_pass_when_the_index_is_used_up(self):
        st = self.stream(mult=16)
        st.set_weights(self.w)
        for _ in range(40):                                           # 40 x 16 x 8 rows > the index's tokens
            segs, _ = st.plan(8)
            self.assertEqual(sum(e - s for _, s, e in segs), 8 * (T + 1))
        self.assertGreaterEqual(st.base.epoch, 1)
        state = json.loads(json.dumps(st.state_dict()))
        nxt = [st.plan(8) for _ in range(3)]
        b = self.stream(mult=16)
        b.load_state_dict(state)
        for (s1, _), (s2, _) in zip(nxt, [b.plan(8) for _ in range(3)]):
            self.assertEqual(s1, s2)


class PoolSelectTest(unittest.TestCase):
    """Over the tokenized pool: phi comes from the shard sidecars, and the stream resumes exactly."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.paths, _ = make_run_data(self.root, n_shards=6, per_shard=400)
        self.w = np.load(self.paths["w_static"])

    def tearDown(self):
        shutil.rmtree(self.root)

    def stream(self):
        base = ShardDocStream(DsPool(self.paths["pool"]), 31, 48, seed=5, rank=0, world=1, n_open=3,
                              held_text=np.zeros(0, np.uint64), held_url=np.zeros(0, np.uint64))
        return SelectingStream(base, mult=4, seed=5, cosine=True)

    def phi_of(self, key):
        sh, row = key >> 32, key & 0xFFFFFFFF
        return np.fromfile(os.path.join(self.paths["pool"], "sidecar", f"{sh:05d}.phi.f16"),
                           dtype=np.float16).reshape(-1, 768)[row].astype(np.float32)

    def test_top_scores_and_resume(self):
        a = self.stream()
        a.set_weights(self.w)
        first = [a.plan(6) for _ in range(10)]
        state = json.loads(json.dumps(a.state_dict()))
        nxt = [a.plan(6) for _ in range(10)]
        cos = lambda k: float(self.phi_of(k) @ self.w[:-1] / np.linalg.norm(self.phi_of(k)))
        for segs, flags in first:
            self.assertEqual(sum(e - s for _, s, e in segs), 6 * 32)
            self.assertTrue((flags == 0).all())
        # the documents picked from the windows score far above the pool's average
        picked = np.mean([cos(k) for segs, _ in first for k, _, _ in segs])
        self.assertGreater(picked, 0.02)
        b = self.stream()
        b.load_state_dict(state)
        for (s1, _), (s2, _) in zip(nxt, [b.plan(6) for _ in range(10)]):
            self.assertEqual(s1, s2)
        mb = b.next_microbatch(6)
        self.assertEqual(mb["x"].shape, (6, 31))
        self.assertEqual(mb["phi"].shape[1], 768)


if __name__ == "__main__":
    unittest.main()
