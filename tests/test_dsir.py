"""The DSIR baseline (daisy/scorers/dsir.py, daisy/scorers/dsir_build.py, the selector's dsir scores), CPU.

  * a document's score is the sum of r over its unigram and bigram buckets, bigrams never cross documents, and a
    document longer than the cap is scored as t[:cap - 1] + t[-1:] (brute force on explicit n-gram lists);
  * shard scores equal the scores of DsPool.doc(key, cap), the trainer's own capped read, and chunking changes nothing;
  * fit: r = log of the equal-weight mean of the smoothed target distributions minus log of the smoothed raw one;
  * dsir_build.py scores exactly the shards the ranks open (+ margin) and DsirScores reads them; a shard without a file
    is scored on the fly to the same values;
  * with dsir scores set, every selected candidate scores at least as high as every candidate of its window that was
    not selected.

    cd tests && python3 -W ignore -m unittest test_dsir
"""

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
from daisy.scorers.dsir import DsirScores, bucket, doc_scores, fit, score_shard  # noqa: E402
from daisy.scorers.dsir_build import main as build_main, run_shards  # noqa: E402
from helpers import cfg_for, make_benchmarks, make_index, plain_cfg, run  # noqa: E402
from test_pool import make_pool  # noqa: E402

EOS = 50256


def brute_score(doc, cap, r):
    t = [int(x) for x in doc]
    if len(t) > cap:
        t = t[: cap - 1] + t[-1:]
    keys = t + [a * 65536 + b + (1 << 32) for a, b in zip(t[:-1], t[1:])]
    return float(sum(r[int(bucket(np.array([k], np.uint64), len(r))[0])] for k in keys))


class DsirTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.rng = np.random.default_rng(0)

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_scores_brute_force_and_cap(self):
        r = self.rng.standard_normal(97)
        lens = [1, 2, 5, 9, 30, 64, 65, 200]
        docs = [np.append(self.rng.integers(0, 50000, size=L - 1), EOS).astype(np.uint16) for L in lens]
        for cap in (8, 64, 4096):
            got = doc_scores(np.concatenate(docs), lens, cap, r)
            want = [brute_score(d, cap, r) for d in docs]
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12, err_msg=f"cap {cap}")
            ends = np.cumsum(lens)
            for chunk in (1, 50, 1 << 20):                         # chunking only changes how much is read at once
                np.testing.assert_allclose(score_shard(np.concatenate(docs), ends, cap, r, chunk_tokens=chunk), want,
                                           rtol=1e-5, err_msg=f"cap {cap} chunk {chunk}")

    def test_shard_scores_match_trainer_reads(self):
        make_pool(self.root, self.rng, n_shards=3, docs=(40, 60))
        pool = DsPool(self.root)
        r = self.rng.standard_normal(1000)
        for s in range(3):
            toks, ends, _, _ = pool.shard(s)
            v = score_shard(toks, ends, 16, r)
            for i in range(len(ends)):
                d = pool.doc((s << 32) | i, 16)
                self.assertAlmostEqual(float(v[i]), brute_score(d, 10**9, r), places=3)

    def test_fit(self):
        t = np.array([[3.0, 0, 1, 0], [0, 0, 5, 1]])
        raw = np.array([10.0, 10, 10, 10])
        pt = (((t + 1) / (t + 1).sum(1, keepdims=True))).mean(0)
        np.testing.assert_allclose(fit(t, raw, 1.0), np.log(pt) - np.log(0.25))
        self.assertGreater(fit(t, raw)[2], 0)                       # bucket common in the targets
        self.assertLess(fit(t, raw)[1], 0)                          # bucket absent from both targets

    def test_build_tool_and_lookup(self):
        pool_dir, bench, out = (os.path.join(self.root, d) for d in ("pool", "bench", "out"))
        os.makedirs(pool_dir)
        make_pool(pool_dir, self.rng, n_shards=12, docs=(30, 60))
        make_benchmarks(bench, self.rng, tasks=("ta", "tb"))
        counts = [2, 1]
        build_main(["--pool", pool_dir, "--bench", bench, "--out", out, "--tasks", "ta,tb", "--seq_len", "31",
                    "--target_tokens", "256", "--seed", "7", "--world", "2", "--counts", "2,1", "--margin", "1",
                    "--cap", "64", "--buckets", "512", "--raw_shards", "3", "--raw_docs", "10", "--workers", "2",
                    "--check", "5"])
        pool = DsPool(pool_dir)
        want = run_shards(pool.n_shards, 7, 2, counts, 1)
        self.assertEqual(want, sorted(int(f[:5]) for f in os.listdir(out) if f.endswith(".f32")))
        perm = np.random.default_rng(7).permutation(12)                  # the ranks' own lists, as ShardDocStream
        self.assertEqual(want, sorted({int(s) for k in range(2) for s in perm[k::2][: counts[k] + 1]}))
        ds = DsirScores(out, pool, 64)
        r = np.load(os.path.join(out, "dsir_model.npz"))["r"]
        s = want[0]
        keys = (s << 32) | np.arange(len(pool.shard(s)[1]))
        np.testing.assert_allclose(ds.scores(keys), [brute_score(pool.doc(int(k), 64), 64, r) for k in keys], rtol=1e-5)
        other = next(x for x in range(12) if x not in want)               # no file: scored on the fly, file written
        keys = (other << 32) | np.arange(len(pool.shard(other)[1]))
        np.testing.assert_allclose(ds.scores(keys), [brute_score(pool.doc(int(k), 64), 64, r) for k in keys], rtol=1e-5)
        self.assertTrue(os.path.exists(os.path.join(out, f"{other:05d}.f32")))
        with self.assertRaises(ValueError):
            DsirScores(out, pool, 32)                                      # scores computed with another cap

    def test_selection_takes_top_dsir_scores(self):
        make_pool(self.root, self.rng, n_shards=6, docs=(80, 140))
        pool = DsPool(self.root)
        out = os.path.join(self.root, "dsir")
        os.makedirs(out)
        r = self.rng.standard_normal(256)
        np.savez(os.path.join(out, "dsir_model.npz"), r=r)
        with open(os.path.join(out, "dsir.json"), "w") as f:
            f.write('{"cap": 64, "buckets": 256, "alpha": 1.0}')
        base = ShardDocStream(pool, seq_len=31, max_doc_tokens=64, seed=3, n_open=3)
        st = SelectingStream(base, mult=4, seed=3)
        st.dsir = DsirScores(out, pool, 64)
        windows = []
        draw = st._draw_window
        st._draw_window = lambda need: windows.append(draw(need)) or windows[-1]
        st.set_weights(np.zeros(769, np.float32))
        n_sel = 0
        for _ in range(30):
            segs, fl = st.plan(4)
            cands = windows[-1][0]
            score = dict(zip(cands.tolist(), st.dsir.scores(cands).tolist()))
            taken = {k for k, _, _ in segs}
            sel = [score[k] for (k, _, _), f in zip(segs, fl) if not f]
            rest = [score[k] for k in cands.tolist() if k not in taken]
            n_sel += len(sel)
            if sel and rest:
                self.assertGreaterEqual(min(sel), max(rest))
        self.assertGreater(n_sel, 0)


class DsirTrainerTest(unittest.TestCase):
    """Through the trainer (train_daisy), over a synthetic pool: a DSIR run with a learner (labels and ridge, which do
    not select here) and a DSIR run without a learner train the same model, the scorer selects from the scorer start,
    a stopped and resumed run ends identical, and the flag refuses a run without the pool or with a second fixed
    scorer."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        rng = np.random.default_rng(0)
        self.idx, self.bench, self.pool = (os.path.join(self.root, d) for d in ("idx", "bench", "pool"))
        os.makedirs(self.idx)
        os.makedirs(self.pool)
        make_index(self.idx, rng)
        make_benchmarks(self.bench, rng)
        make_pool(self.pool, np.random.default_rng(3), n_shards=8, docs=(80, 140))
        self.dsir = os.path.join(self.root, "dsir")
        build_main(["--pool", self.pool, "--bench", self.bench, "--out", self.dsir, "--tasks", "taska,taskb",
                    "--seq_len", "64", "--target_tokens", "260", "--shards", ",".join(map(str, range(8))),
                    "--buckets", "512", "--raw_shards", "4", "--raw_docs", "20", "--workers", "2", "--check", "4"])

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_dsir_selection_through_trainer(self):
        import torch
        from unittest import mock

        sel = ["--batch_size", "4", "--daisy_pool", self.pool, "--daisy_pool_open_shards", "4",
               "--daisy_eval_dir", self.bench, "--daisy_select_mult", "2", "--daisy_select_dsir", self.dsir]
        exp_l, exp_n = os.path.join(self.root, "learner"), os.path.join(self.root, "no_learner")
        picks = {"random": 0, "dsir": 0}                    # micro-batches filled at random / by DSIR score
        take_random, take_top = SelectingStream._take_random, SelectingStream._take_top_scores

        def count_random(stream, *a):
            picks["random"] += 1
            return take_random(stream, *a)

        def count_top(stream, *a):
            picks["dsir"] += stream.dsir is not None
            return take_top(stream, *a)

        with mock.patch.object(SelectingStream, "_take_random", count_random), \
                mock.patch.object(SelectingStream, "_take_top_scores", count_top):
            m_l = run(cfg_for(self.idx, self.bench, 28, ["--daisy_label_every", "2", *sel]), exp_l)
        self.assertTrue(picks["random"] > 0 and picks["dsir"] > 0)          # uniform first, then DSIR selects
        no_learner = lambda *x: plain_cfg(self.idx, 28, ["--daisy_scorer_start", "16", *sel, *x])
        m_n = run(no_learner(), exp_n)
        for (n, p1), p2 in zip(m_l.named_parameters(), m_n.parameters()):
            self.assertTrue(torch.equal(p1, p2), n)
        part = os.path.join(self.root, "stopped")
        with mock.patch.dict(os.environ, {"DAISY_STOP_AT": "1", "DAISY_STOP_MIN_ITER": "20"}):
            run(no_learner(), part)
        m_r = run(no_learner("--resume_from", os.path.join(part, "ckpts", "latest")), part)
        for (n, p1), p2 in zip(m_n.named_parameters(), m_r.parameters()):
            self.assertTrue(torch.equal(p1, p2), n)
        with self.assertRaises(ValueError):                                 # DSIR scores are per pool document
            run(plain_cfg(self.idx, 4, ["--daisy_select_mult", "2", "--daisy_select_dsir", self.dsir]),
                os.path.join(self.root, "no_pool"))
        wp = os.path.join(self.root, "w.npy")
        np.save(wp, np.zeros(769, np.float32))
        with self.assertRaises(ValueError):
            run(no_learner("--daisy_select_static", wp), os.path.join(self.root, "two_scorers"))


if __name__ == "__main__":
    unittest.main()
