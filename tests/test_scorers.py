"""The scorer (daisy/scorers): Gaussianized ranks, the running ridge with forgetting, the online scorer's schedule
(start at t0, refit every U steps, evaluation before learning) and its checkpoint, the per-step metrics.

    cd tests && python -m unittest test_scorers
"""

import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from daisy.scorers.metrics import pearson, spearman, step_metrics, top_precision  # noqa: E402
from daisy.scorers.online import OnlineScorer  # noqa: E402
from daisy.scorers.ridge import RunningRidge, rank_transform  # noqa: E402

D = 16


class ScorerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.w = torch.randn(D)

    def step(self, n=64, w=None, noise=0.1):
        phi = torch.randn(n, D)
        return phi, phi @ (self.w if w is None else w) + noise * torch.randn(n)

    def test_rank_transform(self):
        y = torch.tensor([3.0, -1e9, 0.5, 2.0, 1e12])                 # heavy tails count like any other value
        t = rank_transform(y)
        self.assertEqual(torch.argsort(t).tolist(), torch.argsort(y).tolist())
        self.assertAlmostEqual(float(t.sum()), 0.0, places=5)
        self.assertAlmostEqual(float(t[1]), -float(t[4]), places=5)
        torch.testing.assert_close(rank_transform(5 * y + 3), t)      # invariant to the labels' scale and offset

    def test_metric_extremes(self):
        _, y = self.step()
        m = step_metrics(rank_transform(y), y)
        self.assertAlmostEqual(m["spearman"], 1.0, places=5)
        self.assertEqual(m["top25"], 1.0)
        self.assertLess(m["mse_raw"], 1e-10)
        self.assertLess(m["mse_cal"], 1e-6)
        self.assertAlmostEqual(spearman(-y, y), -1.0, places=5)
        self.assertEqual(top_precision(-y, y), 0.0)
        self.assertAlmostEqual(pearson(2 * y + 1, y), 1.0, places=5)

    def test_running_ridge_recovers_direction(self):
        ridge = RunningRidge(D, "cpu")
        for _ in range(50):
            phi, y = self.step()
            ridge.add(phi, rank_transform(y))
        ridge.solve()
        cos = torch.nn.functional.cosine_similarity(ridge.w[:-1], self.w, dim=0)
        self.assertGreater(float(cos), 0.99)
        phi, y = self.step()
        self.assertGreater(spearman(ridge.score(phi), y), 0.95)
        other = RunningRidge(D, "cpu")
        other.load_state_dict(ridge.state_dict())
        torch.testing.assert_close(other.score(phi), ridge.score(phi))

    def test_forgetting_follows_a_changed_target(self):
        """After the label direction flips, a ridge with a short memory follows it; one without forgetting does not."""
        fast, slow = RunningRidge(D, "cpu", decay=0.9), RunningRidge(D, "cpu", decay=1.0)
        for k in range(300):
            phi, y = self.step(w=self.w if k < 250 else -self.w)
            for r in (fast, slow):
                r.add(phi, rank_transform(y))
        fast.solve()
        slow.solve()
        cos = lambda r: float(torch.nn.functional.cosine_similarity(r.w[:-1], -self.w, dim=0))
        self.assertGreater(cos(fast), 0.95)
        self.assertLess(cos(slow), 0.0)

    def test_online_schedule_and_checkpoint(self):
        sc = OnlineScorer(D, start=10, update_every=4, ridge_reg=0.01, ridge_decay=0.999, device="cpu",
                          log=lambda *a: None)
        seen = {}
        for step in range(40):
            sc.before_step(step)
            if step % 2 == 0:                                         # labels on every 2nd step
                phi, y = self.step()
                sc.on_labels(step, phi, y)
            sc.maybe_update(step)
            seen[step] = None if sc.weights() is None else sc.weights().clone()
            if step == 21:
                state = sc.state_dict()
        self.assertTrue(all(seen[s] is None for s in range(12)))      # labels before t0 = 10 are not used
        self.assertIsNotNone(seen[12])                                # the first refit: the first multiple of U from t0
        self.assertTrue(torch.equal(seen[13], seen[12]) and torch.equal(seen[15], seen[12]))
        self.assertFalse(torch.equal(seen[16], seen[12]))             # refitted every U = 4 steps
        self.assertEqual(sc.ridge.n, 15)                              # labelled steps 10, 12, ..., 38
        m = sc.pop_metrics()
        self.assertGreater(m["spearman"], 0.9)                        # scored before the ridge learned from them
        self.assertEqual(m["n"], 13)                                  # labelled steps 14, ..., 38 had a fit to score with
        self.assertEqual(sc.pop_metrics()["n"], 0)
        other = OnlineScorer(D, 10, 4, 0.01, 0.999, "cpu", log=lambda *a: None)
        other.load_state_dict(state)
        self.assertTrue(other.active)
        torch.testing.assert_close(other.weights(), seen[21])

    def test_task_agreement(self):
        sc = OnlineScorer(D, start=0, update_every=1, ridge_reg=0.01, ridge_decay=0.999, device="cpu",
                          log=lambda *a: None)
        sc.before_step(0)
        for step in range(5):
            phi, y = self.step()
            out = sc.on_labels(step, phi, y, {"a": y, "b": -y})
            sc.maybe_update(step)
        self.assertAlmostEqual(out["spearman_vs_combined/a"], 1.0, places=5)
        self.assertAlmostEqual(out["spearman_vs_combined/b"], -1.0, places=5)
        self.assertGreater(out["spearman/a/ridge"], 0.9)
        self.assertLess(out["spearman/b/ridge"], -0.9)


if __name__ == "__main__":
    unittest.main()
