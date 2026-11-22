"""pread read path + Prefetcher (daisy/data/docstore.py, stream.py, pool.py, prefetch.py), CPU.

  * DocStore.read_segment / read_phi == the memmap path doc()[s:e] / phi rows, for every segment of capped and
    uncapped documents (including the segment that holds only the kept EOS of a capped document);
  * UniformDocStream / ShardDocStream: load(plan) == pack_segments + the memmap phi rows, batch for batch;
  * Prefetcher: the same batches in the same order as the unwrapped stream, with a thread pool or without;
    state_dict() after k batches == the unwrapped stream's state after k batches; resuming from it continues exactly.

    cd tests && python -m unittest test_prefetch
"""

import os
import shutil
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from daisy.data.docstore import DocStore  # noqa: E402
from daisy.data.prefetch import Prefetcher  # noqa: E402
from daisy.data.stream import UniformDocStream, pack_segments  # noqa: E402
from daisy.data.pool import DsPool, ShardDocStream  # noqa: E402
from test_pool import make_pool  # noqa: E402

EOT = 50256
KEYS = ("x", "y", "seg", "rows", "starts", "ends", "phi")


def make_index(d, rng, n=300):
    lens = rng.integers(2, 200, size=n)
    toks = [np.append(rng.integers(1, 1000, size=L - 1), EOT).astype(np.uint16) for L in lens]
    np.concatenate(toks).tofile(os.path.join(d, "tokens.u16"))
    np.save(os.path.join(d, "offsets.i64.npy"), np.concatenate([[0], np.cumsum(lens)]).astype(np.int64))
    rng.standard_normal((n, 768)).astype(np.float16).tofile(os.path.join(d, "phi.f16"))
    mask = np.ones(n, np.uint8)
    mask[::7] = 0
    return mask


def memmap_load(store, segs, T, cap, phi):
    """The reference read path: memmap reads through pack_segments + store.phi."""
    x, y, seg = pack_segments(store, segs, T, cap)
    rows = np.array([s[0] for s in segs], dtype=np.int64)
    return {"x": x, "y": y, "seg": seg, "rows": rows, "starts": np.array([s[1] for s in segs], dtype=np.int32),
            "ends": np.array([s[2] for s in segs], dtype=np.int32), "phi": phi(rows)}


class PrefetchTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        rng = np.random.default_rng(0)
        self.mask = make_index(self.root, rng)
        self.store = DocStore(self.root)
        self.pool_root = os.path.join(self.root, "pool")
        os.makedirs(self.pool_root)
        make_pool(self.pool_root, rng)

    def tearDown(self):
        shutil.rmtree(self.root)

    def assert_batch_equal(self, a, b):
        for k in KEYS:
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)

    def test_read_segment_matches_memmap(self):
        cap = 64
        for i in range(self.store.N):
            full = self.store.doc(i, cap)
            for s0, e0 in ((0, len(full)), (0, 1), (len(full) - 1, len(full)), (len(full) // 3, len(full) - 1)):
                if e0 > s0:
                    np.testing.assert_array_equal(self.store.read_segment(i, s0, e0, cap), full[s0:e0])
            np.testing.assert_array_equal(self.store.read_segment(i, 0, len(self.store.doc(i)), None),
                                          self.store.doc(i))
            np.testing.assert_array_equal(self.store.read_phi(i), self.store.phi([i])[0])

    def test_uniform_load_matches_memmap_path(self):
        st = UniformDocStream(self.store, self.mask, 31, 64, seed=3)
        with ThreadPoolExecutor(4) as pool:
            for _ in range(30):
                segs = st.plan(3)
                self.assert_batch_equal(st.load(segs, pool), memmap_load(self.store, segs, 31, 64, self.store.phi))

    def test_pool_load_matches_memmap_path(self):
        dp = DsPool(self.pool_root)
        st = ShardDocStream(dp, 31, 64, seed=5, n_open=4)
        with ThreadPoolExecutor(4) as pool:
            for _ in range(30):
                segs = st.plan(2)
                self.assert_batch_equal(st.load(segs, pool), memmap_load(dp, segs, 31, 64, dp.phi))

    def _check_prefetcher(self, make, B, n=40, k=13):
        bare = make()
        ref = [bare.next_microbatch(B) for _ in range(n)]
        pf = Prefetcher(make(), B, depth=3, threads=4)
        try:
            for i in range(k):
                self.assert_batch_equal(pf.next_microbatch(B), ref[i])
            state = pf.state_dict()
            bare = make()
            for _ in range(k):
                bare.next_microbatch(B)
            self.assertEqual(state, bare.state_dict())
            for i in range(k, n):
                self.assert_batch_equal(pf.next_microbatch(B), ref[i])
        finally:
            pf.close()
        pf2 = Prefetcher(make(), B, depth=3, threads=4)          # resume from the state after k batches
        try:
            pf2.load_state_dict(state)
            for i in range(k, n):
                self.assert_batch_equal(pf2.next_microbatch(B), ref[i])
        finally:
            pf2.close()

    def test_prefetcher_uniform(self):
        self._check_prefetcher(lambda: UniformDocStream(self.store, self.mask, 31, 64, seed=9, rank=1, world=2), 2)

    def test_prefetcher_pool(self):
        self._check_prefetcher(lambda: ShardDocStream(DsPool(self.pool_root), 31, 64, seed=11, n_open=3), 2)


if __name__ == "__main__":
    unittest.main()
