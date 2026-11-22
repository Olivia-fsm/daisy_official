"""Streaming loader over the tokenized pool (daisy/data/pool.py), on a synthetic pool (CPU).

  * every micro-batch holds exactly B (T+1) tokens; documents are consumed contiguously from token 0 (carry-over);
  * each document appears at most once per rank, ranks see disjoint shards, held-out hashes are never trained on;
  * the phi of every segment is its own document's sidecar row;
  * resume from state_dict reproduces the uninterrupted stream exactly;
  * further passes over the pool (reshuffle), shard preloading and the extra exclusion by hash;
  * with many open shards, a window of documents mixes many shards (no long single-shard stretches).

    cd tests && python -m unittest test_pool
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from daisy.data.pool import DsPool, ShardDocStream  # noqa: E402

EOS = 50256


def make_pool(root, rng, n_shards=12, docs=(30, 60)):
    """A synthetic pool of n_shards shards with docs[0] .. docs[1] - 1 documents each.
    -> {key: (tokens, phi row, text hash, url hash)}."""
    os.makedirs(os.path.join(root, "tokens"))
    os.makedirs(os.path.join(root, "sidecar"))
    truth = {}
    for s in range(n_shards):
        n = int(rng.integers(*docs))
        lens = rng.integers(3, 90, size=n)
        toks = [np.append(rng.integers(0, 50000, size=L - 1), EOS).astype(np.uint16) for L in lens]
        np.concatenate(toks).tofile(os.path.join(root, "tokens", f"{s:05d}_unshuffled.ds"))
        np.cumsum(lens).astype(np.uint64).tofile(os.path.join(root, "tokens", f"{s:05d}_unshuffled.ds.index"))
        phi = rng.standard_normal((n, 768)).astype(np.float16)
        phi.tofile(os.path.join(root, "sidecar", f"{s:05d}.phi.f16"))
        th = rng.integers(1, 2**62, size=n, dtype=np.int64).astype(np.uint64)
        uh = rng.integers(1, 2**62, size=n, dtype=np.int64).astype(np.uint64)
        pq.write_table(pa.table({"id": [f"{s}-{i}" for i in range(n)], "text_hash": th, "url_hash": uh}),
                       os.path.join(root, "sidecar", f"{s:05d}.meta.parquet"))
        for i in range(n):
            truth[(s << 32) | i] = (toks[i], phi[i], int(th[i]), int(uh[i]))
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump({"files": [f"f{s}.parquet" for s in range(n_shards)], "pool": "x"}, f)
    return truth


class StreamDsTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.truth = make_pool(self.root, np.random.default_rng(0))
        keys = sorted(self.truth)
        self.held = keys[5:9]                      # held-out documents (by hash)
        self.held_text = [self.truth[k][2] for k in self.held[:2]]
        self.held_url = [self.truth[k][3] for k in self.held[2:]]

    def tearDown(self):
        shutil.rmtree(self.root)

    def stream(self, rank=0, world=1, n_open=4, reshuffle=False, preload=0, subset_tokens=0):
        return ShardDocStream(DsPool(self.root), seq_len=31, max_doc_tokens=64, seed=7, rank=rank, world=world,
                              n_open=n_open, held_text=self.held_text, held_url=self.held_url, reshuffle=reshuffle,
                              preload=preload, subset_tokens=subset_tokens)

    def test_read_retries_transient_errors(self):
        """A pread that fails once (EINVAL) is retried on a fresh descriptor and returns the same tokens; a persistent
        failure raises an OSError naming the shard, row and offsets."""
        from unittest import mock
        import daisy.data.docstore as packed

        pool = DsPool(self.root)
        key = sorted(self.truth)[0]
        good = pool.read_segment(key, 0, 5)
        real, calls = os.pread, {"n": 0}

        def flaky(fd, n, off):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError(22, "Invalid argument")
            return real(fd, n, off)

        with mock.patch.object(packed.os, "pread", flaky), mock.patch("daisy.data.pool.time.sleep"):
            np.testing.assert_array_equal(pool.read_segment(key, 0, 5), good)
        with mock.patch.object(packed.os, "pread", mock.Mock(side_effect=OSError(22, "Invalid argument"))), \
                mock.patch("daisy.data.pool.time.sleep"):
            with self.assertRaises(OSError) as cm:
                pool.read_segment(key, 0, 5)
            self.assertIn(f"row {int(key) & 0xFFFFFFFF}", str(cm.exception))

    def test_packing_coverage_heldout_phi(self):
        st = self.stream()
        segs = []
        for _ in range(40):
            mb = st.next_microbatch(2)
            self.assertEqual(mb["x"].size, 2 * 31)
            self.assertEqual(int((mb["ends"] - mb["starts"]).sum()), 2 * 32)
            for k, s0, e0, ph in zip(mb["rows"], mb["starts"], mb["ends"], mb["phi"]):
                segs.append((int(k), int(s0), int(e0)))
                np.testing.assert_array_equal(ph, self.truth[int(k)][1])
        seen = {}
        for k, s0, e0 in segs:
            seen.setdefault(k, []).append((s0, e0))
        for k, parts in seen.items():
            self.assertNotIn(k, self.held)
            self.assertEqual(parts[0][0], 0)
            for (a0, a1), (b0, b1) in zip(parts, parts[1:]):
                self.assertEqual(a1, b0)                       # contiguous, never revisited
            self.assertLessEqual(parts[-1][1], min(len(self.truth[k][0]), 64))
        self.assertEqual(st.skipped >= 4, True)

    def test_ranks_disjoint_and_resume(self):
        a, b = self.stream(0, 2), self.stream(1, 2)
        self.assertEqual(len(set(a.shards) & set(b.shards)), 0)
        full = self.stream()
        ref = [full.next_microbatch(2)["rows"] for _ in range(12)]
        part = self.stream()
        for _ in range(5):
            part.next_microbatch(2)
        stt = json.loads(json.dumps(part.state_dict()))
        res = self.stream()
        res.load_state_dict(stt)
        for i in range(5, 12):
            np.testing.assert_array_equal(res.next_microbatch(2)["rows"], ref[i])

    def test_reshuffle_passes(self):
        """--daisy_pool_reshuffle: pass 0 (documents and state) is exactly the stream without it, which stops when
        the rank's shards are used up; with it, every later pass draws each allowed document exactly once, in a new
        order of shards and documents per pass (held-out still excluded), and a resume inside a pass or at its end
        continues exactly."""
        allowed = {k for k in self.truth if k not in self.held}
        n = len(allowed)
        a, b = self.stream(reshuffle=True), self.stream()
        p0 = [a._next_doc() for _ in range(n)]
        self.assertEqual(p0, [b._next_doc() for _ in range(n)])
        self.assertEqual(json.dumps(a.state_dict()), json.dumps(b.state_dict()))
        self.assertEqual(set(p0), allowed)
        with self.assertRaises(RuntimeError):
            b._next_doc()
        passes = [p0]
        for p in (1, 2):
            docs = [a._next_doc() for _ in range(n)]
            self.assertEqual(a.epoch, p)
            self.assertEqual(len(set(docs)), n)
            self.assertEqual(set(docs), allowed)                     # each allowed document once per pass
            self.assertNotEqual(docs, passes[-1])
            first = lambda d: list(dict.fromkeys(k >> 32 for k in d))[:4]   # noqa: E731  the shards opened first
            self.assertNotEqual(first(docs), first(passes[-1]))
            passes.append(docs)
        for cut in (n, n + 37):                                      # at the end of pass 0, inside pass 1
            c = self.stream(reshuffle=True)
            for _ in range(cut):
                c._next_doc()
            st = json.loads(json.dumps(c.state_dict()))
            ref = [c._next_doc() for _ in range(n)]
            d = self.stream(reshuffle=True)
            d.load_state_dict(st)
            self.assertEqual([d._next_doc() for _ in range(n)], ref)
        with self.assertRaises(ValueError):
            ShardDocStream(DsPool(self.root), seq_len=31, max_doc_tokens=64, seed=7, subset_tokens=1000, reshuffle=True)

    def test_preload_same_stream(self):
        """--daisy_pool_preload prepares the orders of the shards the stream opens next on a background thread:
        upcoming() names exactly the shards that open next (across a pass boundary with reshuffle), and the
        documents, micro-batches and states are those of preload 0, with a resume inside a pass and at its end, for
        the whole pool, reshuffled passes and a token subset; the prepared orders are used."""
        allowed = len([k for k in self.truth if k not in self.held])
        for kw in ({}, {"reshuffle": True}, {"subset_tokens": 9000}):
            n = 3 * allowed if kw.get("reshuffle") else (allowed if not kw else 600)
            ref, a = self.stream(n_open=2, **kw), self.stream(n_open=2, preload=3, **kw)
            docs_ref = [ref._next_doc() for _ in range(n)]
            docs = [a._next_doc() for _ in range(n)]
            self.assertEqual(docs, docs_ref, kw)
            self.assertEqual(json.dumps(a.state_dict()), json.dumps(ref.state_dict()), kw)
            st = a.prep_stats
            self.assertGreater(st["order_ready"] + st["order_waited"], 1, (kw, st))
            for cut in (allowed // 2, allowed):                 # inside pass 0, at its end
                c = self.stream(n_open=2, preload=3, **kw)
                for _ in range(cut):
                    c._next_doc()
                state = json.loads(json.dumps(c.state_dict()))
                tail = [c._next_doc() for _ in range(n - cut)]
                d = self.stream(n_open=2, preload=3, **kw)
                d.load_state_dict(state)
                self.assertEqual([d._next_doc() for _ in range(n - cut)], tail, (kw, cut))
                self.assertEqual(tail, docs_ref[cut:], (kw, cut))
        # upcoming(m): the next m shards to open, in order, also across the end of a pass
        a = self.stream(n_open=2, reshuffle=True, preload=3)
        opened = []
        fill = a._fill

        def record():
            before = {s for s, _ in a.open}
            fill()
            opened.extend(s for s, _ in a.open if s not in before)

        a._fill = record
        for _ in range(allowed - 40):
            a._next_doc()
        up = a.upcoming(6)
        k = len(opened)
        while len(opened) < k + 6:
            a._next_doc()
        self.assertEqual(opened[k:k + 6], up)
        self.assertGreaterEqual(a.epoch, 1)                      # the 6 shards span the start of pass 1
        mbs = [self.stream(n_open=2, reshuffle=True, preload=p).next_microbatch(4) for p in (0, 3)]
        for key in ("x", "y", "rows", "phi"):
            np.testing.assert_array_equal(mbs[0][key], mbs[1][key])

    def test_exclude_keeps_order(self):
        """excl_text / excl_url (--daisy_pool_exclude): a shard's order is its order without the exclusion minus the
        excluded documents (every other document keeps its place), excluded documents are never drawn, every other
        document comes exactly once per pass, and the stream is unchanged until it would have drawn an excluded one."""
        base = self.stream(n_open=2)
        late = []
        for s in range(12):
            o = base._order0_compute(s)[0]
            late += [(s << 32) | int(i) for i in o[-3:]]           # each shard's last 3 documents
        xt = [self.truth[k][2] for k in late[0::2]]                 # half by text hash, half by url hash
        xu = [self.truth[k][3] for k in late[1::2]]

        def ex():
            return ShardDocStream(DsPool(self.root), seq_len=31, max_doc_tokens=64, seed=7, n_open=2,
                                  held_text=self.held_text, held_url=self.held_url, excl_text=xt, excl_url=xu)

        e = ex()
        for s in range(12):
            o0, o1 = base._order0_compute(s)[0].tolist(), e._order0_compute(s)[0].tolist()
            drop = {k & 0xFFFFFFFF for k in late if k >> 32 == s}
            self.assertEqual(o1, [i for i in o0 if i not in drop])
        allowed = len([k for k in self.truth if k not in self.held])
        ref, e = self.stream(n_open=2), ex()
        docs_ref = [ref._next_doc() for _ in range(allowed)]
        docs = [e._next_doc() for _ in range(allowed - len(late))]
        self.assertFalse(set(docs) & set(late))
        self.assertEqual(sorted(docs), sorted(set(docs_ref) - set(late)))
        first = min(docs_ref.index(k) for k in late)
        self.assertGreater(first, 10)
        self.assertEqual(docs[:first], docs_ref[:first])
        with self.assertRaises(RuntimeError):                        # the pass ends (no reshuffle)
            e._next_doc()

    def test_mixing_across_shards(self):
        st = self.stream(n_open=8)
        shards = []
        for _ in range(15):
            shards += [int(k) >> 32 for k in st.next_microbatch(2)["rows"]]
        self.assertGreaterEqual(len(set(shards[:30])), 5)


if __name__ == "__main__":
    unittest.main()
