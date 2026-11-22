"""Multilingual pool support, on synthetic data (CPU):

  * a uint32 pool (manifest token_bytes 4, ids above 65,535): DsPool reads exact tokens, capped or not, through the
    memmap and the pread path, and the stream packs them unchanged;
  * DsPool.shard_langs: the language of every shard, from its input file "<lang>/<file>";
  * a rank's shard list is the seeded permutation of all shards split rank::world, and
    daisy/multilingual/build_fw2edu_target.py computes the same lists;
  * target directories with meta.json (uint32, </s> = 2, unpadded): load_items, target_item_indices = the items
    target_items packs; directories without meta.json read as GPT-2 targets (uint16, EOT 50256, padded);
  * DocStore over tokens.u32.

    cd tests && python -m unittest test_mling
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
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from daisy.data.pool import DsPool, ShardDocStream  # noqa: E402

EOS = 2
LANGS = ("cmn_Hani", "deu_Latn", "fra_Latn")


def make_pool(root, rng, shards_per_lang=(5, 3, 4), docs=(20, 50)):
    """uint32 tokens in [100, 200064), </s> = 2 at every end; files '<lang>/000_NNNNN.parquet' in sorted order.
    -> ({key: (tokens, phi row, text hash, url hash)}, {key: language})."""
    os.makedirs(os.path.join(root, "tokens"))
    os.makedirs(os.path.join(root, "sidecar"))
    files, truth, lang_of = [], {}, {}
    for lang, ns in zip(LANGS, shards_per_lang):
        for j in range(ns):
            files.append(f"{lang}/000_{j:05d}.parquet")
    for s, f in enumerate(files):
        lang = f.split("/")[0]
        n = int(rng.integers(*docs))
        lmax = {"cmn_Hani": 140, "deu_Latn": 70, "fra_Latn": 40}[lang]   # languages differ in document length
        lens = rng.integers(3, lmax, size=n)
        toks = [np.append(rng.integers(100, 200064, size=L - 1), EOS).astype(np.uint32) for L in lens]
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
            lang_of[(s << 32) | i] = lang
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump({"files": files, "pool": "x", "token_bytes": 4, "eos_id": EOS, "vocab": 200064}, f)
    return truth, lang_of


class MlingPoolTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.truth, self.lang_of = make_pool(self.root, np.random.default_rng(0))

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_uint32_reads(self):
        pool = DsPool(self.root)
        self.assertEqual(pool.dtype, np.uint32)
        for key, (t, phi, _, _) in list(self.truth.items())[::7]:
            np.testing.assert_array_equal(pool.doc(key), t)
            np.testing.assert_array_equal(pool.read_segment(key, 0, len(t)), t)
            cap = 16
            capped = np.concatenate([t[: cap - 1], t[-1:]]) if len(t) > cap else t
            np.testing.assert_array_equal(pool.doc(key, cap), capped)
            np.testing.assert_array_equal(pool.read_segment(key, 0, len(capped), cap), capped)
            if len(capped) > 3:
                np.testing.assert_array_equal(pool.read_segment(key, 2, len(capped), cap), capped[2:])
            np.testing.assert_array_equal(pool.read_phi(key), phi)
        self.assertTrue(max(int(t.max()) for t, _, _, _ in self.truth.values()) > 65535)

    def test_stream_packs_uint32_tokens(self):
        st = ShardDocStream(DsPool(self.root), seq_len=31, max_doc_tokens=64, seed=3, n_open=3)
        segs = st.plan(4)
        b = st.load(segs)
        x = b[0] if isinstance(b, tuple) else b["x"]
        flat = np.concatenate([DsPool(self.root).read_segment(k, s, e, 64) for k, s, e in segs]).astype(np.int64)
        np.testing.assert_array_equal(np.asarray(x).reshape(-1)[:31], flat[:31])
        self.assertTrue((flat > 65535).any())

    def test_shard_langs(self):
        pool = DsPool(self.root)
        langs = pool.shard_langs()
        self.assertEqual(len(langs), pool.n_shards)
        self.assertEqual(sorted(set(langs)), sorted(LANGS))
        for key, lang in self.lang_of.items():
            self.assertEqual(langs[key >> 32], lang)

    def test_rank_shard_lists(self):
        """A rank's shard list is perm(seed)[rank::world]; build_fw2edu_target.py (which takes its documents from the
        tail of every rank's list) computes exactly the stream's lists."""
        from daisy.multilingual.build_fw2edu_target import rank_lists

        pool = DsPool(self.root)
        for world in (1, 2, 3, 4):
            lists = rank_lists(pool.n_shards, 11, world)
            for r in range(world):
                st = ShardDocStream(pool, 31, 64, seed=11, rank=r, world=world, n_open=2)
                np.testing.assert_array_equal(st.shards, np.random.default_rng(11).permutation(pool.n_shards)[r::world])
                np.testing.assert_array_equal(st.shards, lists[r])


class MlingTargetsTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d)

    def test_meta_targets(self):
        from daisy.labels.targets import load_items, target_item_indices, target_items

        rng = np.random.default_rng(1)
        items = [np.append(rng.integers(100, 200064, size=int(rng.integers(5, 60))), EOS).astype(np.uint32)
                 for _ in range(300)]
        np.concatenate(items).tofile(os.path.join(self.d, "train.bin"))
        with open(os.path.join(self.d, "train.len"), "wb") as f:
            np.save(f, np.array([len(x) for x in items], dtype=np.int64))
        with open(os.path.join(self.d, "meta.json"), "w") as f:
            json.dump({"dtype": "uint32", "eos_id": EOS, "padded": False}, f)
        got = load_items(self.d, "train")
        self.assertEqual(len(got), len(items))
        for a, b in zip(got, items):
            np.testing.assert_array_equal(a, b)
        idx = target_item_indices(self.d, 31, 400)
        spans = target_items(self.d, 31, 400)
        self.assertEqual(len(idx), len(spans))
        for i, sp in zip(idx, spans):
            np.testing.assert_array_equal(items[i][: len(sp)], sp)

    def test_gpt2_targets(self):
        from daisy.labels.targets import load_items

        EOT = 50256
        items = [np.array([5, 6, 7, EOT], dtype=np.uint16), np.array([8, 9, EOT], dtype=np.uint16)]
        padded = [np.concatenate([x, np.full(8 - len(x), EOT, dtype=np.uint16)]) for x in items]
        np.concatenate(padded).tofile(os.path.join(self.d, "train.bin"))
        with open(os.path.join(self.d, "train.len"), "wb") as f:
            np.save(f, np.array([8, 8], dtype=np.uint16))
        got = load_items(self.d, "train")
        for a, b in zip(got, items):
            np.testing.assert_array_equal(a, b)
            self.assertEqual(a.dtype, np.uint16)


class MlingDocStoreTest(unittest.TestCase):
    def test_u32_docstore(self):
        from daisy.data.docstore import DocStore

        d = tempfile.mkdtemp()
        try:
            rng = np.random.default_rng(2)
            docs = [np.append(rng.integers(70000, 200064, size=int(rng.integers(3, 40))), EOS).astype(np.uint32)
                    for _ in range(20)]
            np.concatenate(docs).tofile(os.path.join(d, "tokens.u32"))
            np.save(os.path.join(d, "offsets.i64.npy"), np.concatenate([[0], np.cumsum([len(x) for x in docs])]).astype(np.int64))
            st = DocStore(d)
            self.assertEqual(st.dtype, np.uint32)
            for i, t in enumerate(docs):
                np.testing.assert_array_equal(st.doc(i), t)
                np.testing.assert_array_equal(st.read_segment(i, 0, len(t)), t)
                np.testing.assert_array_equal(st.doc(i, 8), np.concatenate([t[:7], t[-1:]]) if len(t) > 8 else t)
        finally:
            shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main()
