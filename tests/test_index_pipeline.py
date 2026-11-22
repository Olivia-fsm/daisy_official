"""End-to-end check of daisy/index/build_index.py on synthetic parquet files in both input layouts (one vector per
document inside the `metadata` struct; top-level columns with one vector per chunk).

    cd tests && python3 -W ignore -m unittest test_index_pipeline
"""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from daisy.index import build_index as bi  # noqa: E402

DIM = 16


def _texts(rng, n, start):
    return [f"doc {start + i} " + "w" * int(rng.integers(1, 60)) for i in range(n)]


def make_pools(root, rng, n_files=2, rows=230, rg=50):
    main_dir, hq_dir = os.path.join(root, "main"), os.path.join(root, "hq")
    centers = rng.normal(size=(4, DIM)) * 3
    truth = {}
    k = 0
    for dump in ("CC-MAIN-A", "CC-MAIN-B"):
        os.makedirs(os.path.join(main_dir, dump), exist_ok=True)
        os.makedirs(os.path.join(hq_dir, dump), exist_ok=True)
        for f in range(n_files):
            ids = [f"<urn:{k + i}>" for i in range(rows)]
            topic = rng.integers(0, 4, size=rows)
            first = centers[topic] + rng.normal(size=(rows, DIM))
            first[:, 3] += 17.0  # an XLM-R-like outlier coordinate
            texts = _texts(rng, rows, k)
            urls = [f"http://site{(k + i) % 97}.org/{k + i}" for i in range(rows)]
            for i, x in zip(ids, first):
                truth[i] = x
            meta = pa.StructArray.from_arrays(
                [pa.array([dump] * rows), pa.array([list(map(float, x)) for x in first], pa.list_(pa.float64())),
                 pa.array(urls), pa.array(rng.integers(20, 900, size=rows)), pa.array(rng.random(rows))],
                names=["dump", "embeddings", "url", "token_count", "language_score"])
            pq.write_table(pa.table({"text": texts, "id": ids, "metadata": meta}),
                           os.path.join(main_dir, dump, f"000_{f:05d}.parquet"), row_group_size=rg)
            # the per-chunk pool: every 3rd doc, all chunks, chunk 0 == the per-document vector
            sel = list(range(0, rows, 3))
            chunks = [[list(map(float, first[i]))] + [list(map(float, first[i] + rng.normal(size=DIM)))
                                                      for _ in range(int(rng.integers(0, 3)))] for i in sel]
            pq.write_table(pa.table({
                "text": [texts[i] for i in sel], "id": [ids[i] for i in sel], "dump": [dump] * len(sel),
                "embeddings": pa.array(chunks, pa.list_(pa.list_(pa.float64()))),
                "url": [urls[i] for i in sel], "token_count": rng.integers(20, 900, size=len(sel)),
                "language_score": rng.random(len(sel)), "quality_score": rng.random(len(sel))}),
                os.path.join(hq_dir, dump, f"000_{f:05d}.parquet"), row_group_size=rg)
            k += rows
    return main_dir, hq_dir, truth


class IndexPipelineTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.rng = np.random.default_rng(0)
        self.main_dir, self.hq_dir, self.truth = make_pools(self.root, self.rng)
        self.out = os.path.join(self.root, "idx")

    def tearDown(self):
        shutil.rmtree(self.root)

    def run_stage(self, *args):
        bi.main(list(args) + ["--out", self.out, "--pool-dir", self.main_dir, "--dim", str(DIM),
                              "--tokenizer", "bytes", "--workers", "2"])

    def test_pipeline(self):
        self.run_stage("plan", "--target-docs", "700", "--unit-rowgroups", "2")
        plan = bi._jload(os.path.join(self.out, "plan.json"))
        self.assertGreaterEqual(plan["N"], 700)
        self.assertEqual(len({f["dump"] for f in plan["files"]}), 2)  # round-robin over dumps
        self.run_stage("stats", "--stats-files", "4", "--stats-rowgroups", "10", "--stats-keep", "500")
        self.run_stage("shard")
        self.run_stage("tokens")
        self.run_stage("kmeans", "--D", "4", "--kmeans-sample", "600", "--kmeans-iters", "50", "--kmeans-init", "2")
        self.run_stage("heldout", "--heldout-per-topic", "5")

        N = plan["N"]
        st = np.load(os.path.join(self.out, "stats.npz"))
        phi = np.memmap(os.path.join(self.out, "phi.f16"), dtype=np.float16, mode="r", shape=(N, DIM))
        offsets = np.load(os.path.join(self.out, "offsets.i64.npy"))
        toks = np.memmap(os.path.join(self.out, "tokens.u16"), dtype=np.uint16, mode="r")
        self.assertEqual(offsets[-1], len(toks))
        # every row: standardized vector and tokens of the right document
        for u in plan["units"]:
            dm = pq.read_table(os.path.join(self.out, "units", f"u{u['unit']:05d}.docmap.parquet")).to_pydict()
            src = pq.read_table(plan["files"][u["file"]]["path"]).to_pydict()
            for j, (row, doc_id, frow) in enumerate(zip(dm["row"], dm["id"], dm["file_row"])):
                self.assertEqual(src["id"][frow], doc_id)
                want = (self.truth[doc_id] - st["mean"]) / st["std"]
                np.testing.assert_allclose(phi[row].astype(np.float64), want, atol=2e-2, rtol=2e-3)
                t = toks[offsets[row]:offsets[row + 1]]
                self.assertEqual(t[-1], 0)  # eos of the byte tokenizer
                self.assertEqual(bytes(t[:-1].astype(np.uint8)), src["text"][frow].encode()[:4096])
        # standardization removes the outlier coordinate
        self.assertLess(abs(float(phi[:, 3].astype(np.float32).mean())), 0.2)
        # every row gets a topic
        cl = np.load(os.path.join(self.out, "cluster.u8.npy"))
        self.assertEqual(len(cl), N)
        # held-out docs are excluded from training
        held = np.load(os.path.join(self.out, "heldout_rows.npy"))
        mask = np.load(os.path.join(self.out, "train_mask.u8.npy"))
        self.assertTrue((mask[held] == 0).all())
        self.assertEqual(len(held), 20)

    def test_per_chunk_layout(self):
        """Top-level columns with one vector per chunk: read_batches yields the first chunk (or the mean)."""
        f = os.path.join(self.hq_dir, "CC-MAIN-A", "000_00000.parquet")
        rgs = list(range(pq.ParquetFile(f).metadata.num_row_groups))
        n = 0
        for b in bi.read_batches(f, rgs, DIM, "first"):
            np.testing.assert_allclose(b["vec"], [self.truth[i] for i in b["id"]], rtol=1e-6)
            self.assertEqual(len(b["text"]), len(b["id"]))
            n += len(b["id"])
        self.assertEqual(n, pq.ParquetFile(f).metadata.num_rows)
        mean = np.concatenate([b["vec"] for b in bi.read_batches(f, rgs, DIM, "mean", with_text=False)])
        first = np.concatenate([b["vec"] for b in bi.read_batches(f, rgs, DIM, "first", with_text=False)])
        self.assertFalse(np.allclose(mean, first))

    def test_pool_dir_is_required(self):
        with self.assertRaises(SystemExit):
            bi.main(["plan", "--out", self.out])

    def test_doc_vectors_policies(self):
        nested = pa.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0]]], pa.list_(pa.list_(pa.float64())))
        np.testing.assert_allclose(bi.doc_vectors(nested, 2, "first"), [[1, 2], [5, 6]])
        np.testing.assert_allclose(bi.doc_vectors(nested, 2, "mean"), [[2, 3], [5, 6]])
        flat = pa.array([[1.0, 2.0], [3.0, 4.0]], pa.list_(pa.float64()))
        np.testing.assert_allclose(bi.doc_vectors(flat, 2), [[1, 2], [3, 4]])
        with self.assertRaises(ValueError):
            bi.doc_vectors(pa.array([[1.0, 2.0, 3.0]], pa.list_(pa.float64())), 2)


if __name__ == "__main__":
    unittest.main()
