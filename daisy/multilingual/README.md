# Multilingual experiments

The multilingual runs train on six languages of FineWeb2 (French, German, Chinese, Spanish, Portuguese, Italian;
`fra_Latn`, `deu_Latn`, `cmn_Hani`, `spa_Latn`, `por_Latn`, `ita_Latn`) with the same trainer as the English runs.
What differs is the data: a multilingual tokenizer with a 200k vocabulary (tokens are `uint32`), mmBERT-base as the
frozen document embedding phi, and targets built from Global-MMLU and INCLUDE. The tools of this directory build that
data and evaluate the checkpoints. Each file's docstring describes its inputs, outputs and options in full.

## Inputs you provide

- **The embedded FineWeb2 copy** (`DATAPATH/fineweb2-embedded` below): `<lang>/*.parquet`, one directory per
  language, every file with the columns `text`, `id` and a struct `metadata` holding `embeddings` (768 floats: the
  `jhu-clsp/mmBERT-base` vector of the document, the mean of the last hidden states over all its tokens, up to 8192
  tokens) and FineWeb2's `url`, `dump`, `language_score`, `minhash_cluster_size`.
- **The multilingual tokenizer** (`TOKENIZER/tokenizer.json` below): a `tokenizers`-library `tokenizer.json` with a
  200k vocabulary (200,064 ids), `</s>` = 2 as end of document, no BOS.

Public models and datasets that the tools download from the Hugging Face hub: `jhu-clsp/mmBERT-base`, the quality
classifier `epfml/FineWeb2-HQ-PlusPlus-Classifier` (two heads on the mmBERT vector), `CohereLabs/Global-MMLU` and
`CohereLabs/include-base-44`.

## Pipeline

All commands run from the repository root. `DATAPATH`, `TARGETS`, `SCORERS`, `EXPS` and `EVALS` are the locations of
`scripts/env.sh`.

1. **Tokenized pool with embedding sidecars** (`fw2_tokenize.py`). Four modes: the statistics that standardize phi
   (GPU), the sidecars with phi, the hashes and the two quality scores of every document (GPU), the tokens
   (`datatrove`, CPU only), and a check of the result. The sidecar and tokenize modes can be split over jobs with
   `--local-tasks` / `--local-rank-offset`.

   ```bash
   P=DATAPATH/fineweb2-embedded; O=DATAPATH/fineweb2-tokenized
   python daisy/multilingual/fw2_tokenize.py --mode stats    --pool $P --stats $O/stats.npz
   python daisy/multilingual/fw2_tokenize.py --mode sidecar  --pool $P --tokenizer TOKENIZER/tokenizer.json --out $O --stats $O/stats.npz
   python daisy/multilingual/fw2_tokenize.py --mode tokenize --pool $P --tokenizer TOKENIZER/tokenizer.json --out $O
   python daisy/multilingual/fw2_tokenize.py --mode verify   --out $O
   ```

2. **Held-out FineWeb2 set** (`build_mling_index.py`): 1,024 documents per language drawn from the pool, in the
   layout of the embedding index. A run passes it as `--daisy_index_dir`; the trainer excludes these documents and
   their copies from training and reports the held-out FineWeb2 loss per language.

   ```bash
   python daisy/multilingual/build_mling_index.py --pool DATAPATH/fineweb2-tokenized --out DATAPATH/fineweb2-index
   ```

3. **Benchmark targets** (`build_mling_target.py`, GPU): Global-MMLU and INCLUDE in the six languages, 12 tasks
   `gmmlu_<lang>` and `include_<lang>` (`fr`, `de`, `zh`, `es`, `pt`, `it`). Each task has a target split (`train.*`,
   used for the utility labels) and a disjoint held-out split (`val.*`, used for evaluation only); Global-MMLU is
   split by question id, so no held-out question has a translation among the target items.

   ```bash
   python daisy/multilingual/build_mling_target.py --out TARGETS/mling --tokenizer TOKENIZER/tokenizer.json
   ```

4. **Similarity baseline** (`similarity_mling.py`): the target centroid in the pool's standardized mmBERT space.

   ```bash
   TASKS=gmmlu_fr,gmmlu_de,gmmlu_zh,gmmlu_es,gmmlu_pt,gmmlu_it,include_fr,include_de,include_zh,include_es,include_pt,include_it
   python daisy/multilingual/similarity_mling.py --bench TARGETS/mling --tasks $TASKS \
       --stats DATAPATH/fineweb2-tokenized/stats.npz --out SCORERS/similarity_mling
   ```

5. **Training** (`scripts/train.sh`, as the English runs, with the multilingual pool, index and vocabulary):

   ```bash
   export POOL=DATAPATH/fineweb2-tokenized INDEX=DATAPATH/fineweb2-index VOCAB=200064
   EVAL="--daisy_eval_dir TARGETS/mling --daisy_eval_tasks $TASKS"
   NAME=mling-uniform       EXTRA="$EVAL" bash scripts/train.sh
   NAME=mling-daisy-l4      EXTRA="--daisy_learn --daisy_select_mult 4 --daisy_targets $TASKS --daisy_target_dir TARGETS/mling $EVAL" bash scripts/train.sh
   NAME=mling-similarity-l4 EXTRA="--daisy_select_mult 4 --daisy_select_cosine --daisy_select_static SCORERS/similarity_mling/w_static.npy $EVAL" bash scripts/train.sh
   ```

   Other window factors change `--daisy_select_mult`; at lambda 64 add `--daisy_pool_reshuffle`, as
   `scripts/runs.sh` does for the English runs.

6. **Zero-shot accuracy** (`lm_eval_mling.py`, needs `lm_eval`): accuracy on the held-out items of the 12 tasks, in
   the harness' multiple-choice format (`mc`) and in a cloze format scored on the option texts (`cf`). `--dry` prints
   the items, the checks and one rendered request per task without a model.

   ```bash
   python daisy/multilingual/lm_eval_mling.py --run EXPS/mling-daisy-l4 --out EVALS/mling-daisy-l4.json \
       --targets TARGETS/mling --tokenizer TOKENIZER/tokenizer.json [--ckpt STEP]
   ```

## High-quality FineWeb2 target

The counterpart of the English FineWeb-Edu target: documents of the pool itself that the FineWeb-Edu head of the
quality classifier scores highly.

7. **Scores** (`fw2_edu_scores.py`): one byte per pool document, `floor(10 x score)`, in the format of
   `--daisy_pool_keep`, and a report of the shares of documents and tokens above each cutoff per language.

   ```bash
   python daisy/multilingual/fw2_edu_scores.py --pool DATAPATH/fineweb2-tokenized --out DATAPATH/fineweb2-edu-scores
   ```

8. **Target and held-out sets** (`build_fw2edu_target.py`): per language two target tasks (`fw2edu_<lang>_t0`, `_t1`)
   and a held-out set (`fw2edu_<lang>_heldout`), taken from pool shards at the end of every rank's shard list and
   screened against the held-out FineWeb2 documents and the held-out benchmark items. `exclude/` holds the hashes of
   all chosen documents: a run passes it as `--daisy_pool_exclude`, so that neither they nor their copies are trained
   on. `--stream_seed` and `--world` are the stream seed (`"seed"` in a checkpoint's `stream_r0.json`) and the number
   of GPUs of the runs; `--check_runs` verifies that existing runs have not opened the source shards.

   ```bash
   python daisy/multilingual/build_fw2edu_target.py --pool DATAPATH/fineweb2-tokenized \
       --scores DATAPATH/fineweb2-edu-scores --index DATAPATH/fineweb2-index --bench TARGETS/mling \
       --out TARGETS/fw2edu --stream_seed SEED [--check_runs EXPS/mling-uniform,...]
   ```

   The names of the target tasks are in `TARGETS/fw2edu/report.json` (`"target_tasks"`). The similarity scorer of
   this target is step 4 with `--bench TARGETS/fw2edu --tasks <target_tasks>`. A run on this target passes
   `--daisy_targets <target_tasks> --daisy_target_dir TARGETS/fw2edu --daisy_pool_exclude TARGETS/fw2edu/exclude`;
   uniform sampling from the high-scoring documents only is
   `--daisy_pool_keep DATAPATH/fineweb2-edu-scores --daisy_pool_keep_min 30` (score >= 3.0, the default cutoff of the
   target). The loss on the held-out sets is reported during training with
   `--daisy_eval_dir TARGETS/fw2edu --daisy_eval_tasks fw2edu_fr_heldout,...`, or afterwards for saved checkpoints
   by `daisy/eval/heldout_ckpt.py`.
