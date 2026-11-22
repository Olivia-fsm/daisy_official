# DAiSY: Online Data Selection by Adaptive Utility Estimation for LLM Pretraining

Code of the paper "DAiSY: Online Data Selection by Adaptive Utility Estimation for LLM Pretraining". DAiSY selects the pretraining data of
a language model during training. It learns a linear estimator of per-document attribution scores alongside the model,
from labels that are read off the training backward pass, and scores every candidate with one dot product on a frozen,
precomputed embedding. No candidate passes through the model unless it is trained on.

This repository contains the method, the baselines that share its data stream (uniform sampling, embedding similarity,
DSIR, MATES), the construction of the selection targets, the evaluation, and the commands that reproduce the main
experiments of the paper.

## Contents

- [The method in the code](#the-method-in-the-code)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Reproducing the main experiments](#reproducing-the-main-experiments)
  - [0. Paths](#0-paths)
  - [1. Data: the embedding index and the tokenized pool](#1-data-the-embedding-index-and-the-tokenized-pool)
  - [2. Targets and fixed scorers](#2-targets-and-fixed-scorers)
  - [3. Training](#3-training)
  - [4. Evaluation](#4-evaluation)
- [Outputs of a run](#outputs-of-a-run)
- [Further experiments](#further-experiments)
- [Tests](#tests)
- [License](#license)

## The method in the code

Every document `x` of the pool has a frozen embedding `phi(x)`, computed once before training. Training streams
documents from the tokenized pool. One training step:

1. **Candidate window** (`daisy/data/select.py`: `SelectingStream._draw_window`). For each micro-batch, whole documents
   are drawn in the stream's order until they hold `lambda` times the tokens of the micro-batch
   (`--daisy_select_mult`).
2. **Scoring and selection** (`SelectingStream._take_top_scores`). Every candidate is scored by
   `S(x) = w^T [phi(x); 1]`; the highest-scored candidates fill the micro-batch, and the last one is cut to fill it
   exactly. The rest of the window is discarded. Until the scorer starts (step `t0 = 2,500`), the micro-batch is a
   uniformly random subset of its window.
3. **Training with label gates** (`daisy/train/trainer.py`: `DaisyTrainer._micro_step`; `daisy/models/label_gate.py`).
   The selected documents are packed into rows with document-masked attention (`daisy/models/doc_mask.py`) and trained
   on as usual. On every `L`-th step (`L = 4`), a gate on every linear layer of the transformer blocks reads, inside
   the same backward pass, the inner product between the document's gradient and the preconditioned target direction
   `P_t ⊙ grad L_V`. Summed over the tokens of a document and over the layers, this is the document's utility label
   `y(x)` (`daisy/labels/gates.py`: `TargetLabels`). The gates pass the gradients through unchanged, so training is
   identical with and without them.
4. **Ridge update** (`daisy/train/learner.py`: `OnlineLearner.after_step`; `daisy/scorers/ridge.py`). The labels of a
   step are turned into Gaussianized ranks and added to a running ridge regression from `phi` to the labels, with
   forgetting factor `gamma = 0.999` per labelled step. Every `U = 10` steps the ridge is solved in closed form and
   its weights `w` go to the selector on every process.
5. **Target refresh** (`daisy/labels/targets.py`: `SelectionTargets.refresh_combined`). Every `R = 200` steps, one
   forward and backward pass over the target rows refreshes the target gradient `grad L_V`, which is multiplied by the
   preconditioner of Adam. The target is the mean over the target tasks, for example the training splits of five
   benchmarks.

The held-out losses are computed at every refresh and never feed back into training (`DaisyTrainer._refresh`).

## Repository layout

```
train.py                 entry point of every training run
daisy/
  config.py              all command-line flags
  models/                the Llama-style decoder, document-masked attention, the label gates
  data/                  the tokenized pool stream, the candidate-window selector, prefetching, the index store
  labels/                utility labels (gates.py), the selection targets (targets.py), label dumps
  scorers/               the running ridge and the online scorer; the fixed scorers of the baselines
                         (similarity.py, dsir.py, dsir_build.py)
  train/                 the training loop (trainer.py), the online learner (learner.py), checkpoints
  eval/                  held-out losses during training and of saved checkpoints; zero-shot accuracy
  index/                 the embedding index
  tokenize/              the tokenized pool
  targets/               the benchmark target, the FineWeb topic target, the FineWeb-Edu target
  mates/                 the MATES baseline (docs/mates_baseline.md)
  sft/                   instruction tuning (daisy/sft/README.md)
  multilingual/          the multilingual experiments (daisy/multilingual/README.md)
scripts/                 launchers: env.sh (paths), train.sh, runs.sh (the runs of the main experiments), ...
tests/                   CPU tests
docs/mates_baseline.md   specification of the MATES baseline
```

## Installation

Python 3.9 or later and PyTorch 2.6 or later with CUDA (the document-masked attention uses
`torch.nn.attention.flex_attention`, and all training runs are compiled with `torch.compile`).

```bash
pip install -r requirements.txt
```

`requirements.txt` lists the packages by the step that needs them. Training itself needs only `torch`, `numpy` and
`pyarrow`.

## Reproducing the main experiments

The main experiments (Sections 4.2 and 4.3 of the paper) pretrain 124M- and 720M-parameter models on 33.6B tokens of
FineWeb with uniform sampling, DAiSY, and the similarity, DSIR and MATES baselines, with three selection targets: the
training splits of five benchmarks, FineWeb-Edu, and the topics of FineWeb itself. The steps below build the data
once, train the runs and evaluate them. Each tool documents its inputs, outputs and options in its module docstring.

### 0. Paths

All scripts read their locations from `scripts/env.sh`. Set `DATAPATH` there (or export it); every other location
defaults to a subdirectory of it and can be set on its own:

| Variable | Content | Built by |
|---|---|---|
| `EMBEDDED_POOL` | FineWeb parquet files with a document embedding (input) | you, see step 1 |
| `FINEWEB_EDU` | a local copy of FineWeb-Edu, score-2 version (input) | you, see step 2 |
| `INDEX` | the embedding index | `scripts/build_index.sh` |
| `POOL` | the tokenized pool | `scripts/tokenize_pool.sh` |
| `BENCH` | the benchmark target tasks | `scripts/prepare_targets.sh benchmarks` |
| `TARGETS` | the web-text targets | `scripts/prepare_targets.sh fineweb_topics`, `fineweb_edu` |
| `SCORERS` | the fixed scorers of the baselines | `scripts/prepare_targets.sh similarity`, `dsir`, ... |
| `EDU_SCORES` | the FineWeb-Edu score of every pool document | `scripts/prepare_targets.sh edu_scores` |
| `EXPS` | one directory per run | `scripts/train.sh` |
| `EVALS` | evaluations of checkpoints | `scripts/evaluate.sh` |

### 1. Data: the embedding index and the tokenized pool

**Input.** The paper uses FineWeb (the `CC-MAIN-2023-14`, `CC-MAIN-2023-23`, `CC-MAIN-2023-40`, `CC-MAIN-2023-50`,
`CC-MAIN-2024-10` and `CC-MAIN-2024-18` dumps; 3,282 parquet files, 1.77B documents) as training data, but feel free
to define your own data sources, as parquet files in the same format. The pipeline starts from a copy of the files in
which every document carries its embedding: `EMBEDDED_POOL/<snapshot>/<name>.parquet` with the columns `id`, `text`,
`url`, `dump`, `token_count`, `language_score` and `embeddings` (either at the top level or inside a struct column
`metadata`). `embeddings` is the XLM-RoBERTa-base (`FacebookAI/xlm-roberta-base`) embedding of the first chunk of the
document, of at most 512 tokens: the mean of the last hidden states over the tokens of the chunk, as 768 floats.
Computing these embeddings is the one-time cost of the method; this repository starts after it.
`daisy/index/build_index.py` describes the accepted layouts in full, and `daisy/scorers/similarity.py` checks that the
embedding model reproduces the stored vectors.

**Index** (`scripts/build_index.sh`). The index holds 100M documents (`TARGET_DOCS`), drawn from the files of all
snapshots of the pool: their embeddings, standardized per coordinate, 16 topics from spherical k-means, and the
held-out FineWeb set (2,000 documents per topic; training excludes them and every pool document with the same text or
URL). The runs evaluate the first 64 held-out documents of every topic (1,024 documents, cut at 1,024 tokens); the
rest is the source of the FineWeb topic target.

```bash
bash scripts/build_index.sh          # stages: plan, stats, shard, tokens, kmeans, heldout
```

**Tokenized pool** (`scripts/tokenize_pool.sh`). Every document of the embedded pool as GPT-2 tokens (one shard per
input file), with its embedding, standardized with the statistics of the index, in an aligned sidecar:

```bash
bash scripts/tokenize_pool.sh                     # or in parts: NJOBS=16 TASK=<0..15> bash scripts/tokenize_pool.sh
bash scripts/tokenize_pool.sh verify
```

```
POOL/manifest.json
POOL/tokens/<shard>_unshuffled.ds          uint16 tokens, EOS after every document
POOL/tokens/<shard>_unshuffled.ds.index    uint64 cumulative document ends
POOL/sidecar/<shard>.phi.f16               standardized embeddings, 768 fp16 per document, same order
POOL/sidecar/<shard>.meta.parquet          id, url, dump, text_hash, url_hash, ...
```

Training reads documents from `n` open shards per process in a seeded random order, so the stream is close to uniform
over the snapshots, and it reads the embeddings of the candidates from the sidecars of the open shards.

### 2. Targets and fixed scorers

```bash
bash scripts/prepare_targets.sh benchmarks         # BENCH: ARC-Easy, ARC-Challenge, PIQA, SciQ, HellaSwag
bash scripts/prepare_targets.sh similarity         # the similarity baseline's scorer for the benchmark target
bash scripts/prepare_targets.sh fineweb_topics     # the FineWeb topic target
bash scripts/prepare_targets.sh fineweb_edu        # the FineWeb-Edu target and the held-out FineWeb-Edu set
bash scripts/prepare_targets.sh similarity_fwedu   # the similarity baseline's scorer for the FineWeb-Edu target
bash scripts/prepare_targets.sh edu_scores         # FineWeb-Edu score of every pool document (run uniform-fwedu)
bash scripts/prepare_targets.sh pool_copies        # held-out FineWeb-Edu documents without a copy in the pool
COUNTS=N0,N1,N2,N3 bash scripts/prepare_targets.sh dsir    # the DSIR baseline's importance weights
```

- **Benchmark target** (`daisy/targets/prepare_benchmarks.py`). A target task is a directory with `train.bin` /
  `train.len`: the token ids of its items and their lengths. For a benchmark, an item is the question followed by its
  correct answer, from the benchmark's training split. The target loss is the mean over the five tasks of the mean
  token loss on 32,768 tokens of each (15 packed rows). `val.bin` / `val.len` hold the benchmark's test split
  (validation for HellaSwag) and are used only for the held-out downstream loss.
- **FineWeb topic target** (`daisy/targets/build_fineweb_target.py`). 16 tasks `fineweb_t00..15`, one per topic, from
  held-out FineWeb documents that are never evaluated.
- **FineWeb-Edu target** (`daisy/targets/build_fwedu_target.py`). 16 tasks `fwedu_t00..15` of FineWeb-Edu documents
  with an educational score of at least 4 from a crawl snapshot that is not in the pool (`EDU_DUMP`, default
  `CC-MAIN-2023-06`; choose one that your pool does not contain), without text or URL duplicates of index documents
  and without overlap with evaluation items; and 1,024 held-out documents from the same source (`fwedu_theldout`).
  `daisy/targets/pool_copies.py` finds the copies of these documents in the pool and writes the held-out documents
  without any copy as `fwedu_theldout_nocopy`. `FINEWEB_EDU` is a local copy of the score-2 version of FineWeb-Edu
  (`<snapshot>/*.parquet` or `data/<snapshot>/*.parquet`).
- **Similarity scorer** (`daisy/scorers/similarity.py`). The centroid of the target items in the embedding space of
  the index; the score of a candidate is its cosine to the centroid.
- **DSIR** (`daisy/scorers/dsir_build.py`). The log importance weight of DSIR's hashed n-gram model (unigrams and
  bigrams of GPT-2 tokens in 10,000 buckets; target = the items of the benchmark target, raw = a uniform sample of
  the pool), precomputed for the pool shards that a run opens. `COUNTS` is the number of shards each process opens;
  the candidate windows do not depend on the scorer, so it is read from a finished run with the same seed, GPU count
  and window factor: the `next_shard` (under `"base"`) of every `EXPS/124M-similarity-l4/ckpts/latest/stream_r<rank>.json`,
  in rank order.

### 3. Training

`scripts/train.sh` starts one run with the recipe of the paper, and `scripts/runs.sh` holds the flags of every run of
the main experiments:

```bash
bash scripts/runs.sh list                 # the names and flags of all runs
bash scripts/runs.sh run uniform          # -> EXPS/124M-uniform
bash scripts/runs.sh run daisy-l4         # DAiSY, lambda = 4, benchmark target -> EXPS/124M-daisy-l4
bash scripts/runs.sh run daisy-l16 720M   # the same at 720M, lambda = 16 -> EXPS/720M-daisy-l16
```

| Setting | Model | Steps x tokens per step | GPUs | Permanent checkpoints |
|---|---|---|---|---|
| `124M` | 12 layers, width 768, 12 heads | 256,000 x 131,072 | 4 (one node) | every 16,000 steps |
| `720M` | 12 layers, width 2,048, 16 heads | 64,000 x 524,288 | 16 (4 nodes x 4) | every 8,000 steps |

Both settings train on 33.6B tokens in packed rows of 2,048 tokens with AdamW (peak learning rate 3e-4, betas 0.8 and
0.999, weight decay 0.1, gradient clipping at 0.5), 2,000 warmup steps and a cosine schedule, under `torch.compile`.
All runs of a setting share the initialization, the document stream and the schedule (`SEED=0`); the stream is split
by process, so the GPU counts above reproduce the streams of the paper.

**Runs of the main comparison (benchmark target, Section 4.2).**

| Run | Method | Flags after the common ones |
|---|---|---|
| `uniform` | uniform sampling | (none) |
| `daisy-l{2,4,8,16,32,64}` | DAiSY with window factor lambda | `--daisy_learn --daisy_select_mult LAMBDA --daisy_target_dir BENCH` |
| `similarity-l{2,...,64}` | cosine to the target centroid, same windows | `--daisy_select_mult LAMBDA --daisy_select_cosine --daisy_select_static SCORERS/similarity/w_static.npy` |
| `dsir-l4` | DSIR importance weights, same windows | `--daisy_select_mult 4 --daisy_select_dsir SCORERS/dsir` |
| MATES | stage-wise selection by a BERT influence model | `bash scripts/run_mates.sh`, see [MATES](#mates) |

The OPUS baseline of the paper was run in a separate codebase and is not part of this repository.

At 720M the paper runs `uniform`, `daisy-l{4,16,64}` and `similarity-l{4,16,64}`. At lambda = 64 a process uses up its
share of the pool before the end of training, so these runs add `--daisy_pool_reshuffle` (further passes over the pool
in new orders). Every run evaluates the held-out downstream loss on the five benchmarks
(`--daisy_eval_dir BENCH --daisy_eval_tasks arc_easy,arc_challenge,piqa,sciq,hellaswag`) and the held-out FineWeb loss
every 200 steps.

#### MATES

`scripts/run_mates.sh` runs the MATES baseline at 124M (`daisy/mates/`; `docs/mates_baseline.md` specifies what is
taken from the MATES paper and its released code and where the implementation differs). It trains in stages of 32,000
steps. Stage 1 is uniform training, so the run starts from the permanent checkpoint of the uniform run at step
32,000. At every stage boundary it probes held-out documents (one optimizer step on each from a copy of the model,
then the loss on the target rows), fine-tunes a BERT influence model on the probes, scores a fresh set of candidates
and keeps 20% of them for the next stage. It needs a local copy of `bert-base-uncased` (`BERT` in `scripts/env.sh`).

```bash
bash scripts/runs.sh run uniform        # at least up to step 32,000
bash scripts/run_mates.sh               # -> EXPS/124M-mates
```

With `BUDGET=<GPU-hours>` the run stops when its cost, which counts stage 1, the training steps and every stage
boundary, reaches the budget; the paper compares the methods at an equal budget and then continues MATES to the full
schedule.

**Runs with web-text targets (Section 4.3).** A run with another target is the benchmark-target run with the same
lambda in which only the target changes.

| Run | Change against the benchmark-target run |
|---|---|
| `daisy-l{4,16,64}-fwedu` | `--daisy_targets fwedu_t00,...,fwedu_t15 --daisy_target_dir TARGETS/fineweb_edu` (720M: also `--daisy_no_task_labels`, which keeps only the combined target gradient in memory) |
| `similarity-l{4,16,64}-fwedu` | `--daisy_select_static SCORERS/similarity_fwedu/w_static.npy` |
| `uniform-fwedu` | uniform sampling from the pool documents with a FineWeb-Edu score of at least 3: `--daisy_pool_keep EDU_SCORES --daisy_pool_keep_min 3` |
| `daisy-l{4,16,64}-fwtopics` (124M) | `--daisy_targets fineweb_t00,...,fineweb_t15 --daisy_target_dir TARGETS/fineweb_topics` |

**DAiSY settings.** The defaults of `train.py` are those of the paper: labels on every 4th step from step 2,000
(`--daisy_label_every`, `--daisy_label_start`), target refresh every 200 steps (`--daisy_refresh`), ridge refit every
10 steps (`--daisy_scorer_update_every`), forgetting factor 0.999 (`--daisy_ridge_decay`), regularization 0.01 times
the mean eigenvalue of the Gram matrix (`--daisy_ridge_reg`), scorer start at step 2,500 (`--daisy_scorer_start`),
32,768 target tokens per task (`--daisy_target_tokens`).

**Long runs and time limits.** `scripts/train.sh` is restartable. With `TIME_LIMIT_MIN=n`, a run stops at a refresh
step about ten minutes before `n` minutes have passed and writes a checkpoint; calling the script again with the same
name continues from the latest checkpoint (model, optimizer, schedule, random generators, stream position and the
state of the scorer, so the continued run is identical to an uninterrupted one), and a finished run exits at once. On
Slurm, a run is therefore a chain of identical jobs, for example:

```bash
J=$(sbatch --parsable --nodes=1 --gpus-per-node=4 --time=12:00:00 \
      --wrap "TIME_LIMIT_MIN=720 bash scripts/runs.sh run daisy-l4")
sbatch --dependency=afterany:$J --nodes=1 --gpus-per-node=4 --time=12:00:00 \
      --wrap "TIME_LIMIT_MIN=720 bash scripts/runs.sh run daisy-l4"
```

For a 720M run, request 4 nodes (`--nodes=4 --gpus-per-node=4`); `scripts/train.sh` then starts one `torchrun` per
node through `srun`.

**Other runs.** `NAME=... SIZE=... EXTRA="..." bash scripts/train.sh` starts a run with any flags; `ITERS` and `TOTAL`
set the number of steps and the horizon of the schedule, `SEED` the seed. `python train.py --help` documents every
flag.

### 4. Evaluation

**During training.** Every 200 steps a run writes the held-out FineWeb loss (`EXPS/<run>/daisy/refresh.jsonl`) and
the held-out downstream loss, the mean token loss on the held-out splits of the five benchmarks
(`metrics.jsonl`, records of kind `bench_val`). Neither feeds back into training.

**Zero-shot accuracy** with lm-evaluation-harness 0.4.13 on the five target benchmarks and on WinoGrande, OpenBookQA
and LAMBADA, at the permanent checkpoints:

```bash
bash scripts/evaluate.sh accuracy 124M-uniform 124M-daisy-l4          # the final checkpoints
bash scripts/evaluate.sh accuracy_all 124M-uniform 124M-daisy-l4      # every permanent checkpoint
```

The results are `EVALS/lm_eval/<run>[_ck<step>].json` with `acc`, `acc_norm` and their standard errors per task.

**Held-out FineWeb-Edu loss** of the checkpoints of runs (Section 4.3), on all held-out FineWeb-Edu documents and on
those without a copy in the pool:

```bash
bash scripts/evaluate.sh fwedu 124M-uniform 124M-uniform-fwedu 124M-daisy-l4-fwedu 124M-similarity-l4-fwedu
```

**Quantities of the paper.** Delta is the held-out downstream loss of a run minus that of the uniform run of the same
setting, averaged over the last 10% of the steps. The token ratio is the number of tokens of the uniform run divided
by the number of tokens after which the downstream loss of a run first reaches the final loss of the uniform run.
Both are computed from the `bench_val` records. GPU-hours are steps times the run's measured time per step
(`timing.json`) times its number of GPUs.

## Outputs of a run

```
EXPS/<run>/
  daisy/
    meta_train.json        configuration, command line, seeds, versions
    refresh.jsonl          every 200 steps: the held-out FineWeb loss ("target_loss"), its per-topic vector ("ell"),
                           the learning rate and the number of trained tokens
    metrics.jsonl          records by "kind":
                             bench_val   the held-out downstream loss per benchmark and their mean (every 200 steps)
                             targets     the loss of every target task at a refresh (DAiSY)
                             scorer      how well the ridge ranks new documents: Spearman and Pearson correlation
                                         with the labels, precision of the top quarter (DAiSY)
                             tasks       rank agreement between the target tasks, and of the ridge with each (DAiSY)
    labels/labels_*.npz    the labels of every labelled document: step, key (pool shard << 32 | row), y, y_tasks (DAiSY)
    timing.json            per-process step times and peak memory; step_times_r<rank>.npy: every step
    finished.json          written when the run has reached its last step
  ckpts/latest/, ckpts/<step>/
    main.pt                model, optimizer, schedule, step
    worker_<rank>.pt       random generator states
    stream_r<rank>.json    position of the document stream and the selector's weight versions
    daisy_r<rank>.pt       state of the scorer (and of MATES)
  summary.json
```

The two losses of the paper are `target_loss` in `refresh.jsonl` (held-out FineWeb loss) and `loss/mean` of the
`bench_val` records in `metrics.jsonl` (held-out downstream loss).

## Further experiments

- **Instruction tuning** (Section 4.4): `daisy/sft/README.md`. Tulu 3 data preparation, the tuning of pretrained
  checkpoints, and the evaluation with plain prompts and with the chat template (`daisy/eval/lm_eval_chat.py`).
- **Multilingual selection** (Section 4.5): `daisy/multilingual/README.md`. The FineWeb2 pool with mmBERT embeddings,
  the Global-MMLU and INCLUDE targets, the high-quality FineWeb2 target, and the evaluation on held-out items.
- **Selection from a limited pool** (Section 4.6): add `--daisy_pool_subset_tokens 10000000000` to a run. The stream
  then trains on a fixed subset of the pool with that many tokens (the first documents of the stream) and passes
  over it again in new orders; `--daisy_ridge_decay` sets the forgetting factor of the ridge.

## Tests

The tests run on CPU with a tiny model and synthetic data:

```bash
cd tests
python -m unittest test_training      # train.py end to end: uniform, DAiSY, similarity; exact resume; chained jobs
python -m unittest test_label_gates   # the labels equal the exact per-document inner products; training is unchanged
python -m unittest discover -p "test_*.py"
```

Tests of tools that need an optional package (transformers, tokenizers, datasets, lm-eval) are skipped when it is
missing.

## License

This code is released under the MIT License (`LICENSE`).
