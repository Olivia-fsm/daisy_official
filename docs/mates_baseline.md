# The MATES baseline

MATES (Yu et al., "MATES: Model-Aware Data Selection for Efficient Pretraining with Data Influence Models",
arXiv 2406.06046) reimplemented inside the DAiSY trainer (`--mates`; the code is `daisy/mates/`). **The selection
method follows the MATES paper** (oracle probing, a BERT-base data influence model, Gumbel-top-k selection per stage);
**everything about the language model follows the trainer** (model, AdamW and its schedule, the tokenized pool, whole
documents capped at `--daisy_max_doc_tokens` with document-masked attention, the target tasks as the reference task,
the evaluations). A MATES run therefore differs from the uniform run only in the documents it trains on after stage 1.
For comparisons at equal compute the run can stop at a GPU-hour budget that counts every MATES cost (training steps,
probing, fine-tuning, scoring) while the learning-rate schedule keeps its full horizon.

Where the MATES paper and its released code (github.com/cxcscmu/MATES) differ, the tables below say which one is
followed. This file is the specification of what is implemented.

## The algorithm (MATES paper, Alg. 1)

Training runs in stages of U steps. Stage 1 is uniform. At each stage boundary:

1. **Oracle probing.** For n held-out documents x (80,000 at the first boundary, 20,000 at later ones; fresh documents
   each time), the oracle is the change of the reference loss after one optimizer step on x alone:
   o(x) = L_r(theta_t) - L_r(A(theta_t, x)), where A is one AdamW step from the current model **and optimizer state**
   (moments, step count) at the current scheduled learning rate, with the gradient clipped at the training clip. The
   batch is one document. The state is restored after every probe.
2. **Data influence model.** BERT-base (bert-base-uncased) regresses the z-scored oracle (mean and standard deviation
   of the training split; higher = better) with a mean-squared error: the document is decoded from GPT-2 tokens to
   text, WordPiece-encoded to at most 1024 tokens, run as two 512-token chunks (the second without [CLS], positions
   restarting), mean-pooled over the non-pad tokens of both chunks, then dropout 0.1 and Linear(768, 1). Fine-tuning:
   a 90/10 split, 5 epochs, global batch 256, learning rate 5e-5, weight decay 0.01, linear decay without warmup,
   bf16; validation every 20 steps, and the model with the best validation Spearman correlation is kept. The first
   boundary starts from pretrained BERT; every later boundary **continues from the previous stage's best model** on
   that stage's new probes.
3. **Selection.** A fresh candidate pool holding 5x the stage's training tokens is scored by the influence model, and
   20% of the candidate tokens are kept by **Gumbel-top-k at tau = 1**: sort by s(x) / tau + G with G ~ Gumbel(0, 1)
   iid, take documents until the stage's token budget is met (the last one is cut), shuffle, and train the stage on
   that list in order.

## Settings

| Setting | Here | Source |
|---|---|---|
| Stage length U | `--mates_stage_steps` (32,000) | MATES: 10,000 steps x 524k tokens = 5.24B tokens per stage, and its paper reports no difference between 1.3B and 5.2B tokens per stage; 32,000 steps are 4.19B tokens at 64 rows x 2,048 tokens per step and divide the 256,000-step schedule |
| Stage 1 | the uniform pool stream: trained in the run, or taken from the uniform run's permanent checkpoint (`--mates_init_from`: model, optimizer, schedule, random generators, stream position) | the two are bit-identical (same seed, stream and model); the cost of the borrowed steps is counted in the budget |
| First boundary F | `--mates_first_boundary` (0 = U): stage 1 is the steps before F, the boundaries are at F, F + U, F + 2U, ... | F = U is the MATES paper's; a small F (for example DAiSY's `--daisy_scorer_start`, 2,500) starts the selection where DAiSY's starts, with every later stage still U steps |
| Probes | `--mates_probes_first` (80,000) at the first boundary, `--mates_probes_later` (20,000) at each later one, split evenly over the ranks | paper (the released code uses 80,000 every time) |
| Probe documents | random documents of **reserved pool shards** (`--mates_probe_shards`, 8: the last shards of every rank's shard list, which the training stream never opens), held-out documents excluded by hash, capped like training, fresh per stage, seeded by (seed, stage, rank) | paper: disjoint held-out data from the training distribution |
| Probe step | one step of the run's own optimizer on one document, in a (1, T) row; the loss is the document's mean token loss; clipped at `--grad_clip`; the learning rate of the boundary's step; from a GPU snapshot of the parameters and the AdamW state, restored after each probe; on the unwrapped model | released code: model and AdamW state reloaded for every probe, the current learning rate, batch 1, the training clip |
| Reference loss L_r | the mean over the target tasks (`--daisy_targets` under `--daisy_target_dir`) of their mean token loss on their train-split rows: DAiSY's combined target | the same reference task as every other method of the comparison (MATES uses LAMBADA) |
| Probe precision | `--mates_probe_precision bf16` (the training autocast) or `fp32` (no autocast, TF32 off), for the document step and the reference forward | bf16 is training's arithmetic; fp32 is there for learning rates at which bf16 rounding is not small against the oracle |
| Influence model | bert-base-uncased from `--mates_bert_dir`; `--mates_bert_epochs` (5), `--mates_bert_batch` (256), `--mates_bert_lr` (5e-5), `--mates_bert_wd` (0.01), `--mates_bert_val_share` (0.1), `--mates_bert_eval_every` (20) | paper and released code |
| Continuation | every boundary after the first fine-tunes the previous stage's best model on the new probes only | paper (the released code starts from pretrained BERT every time) |
| Candidates | every rank draws documents from its live pool stream until they hold 1 / `--mates_ratio` (0.2) = 5x its stage token budget; the stream never reopens a shard, so no document is a candidate twice | selecting 20% is MATES |
| Selection | Gumbel-top-k by tokens with `--mates_tau` (1.0), per rank (each rank's candidates are an iid sample of the pool), seeded | paper (the released code takes the k smallest of predicted loss / tau + G with tau = 0.5) |
| Budget stop | `--mates_budget` GPU-hours (0 = off): the run ends like a finished one when stage 1 + its training steps + every boundary reach it | comparison at equal GPU-hours; see "The GPU-hour budget" |
| Schedule | unchanged: the horizon is `--scheduler_total_steps`, whether or not the budget stops the run earlier | the same schedule as the uniform run |
| Evaluation | the held-out FineWeb loss and the held-out downstream loss every `--daisy_refresh` steps, permanent checkpoints, step timing files | the same as every run |

Deviations from MATES that the setting makes unavoidable: whole documents instead of 1024-token C4 rows; a token
budget per stage instead of a row count; 5x candidates per trained token (the released configuration amounts to about
7.6 candidates per trained row); the target tasks instead of LAMBADA as the reference; the trainer's cosine schedule
and optimizer recipe (betas 0.8 / 0.999, clip 0.5) instead of MATES's; selection per rank.

## Running it

`--mates` trains on the pool without a selector: it needs `--daisy_pool`, `--daisy_select_mult 0` and no
`--daisy_learn`. On top of the uniform run's flags a MATES run passes

```
--mates --mates_bert_dir BERT --daisy_targets TASKS --daisy_target_dir BENCH
[--mates_init_from EXPS/UNIFORM/ckpts/STEP] [--mates_first_boundary F] [--mates_budget GPU_HOURS]
[--mates_boundary_reserve_min MINUTES]
```

(`scripts/run_mates.sh` is the launcher). It needs, beforehand:

- **bert-base-uncased** in a local directory (`--mates_bert_dir`): `config.json`, the weights, and `tokenizer.json` or
  `vocab.txt`; and the Python packages `transformers` and `tokenizers`.
- **The GPT-2 tokenizer's `tokenizer.json`** (to decode the pool's tokens): `--mates_gpt2_tokenizer`, or by default
  the GPT-2 tokenizer in the Hugging Face hub cache (`daisy/tok.py`).
- **The target tasks** (`train.bin` / `train.len` under `--daisy_target_dir`), as for a DAiSY run.
- With `--mates_init_from`: **a permanent checkpoint of the uniform run** at the step of the first boundary, from a
  run with the same model, batch, optimizer, schedule, pool, seed and number of GPUs (checked against its
  `daisy/meta_train.json`; the run refuses to start otherwise). The flag is read only while the MATES run has no
  checkpoint of its own. The stage-1 cost is taken from that run's `daisy/step_times_r0.npy` and `daisy/timing.json`,
  which it writes when a job of it ends; without them pass `--mates_cost_offset`.

Three configurations: (a) stage 1 from the uniform run's checkpoint at U, with a budget; (b) `--mates_first_boundary`
F without `--mates_init_from`: stage 1 is trained in the run and counted as its training steps; (c) no budget
(`--mates_budget 0`): the run trains to `--iterations`. A run that stopped at its budget is continued without it by
copying its `ckpts/latest` and `daisy/` to a new experiment directory, removing `daisy/finished.json` there and
running the same command on it with `--mates_budget 0`; the result is the run without a budget (the resume is exact).

A boundary keeps the ranks busy without collectives while they probe and score, and a rank that finishes first waits
in the next gather for the slowest: set `DAISY_NCCL_TIMEOUT_MIN` (default 10) above the minutes a boundary's probing
and scoring take. The pool filters `--daisy_pool_keep` and `--daisy_pool_exclude` act on the training stream and the
candidates, not on the probe documents; `--daisy_pool_subset_tokens` cannot be combined with reserved shards.

## The GPU-hour budget

GPU-hours = offset + training + boundaries, with `world` the number of GPUs:

- **offset**: the cost of the steps taken from another run. `--mates_cost_offset` if >= 0; else, with
  `--mates_init_from`, the source run's steady seconds per step x the checkpoint's step x its GPUs / 3600 (steady:
  the mean of rank 0's recorded step times, the first min(50, n / 4) steps and steps over 2 s left out); else 0.
- **training**: the steps trained in this run x the mean of its step times of at most 2 s x world / 3600. The sum and
  count of the step times are checkpointed, so chained jobs accumulate them.
- **boundaries**: per boundary, the sum over its components (probe, gather, finetune, enumerate, score, select) of the
  slowest rank's wall seconds x world / 3600. Artifacts reused after an interruption count with the seconds recorded
  when they were made. Checkpoint writes and evaluations lie outside the step times and are not counted.

The budget is checked at refresh steps (every `--daisy_refresh`) and right after each boundary; the decision is the
same on every rank (the totals are all-reduced with MAX). When it is spent the run writes the latest and a permanent
checkpoint at that step, runs that step's evaluation and writes `finished.json` with `"reason": "budget"`; a job
started again on the run then does nothing. Before every boundary after the first, the boundary's cost is estimated
from the previous one's GPU-seconds per component (probing and fine-tuning scaled by the probe counts); when the total
plus that estimate reaches the budget the run stops at that step instead (`finished.json` records
`stopped_before_boundary`): the next stage could not train a step within the budget, so the loss at the budget is the
same and the selection's GPU time is not spent. `budget.json` is rewritten at every check.

## Outputs (EXP/daisy/mates/)

Stage s is the stage a boundary selects for: the first boundary writes `stage_2/`, the next `stage_3/`, ...

| File | Contents |
|---|---|
| `stage_{s}/probes_r{rank}.npz` | per probe: `key`, `length`, `grad_norm` (before clipping), `clipped`, `loss_after`, `doc_loss`, `seconds`; and `loss_before`, `lr`, `wall_s` |
| `stage_{s}/probe_meta.json` | `loss_before` (rank 0, and per rank), `lr`, `precision`, `clip`, the probe counts, rank 0's reserved shards |
| `stage_{s}/bert/` | `model.pt` (the best influence model's state dict), `config.json` |
| `stage_{s}/bert_log.jsonl` | every validation: step, learning rate, training MSE, validation MSE / MAE / Pearson / Spearman |
| `stage_{s}/bert_done.json` | the split sizes, the z-score mean and standard deviation, the best step and its metrics, the model it started from, seconds (the completion marker of the fine-tuning) |
| `stage_{s}/select_r{rank}.npz` | `cand_key`, `cand_len`, `score`, `gumbel`, `selected` (per candidate, in draw order); `sel_key`, `sel_end` (the selected list in training order: document key and the number of its tokens trained) |
| `stage_{s}/select_r{rank}.json` | the pool-stream state after the candidate draw, the rank's seconds and summary statistics (the completion marker of the selection) |
| `stages.jsonl` | per boundary: oracle mean / standard deviation / positive share, clip rate, probe learning rate, validation Spearman, predicted-score mean and standard deviation, selected-minus-candidate mean score, selected document and token shares |
| `stage_costs.jsonl` | per boundary: wall seconds (slowest rank) and GPU-seconds of every component, the tokenization's CPU seconds and the seconds the GPU waited for it |
| `budget.json` | the GPU-hour total at the last check and its parts |

The checkpoints hold the stream state (`stream_r{rank}.json`: the stage, the pool-stream position, the position in
the stage list) and the cost state (`daisy_r{rank}.pt`, entry `"mates"`).

## Implementation details

- **Probe documents.** Each rank probes n / world of the stage's probes (the remainder on the low ranks). Stage s draws
  uniformly without replacement, seeded by (stream seed, s, rank), from the documents of the rank's reserved shards
  that are not held-out (the stream's hash exclusion), have at least 2 tokens (one target), and were not drawn by an
  earlier stage (earlier draws are replayed from their seeds, so nothing is stored). The stream never opens the
  reserved shards (`ShardDocStream(reserve=...)`); a stream state that opened one is refused, and every candidate is
  checked against them.
- **Probe step.** One (1, T) row: inputs doc[:-1], targets doc[1:], the pads with target -1 and their own document
  id; the loss is the document's mean token loss; `clip_grad_norm_` at `--grad_clip` over the unwrapped model's
  parameters (as training); `optimizer.step()` with the parameter groups as the scheduler left them for the boundary's
  step (the scheduler is not stepped); the parameters and the optimizer state (moments, step count) restored
  bit-exactly from GPU copies. With `--compile` the probe and the reference batches have static shapes and compile
  once each.
- **Reference loss.** The target tasks' train-split rows as DAiSY's selection target packs them (`target_batches`
  with `--daisy_target_tokens` per task), `--mates_ref_rows_per_batch` (5) rows per forward, the task as the topic of
  `validation_pass`; token losses in fp32 (as every evaluation), summed and returned in float64: a float32 L_r has a
  resolution of about 2e-7 at L_r ~ 3, which would quantize the small oracles of late stages with a low learning rate.
- **Influence model.** `BertModel` without the pooler (MATES computes it and never uses it); the mean over the
  non-pad tokens of both chunks and the head in fp32; dropout from the BERT config (0.1); head weights N(0, 0.02),
  bias 0, seeded identically on every rank. One model object per run (its compiled graphs serve every boundary),
  loaded with the pretrained weights (first boundary) or the previous stage's best model.
- **Fine-tuning.** The Hugging Face Trainer defaults that the released code runs with: AdamW betas (0.9, 0.999),
  eps 1e-8, no weight decay on biases and LayerNorm weights, gradient norm clipped at 1.0, linear decay to 0 without
  warmup over epochs x ceil(n_train / batch) steps; the split of `datasets.train_test_split(test_size=0.1, seed=1234)`
  (the validation share rounded up); the z-score with the population standard deviation. Training uses MATES's padded
  2 x 512 layout (one compiled shape), `--mates_bert_micro_batch` (32) documents per rank and forward, every backward
  all-reduced (the accumulated gradient is the global batch's). Validation every `--mates_bert_eval_every` steps and
  also at the last step (the Trainer evaluates only at multiples of the interval); the best model is the first with
  the strictly highest Spearman correlation (average ranks); if no validation has a finite one, the last model.
- **Scoring.** CPU threads (`--mates_tok_workers`; default: the rank's share of the cores minus 2) read the
  candidates in key order with `os.pread`, decode and WordPiece-encode them (the `tokenizers` calls release the GIL;
  one core per call); the GPU loop consumes their output in order, in bf16. A document's score is linear in the
  summed hidden states of its chunks, so an all-pad second chunk is skipped and every chunk is padded only to its
  length bucket (`--mates_score_bucket`, 64 tokens; `--mates_score_tokens`, 65,536 padded tokens per batch); this
  changes nothing but floating-point rounding. `--mates_score_padded` scores in MATES's layout (every document as
  2 x 512 tokens).
- **Candidates and the stream.** The stage's token budget of a rank is U x its rows per step x (T + 1), and it draws
  round(budget / ratio) candidate tokens with the pool stream's own document draw. At a boundary the prefetcher's
  read-ahead is dropped (the stream is rewound to the last micro-batch handed out), and the document that stage 1 had
  carried into the next micro-batch is dropped. A stage list holds exactly the stage's tokens; plans made past its end
  are empty.
- **Selection.** Seeded by (stream seed, s, rank): the Gumbel noise first, then the shuffle of the selected list.
- **Boundaries and resume.** Before a boundary the trainer writes a latest checkpoint (unless it just did), after it
  another one (the stream then plays the new stage). Every artifact is written atomically and reused when present; the
  boundary restores the global random generator states, so training after it is the same whether the boundary ran in
  this job or its artifacts were reused. With a time limit (`DAISY_STOP_AT`), a job starts a boundary only with
  `--mates_boundary_reserve_min` minutes left (default 0; set it to the expected length of the first boundary), else
  it stops before it.
