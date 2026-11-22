#!/bin/bash
# The MATES baseline (daisy/mates/, docs/mates_baseline.md) at 124M: the uniform run's recipe and data, with the
# training data re-selected at every stage boundary (stages of 32,000 steps; one-step oracle probes, a BERT data
# influence model, Gumbel-top-k selection of 20% of the candidates).
#
#   bash scripts/run_mates.sh                 -> EXPS/124M-mates
#
# Stage 1 is uniform training, so the run starts from the permanent checkpoint of the uniform run at step 32,000
# (INIT, default EXPS/124M-uniform/ckpts/32000; INIT=none trains stage 1 in this run). Prepare first:
#   * the uniform run up to step 32,000 (bash scripts/runs.sh run uniform), with the same SEED and GPU count;
#   * BERT: a local copy of bert-base-uncased (config.json, weights, tokenizer.json or vocab.txt);
#   * the GPT-2 tokenizer in the Hugging Face cache (HF_HOME), or MATES_EXTRA="--mates_gpt2_tokenizer PATH".
# BUDGET (GPU-hours, default 0 = none): the run stops like a finished one when its cost reaches the budget; the cost
# counts the steps of stage 1, the training steps and every stage boundary. The paper compares the methods at an equal
# GPU-hour budget and then continues MATES to the full schedule.
# With TIME_LIMIT_MIN (scripts/train.sh), a job with less than RESERVE_MIN minutes left stops before a stage boundary
# instead of starting it.
set -eo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"

NAME=${NAME:-124M-mates} SEED=${SEED:-0}
INIT=${INIT:-$EXPS/124M-uniform/ckpts/32000}
BUDGET=${BUDGET:-0} STAGE=${STAGE:-32000} FIRST=${FIRST:-0} RESERVE_MIN=${RESERVE_MIN:-150} MATES_EXTRA=${MATES_EXTRA:-}
TASKS=arc_easy,arc_challenge,piqa,sciq,hellaswag
[ -s "$BERT/config.json" ] || { echo "no bert-base-uncased in BERT=$BERT"; exit 1; }
[ "$INIT" = none ] || [ -s "$INIT/main.pt" ] || { echo "no checkpoint in INIT=$INIT (train the uniform run to step $STAGE first)"; exit 1; }
# at a boundary the processes probe and score without collectives and then gather: the first to finish waits for the last
export DAISY_NCCL_TIMEOUT_MIN=${DAISY_NCCL_TIMEOUT_MIN:-60}
NAME=$NAME SIZE=124M SEED=$SEED EXTRA="--mates --mates_bert_dir $BERT --mates_stage_steps $STAGE \
  --mates_first_boundary $FIRST $([ "$INIT" = none ] || echo --mates_init_from $INIT) --mates_budget $BUDGET \
  --mates_boundary_reserve_min $RESERVE_MIN --daisy_targets $TASKS --daisy_target_dir $BENCH \
  --daisy_eval_dir $BENCH --daisy_eval_tasks $TASKS $MATES_EXTRA" bash "$HERE/train.sh"
