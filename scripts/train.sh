#!/bin/bash
# One training run with the recipe of the paper. The run is chosen by EXTRA, the flags after the common ones
# (scripts/runs.sh holds the EXTRA of every run of the main experiments; `python train.py --help` lists all flags).
#
#   NAME=124M-uniform  EXTRA="--daisy_eval_dir $BENCH"                                    bash scripts/train.sh
#   NAME=124M-daisy-l4 EXTRA="--daisy_learn --daisy_select_mult 4 --daisy_target_dir $BENCH --daisy_eval_dir $BENCH" \
#                                                                                          bash scripts/train.sh
#
# SIZE=124M (default): 12 layers, width 768, 12 heads; 256,000 steps of 64 rows x 2,048 tokens (131,072 tokens) on
#                      4 GPUs of one node; permanent checkpoints every 16,000 steps.
# SIZE=720M:           12 layers, width 2,048, 16 heads; 64,000 steps of 256 rows (524,288 tokens) on 16 GPUs
#                      (4 nodes x 4); permanent checkpoints every 8,000 steps.
# Both train on 33.6B tokens with AdamW (peak learning rate 3e-4, betas 0.8 / 0.999, weight decay 0.1, clipping 0.5),
# 2,000 warmup steps and a cosine schedule, compiled with torch.compile.
#
# Other variables: SEED (0), GPUS_PER_NODE (4), NODES (1, or the Slurm allocation's), ITERS / TOTAL (steps of this run /
# horizon of the schedule: ITERS < TOTAL trains the first ITERS steps of the longer schedule), PERM, CKPT (latest
# checkpoint every 2,000 steps), VOCAB (50304).
# The global batch is fixed (ROWS rows per step), so other GPU counts train on the same number of tokens per step; the
# document stream is split by process, so the paper's runs are reproduced with the GPU counts above.
#
# Time limits: with TIME_LIMIT_MIN=n the run stops cleanly at a refresh step about 10 minutes before n minutes have
# passed, with a checkpoint. Calling the script again with the same NAME continues the run from its latest checkpoint;
# a finished run (EXPS/NAME/daisy/finished.json) exits at once. So a long run is a chain of identical jobs.
set -eo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"
cd "$HERE/.."

NAME=${NAME:?set NAME} SIZE=${SIZE:-124M} SEED=${SEED:-0} EXTRA=${EXTRA:-}
case "$SIZE" in
  124M) N_EMBD=768  N_HEAD=12 N_LAYER=12 ROWS=64  ITERS=${ITERS:-256000} PERM=${PERM:-16000} ;;
  720M) N_EMBD=2048 N_HEAD=16 N_LAYER=12 ROWS=256 ITERS=${ITERS:-64000}  PERM=${PERM:-8000} ;;
  *) echo "unknown SIZE=$SIZE (124M or 720M)"; exit 1 ;;
esac
TOTAL=${TOTAL:-$ITERS} CKPT=${CKPT:-2000} VOCAB=${VOCAB:-50304}
GPUS_PER_NODE=${GPUS_PER_NODE:-4} NODES=${NODES:-${SLURM_NNODES:-1}}
BATCH=16                      # rows per micro-batch
ACC=$((ROWS / BATCH))         # micro-batches per step over all GPUs (train.py divides them among the processes)

if [ -s "$EXPS/$NAME/daisy/finished.json" ]; then
  echo "run $NAME has finished ($(cat "$EXPS/$NAME/daisy/finished.json")): nothing to do"; exit 0
fi
if [ -n "${TIME_LIMIT_MIN:-}" ]; then
  export DAISY_STOP_AT=$(( $(date +%s) + 60 * TIME_LIMIT_MIN - ${STOP_MARGIN_S:-600} ))
fi
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} PYTORCH_ALLOC_CONF=expandable_segments:True
# compile caches per job and on a node-local file system
export TRITON_CACHE_DIR=${TMPDIR:-/tmp}/daisy_triton_${SLURM_JOB_ID:-$$} TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/tmp}/daisy_inductor_${SLURM_JOB_ID:-$$}

if [ "$NODES" -gt 1 ]; then
  # several nodes: one torchrun per node, started by Slurm; on another scheduler, start the same command on every node
  [ -n "${SLURM_JOB_ID:-}" ] || { echo "NODES > 1 needs a Slurm allocation (or adapt the launch line below)"; exit 1; }
  MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
  RUN="srun --ntasks-per-node=1 torchrun --nnodes=$NODES --rdzv_backend=c10d --rdzv_endpoint=$MASTER:29500 --rdzv_id=$SLURM_JOB_ID"
else
  RUN="torchrun --standalone"
fi
echo "NAME=$NAME SIZE=$SIZE SEED=$SEED ITERS=$ITERS TOTAL=$TOTAL GPUs=${NODES}x${GPUS_PER_NODE} EXTRA=$EXTRA"
$RUN --nproc_per_node=$GPUS_PER_NODE train.py --distributed_backend nccl --compile \
  --experiment_name "$NAME" --results_base_folder "$EXPS" --seed $SEED \
  --n_embd $N_EMBD --n_head $N_HEAD --n_layer $N_LAYER --vocab_size $VOCAB --sequence_length 2048 \
  --batch_size $BATCH --acc_steps $ACC --iterations $ITERS --scheduler_total_steps $TOTAL \
  --lr 3e-4 --beta1 0.8 --beta2 0.999 --weight_decay 0.1 --grad_clip 0.5 --warmup_steps 2000 --scheduler cos \
  --latest_ckpt_interval $CKPT --permanent_ckpt_interval $PERM \
  --daisy_index_dir "$INDEX" --daisy_pool "$POOL" --daisy_pool_open_shards 32 --daisy_max_doc_tokens 2048 \
  --daisy_refresh 200 --daisy_val_docs_per_topic 64 --daisy_val_max_doc_tokens 1024 \
  $EXTRA
