#!/bin/bash
# Step 2: the tokenized pool (daisy/tokenize/datatrove_tokenize.py): every document of the embedded pool as GPT-2
# tokens, with its standardized embedding in an aligned sidecar. One task and one shard per input parquet file.
#   bash scripts/tokenize_pool.sh                         all files in this job
#   NJOBS=16 TASK=3 bash scripts/tokenize_pool.sh         the 4th of 16 equal parts (e.g. a Slurm array:
#                                                         TASK=$SLURM_ARRAY_TASK_ID); finished tasks are skipped
#   bash scripts/tokenize_pool.sh verify                  check every shard, and 200k documents against the index
# CPU only.
set -eo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"
cd "$HERE/.."
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
WORKERS=${WORKERS:-64} NJOBS=${NJOBS:-1} TASK=${TASK:-0}
if [ "${1:-}" = verify ]; then
  python daisy/tokenize/verify_tokenized.py --out "$POOL" --workers $WORKERS --compare-index "$INDEX"
  exit 0
fi
NT=$(find "$EMBEDDED_POOL" -name "*.parquet" | wc -l)
PER=$(( (NT + NJOBS - 1) / NJOBS )); OFF=$(( TASK * PER )); LOC=$(( PER < NT - OFF ? PER : NT - OFF ))
echo "$NT files; this job: tasks [$OFF, $((OFF + LOC)))"
mkdir -p "$POOL"
python daisy/tokenize/datatrove_tokenize.py --pool "$EMBEDDED_POOL" --stats "$INDEX/stats.npz" --out "$POOL" \
  --tasks $NT --workers $WORKERS --local-tasks $LOC --local-rank-offset $OFF
