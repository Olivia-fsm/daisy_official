#!/bin/bash
# Step 5: evaluation of saved checkpoints.
#
#   bash scripts/evaluate.sh accuracy RUN[@STEP] ...     zero-shot accuracy with lm-evaluation-harness on the five
#                                                        target benchmarks and on WinoGrande, OpenBookQA and LAMBADA
#                                                        -> EVALS/lm_eval/RUN[_ckSTEP].json  (one GPU per run,
#                                                        GPUS at a time; finished evaluations are skipped)
#   bash scripts/evaluate.sh accuracy_all RUN ...        the same for every permanent checkpoint of the runs
#   bash scripts/evaluate.sh fwedu RUN ...               held-out FineWeb-Edu loss of every permanent checkpoint, on all
#                                                        held-out documents and on those without a pool copy
#                                                        -> EVALS/heldout/RUN.json
# RUN is the name of a directory under EXPS (e.g. 124M-daisy-l4); without @STEP the latest checkpoint is evaluated.
set -eo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"
cd "$HERE/.."
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
GPUS=${GPUS:-4}
TASKS=${TASKS:-arc_easy,arc_challenge,piqa,sciq,hellaswag,winogrande,openbookqa,lambada_openai}
MODE=${1:-}; shift || true

complete() { [ -s "$1" ] || return 1; for t in ${TASKS//,/ }; do grep -q "\"$t\"" "$1" || return 1; done; }

case "$MODE" in
  accuracy_all)
    ALL=""
    for r in "$@"; do
      for s in $(ls "$EXPS/$r/ckpts" | grep -E '^[0-9]+$' | sort -n); do ALL="$ALL $r@$s"; done
    done
    exec bash "${BASH_SOURCE[0]}" accuracy $ALL ;;
  accuracy)
    OUT=$EVALS/lm_eval; mkdir -p "$OUT"; i=0
    for e in "$@"; do
      r=${e%@*}; ck=latest; o=$r
      if [ "$e" != "$r" ]; then ck=${e#*@}; o=${r}_ck$ck; fi
      if complete "$OUT/$o.json"; then echo "skip $o: $OUT/$o.json has every task"; continue; fi
      CUDA_VISIBLE_DEVICES=$((i % GPUS)) python daisy/eval/lm_eval_run.py --run "$EXPS/$r" --ckpt $ck \
        --out "$OUT/$o.json" --tasks $TASKS --batch_size 64 > "$OUT/$o.log" 2>&1 &
      i=$((i + 1))
      if [ $((i % GPUS)) -eq 0 ]; then wait; fi
    done
    wait
    for e in "$@"; do
      r=${e%@*}; o=$r; [ "$e" != "$r" ] && o=${r}_ck${e#*@}
      echo "== $o"; grep "\[lm_eval\]" "$OUT/$o.log" 2>/dev/null || tail -5 "$OUT/$o.log" 2>/dev/null || true
    done ;;
  fwedu)
    FE=$TARGETS/fineweb_edu
    python daisy/eval/heldout_ckpt.py --exps "$EXPS" --runs "$(echo "$@" | tr ' ' ,)" \
      --tasks "fwedu=$FE/fwedu_theldout,fwedu_nocopy=$FE/fwedu_theldout_nocopy" \
      --check_bench "arc_easy=$BENCH/arc_easy" --out "$EVALS/heldout" ;;
  *) sed -n '2,14p' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
