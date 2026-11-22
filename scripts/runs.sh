#!/bin/bash
# The runs of the main experiments (paper Sections 4.2 and 4.3), by name. Every selecting run differs from the others
# of its block only in the scorer, and every run of a size shares the initialization, the document stream and the
# schedule.
#
#   bash scripts/runs.sh list                      the names and flags of all runs
#   bash scripts/runs.sh run NAME [SIZE]           train one run (SIZE: 124M, the default, or 720M) -> EXPS/SIZE-NAME
#   bash scripts/runs.sh flags NAME [SIZE]         print its EXTRA flags only
#
# Benchmark target (Section 4.2)
#   uniform                  uniform sampling
#   daisy-l{2,4,8,16,32,64}  DAiSY with window factor lambda               (720M: lambda 4, 16, 64)
#   similarity-l{2,...,64}   the similarity baseline in the same windows   (720M: lambda 4, 16, 64)
#   dsir-l4                  the DSIR baseline in the same windows          (124M only)
#   mates                    the MATES baseline: bash scripts/run_mates.sh  (124M only)
# FineWeb-Edu target (Section 4.3)
#   daisy-l{4,16,64}-fwedu, similarity-l{4,16,64}-fwedu
#   uniform-fwedu            uniform sampling from the pool documents that are in FineWeb-Edu (score >= 3)
# FineWeb topic target (Section 4.3, 124M)
#   daisy-l{4,16,64}-fwtopics
set -eo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"

BENCH_TASKS=arc_easy,arc_challenge,piqa,sciq,hellaswag
EVAL="--daisy_eval_dir $BENCH --daisy_eval_tasks $BENCH_TASKS"     # the held-out downstream loss, in every run
tasks() { seq -f "$1%02g" 0 15 | paste -sd, -; }                  # the 16 tasks of a web-text target

flags() {   # NAME SIZE -> the EXTRA flags of scripts/train.sh
  local name=$1 size=${2:-124M} lam="" tgt="" f=""
  [[ $name =~ -l([0-9]+) ]] && lam=${BASH_REMATCH[1]}
  case "$name" in
    *-fwedu)    tgt="--daisy_targets $(tasks fwedu_t) --daisy_target_dir $TARGETS/fineweb_edu" ;;
    *-fwtopics) tgt="--daisy_targets $(tasks fineweb_t) --daisy_target_dir $TARGETS/fineweb_topics" ;;
    *)          tgt="--daisy_targets $BENCH_TASKS --daisy_target_dir $BENCH" ;;
  esac
  case "$name" in
    uniform)        f="" ;;
    uniform-fwedu)  f="--daisy_pool_keep $EDU_SCORES --daisy_pool_keep_min 3" ;;
    daisy-l*)       f="--daisy_learn --daisy_select_mult $lam $tgt"
                    # with 16 target tasks, a 720M model keeps only the combined target reference at a refresh
                    [[ $size == 720M && $name == *-fw* ]] && f="$f --daisy_no_task_labels" ;;
    similarity-l*-fwedu) f="--daisy_select_mult $lam --daisy_select_cosine --daisy_select_static $SCORERS/similarity_fwedu/w_static.npy" ;;
    similarity-l*)  f="--daisy_select_mult $lam --daisy_select_cosine --daisy_select_static $SCORERS/similarity/w_static.npy" ;;
    dsir-l*)        f="--daisy_select_mult $lam --daisy_select_dsir $SCORERS/dsir" ;;
    *) echo "unknown run $name" >&2; return 1 ;;
  esac
  # at lambda 64 a rank uses up its share of the pool before the end: further passes in new orders
  [ "$lam" = 64 ] && f="$f --daisy_pool_reshuffle"
  echo "$f $EVAL"
}

ALL_124M="uniform daisy-l2 daisy-l4 daisy-l8 daisy-l16 daisy-l32 daisy-l64 similarity-l2 similarity-l4 similarity-l8 similarity-l16
similarity-l32 similarity-l64 dsir-l4 daisy-l4-fwedu daisy-l16-fwedu daisy-l64-fwedu similarity-l4-fwedu similarity-l16-fwedu
similarity-l64-fwedu uniform-fwedu daisy-l4-fwtopics daisy-l16-fwtopics daisy-l64-fwtopics"
ALL_720M="uniform daisy-l4 daisy-l16 daisy-l64 similarity-l4 similarity-l16 similarity-l64 daisy-l4-fwedu daisy-l16-fwedu
daisy-l64-fwedu similarity-l4-fwedu similarity-l16-fwedu similarity-l64-fwedu uniform-fwedu"

case "${1:-}" in
  list)
    for size in 124M 720M; do
      v=ALL_$size
      for n in ${!v}; do echo "$size-$n: $(flags $n $size)"; done
    done ;;
  flags) flags "${2:?NAME}" "${3:-124M}" ;;
  run)
    name=${2:?NAME} size=${3:-124M}
    NAME=$size-$name SIZE=$size EXTRA="$(flags $name $size)" bash "$HERE/train.sh" ;;
  *) sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
