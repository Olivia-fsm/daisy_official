#!/bin/bash
# Step 3: the selection targets and the fixed scorers of the baselines. One sub-command per item:
#
#   benchmarks        BENCH: the five benchmark tasks (train = the target, val = the held-out downstream loss)
#   similarity        SCORERS/similarity/w_static.npy: the benchmark target's centroid in embedding space (one GPU)
#   dsir              SCORERS/dsir: DSIR importance weights of the pool documents a run opens; needs
#                     COUNTS=N0,N1,N2,N3 (see below)
#   fineweb_topics    TARGETS/fineweb_topics: 16 target tasks, one per topic, from the held-out reserve of the index
#   fineweb_edu       TARGETS/fineweb_edu: 16 target tasks of FineWeb-Edu documents with score >= 4 from a crawl
#                     snapshot that the pool does not contain (EDU_DUMP, default CC-MAIN-2023-06), and a held-out
#                     FineWeb-Edu set
#   similarity_fwedu  SCORERS/similarity_fwedu/w_static.npy: the FineWeb-Edu target's centroid (one GPU)
#   edu_scores        EDU_SCORES: the FineWeb-Edu score of every pool document (for the run uniform-fwedu)
#   pool_copies       copies of the FineWeb-Edu target and held-out documents in the pool; writes the held-out
#                     documents without any pool copy as TARGETS/fineweb_edu/fwedu_theldout_nocopy
#
# COUNTS (dsir): the number of pool shards each process of the run opens. The candidate windows do not depend on the
# scorer, so take it from a finished run with the same seed, GPU count and window factor, for example the similarity
# run: the "next_shard" of every EXPS/<run>/ckpts/latest/stream_r<rank>.json (under "base"), in rank order.
set -eo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"
cd "$HERE/.."
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
WORKERS=${WORKERS:-64}
FWEDU_TASKS=$(seq -f 'fwedu_t%02g' 0 15 | paste -sd, -)

case "${1:-}" in
  benchmarks)
    python daisy/targets/prepare_benchmarks.py --out "$BENCH" ;;
  similarity)
    python daisy/scorers/similarity.py --index "$INDEX" --bench "$BENCH" --out "$SCORERS/similarity" ;;
  dsir)
    python daisy/scorers/dsir_build.py --pool "$POOL" --index "$INDEX" --bench "$BENCH" --out "$SCORERS/dsir" \
      --counts "${COUNTS:?set COUNTS (see the header of this script)}" --workers $WORKERS ;;
  fineweb_topics)
    python daisy/targets/build_fineweb_target.py --index "$INDEX" --out "$TARGETS/fineweb_topics" ;;
  fineweb_edu)
    python daisy/targets/build_fwedu_target.py --index "$INDEX" --bench "$BENCH" --edu_dir "$FINEWEB_EDU" \
      --out "$TARGETS/fineweb_edu" ${EDU_DUMP:+--dump "$EDU_DUMP"} ;;
  similarity_fwedu)
    python daisy/scorers/similarity.py --index "$INDEX" --bench "$TARGETS/fineweb_edu" --tasks "$FWEDU_TASKS" \
      --out "$SCORERS/similarity_fwedu" ;;
  edu_scores)
    python daisy/targets/edu_scores.py --pool "$POOL" --edu_dir "$FINEWEB_EDU" --out "$EDU_SCORES" --workers $WORKERS ;;
  pool_copies)
    python daisy/targets/pool_copies.py --pool "$POOL" --target "$TARGETS/fineweb_edu" --keep_dir "$EDU_SCORES" \
      --out "$TARGETS/fineweb_edu/pool_copies" --clean_out "$TARGETS/fineweb_edu/fwedu_theldout_nocopy" \
      --workers $WORKERS ;;
  *) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
