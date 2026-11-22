#!/bin/bash
# Step 1: the embedding index (daisy/index/build_index.py) from the embedded pool: 100M documents with their
# standardized embeddings, 16 topics (k-means), and the held-out web-text set (2,000 documents per topic, of which the
# runs evaluate the first 64; the rest is the reserve of the FineWeb topic target). The statistics of the index
# (INDEX/stats.npz) standardize the embeddings of the whole pool in step 2.
#   bash scripts/build_index.sh              all stages
#   bash scripts/build_index.sh kmeans       one stage (plan, stats, shard, tokens, kmeans, heldout)
# CPU only except kmeans, which uses a GPU when there is one. The shard stage is resumable.
set -eo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"
cd "$HERE/.."
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
B="python daisy/index/build_index.py --out $INDEX --workers ${WORKERS:-64}"
mkdir -p "$INDEX"
for stage in ${@:-plan stats shard tokens kmeans heldout}; do
  t=$(date +%s)
  case $stage in
    plan)  $B plan --pool-dir "$EMBEDDED_POOL" --target-docs ${TARGET_DOCS:-100000000} ;;
    stats) $B stats --stats-files 64 --stats-rowgroups 16 ;;
    *)     $B $stage ;;
  esac
  echo "=== stage $stage done in $(( $(date +%s) - t )) s"
done
