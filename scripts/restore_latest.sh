#!/bin/bash
# After a job was killed during a checkpoint swap (DAISY_ATOMIC_LATEST=1, daisy/train/checkpoint.py: swap_latest): put
# the whole checkpoint back at ckpts/latest and remove the leftovers. Run it before the job restarts.
#   scripts/restore_latest.sh EXP_DIR
C=$1/ckpts
[ -d "$C" ] || exit 0
if [ ! -e "$C/latest/main.pt" ] && [ -d "$C/latest.old" ]; then
  rm -rf "$C/latest"
  mv "$C/latest.old" "$C/latest"
  echo "[restore_latest] ckpts/latest.old put back"
fi
rm -rf "$C/latest.new" "$C/latest.old"
