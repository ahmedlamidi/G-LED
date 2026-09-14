#!/bin/bash
# Optional: submit every baseline job at once, each waiting only for what it
# needs. Every job is a single-GPU script that can also be submitted on its own
# (see baselines/README.md). Run from anywhere on the login node:
#
#   baselines/run_all.sh
#   DATA_ROOT=/other/path/LIDC-IDRI PARTITION=YES baselines/run_all.sh
#
#   prepare ─┬─ fbp/run.sh, sart/run.sh, tv/run.sh
#            ├─ fbpconvnet/run.sh
#            ├─ dolce/train.sh ── dolce/sample.sh
#            └─ sword/train.sh full ─┐
#               sword/train.sh high ─┴─ sword/sample.sh
#   evaluate.sh once all of those have ended
set -euo pipefail
cd "$(dirname "$0")/.."

# submit [sbatch options...] SCRIPT [args...]   -> prints the job id
submit() {
	sbatch --parsable ${PARTITION:+-p "$PARTITION"} "$@" | cut -d';' -f1
}

PREP=$(submit baselines/prepare.sh)
AFTER_PREP=--dependency=afterok:$PREP

FBP=$(submit "$AFTER_PREP" baselines/fbp/run.sh)
SART=$(submit "$AFTER_PREP" baselines/sart/run.sh)
TV=$(submit "$AFTER_PREP" baselines/tv/run.sh)
FBPCONV=$(submit "$AFTER_PREP" baselines/fbpconvnet/run.sh)

DOLCE_TRAIN=$(submit "$AFTER_PREP" baselines/dolce/train.sh)
DOLCE=$(submit --dependency=afterok:"$DOLCE_TRAIN" baselines/dolce/sample.sh)

SWORD_FULL=$(submit "$AFTER_PREP" baselines/sword/train.sh full)
SWORD_HIGH=$(submit "$AFTER_PREP" baselines/sword/train.sh high)
SWORD=$(submit --dependency=afterok:"$SWORD_FULL":"$SWORD_HIGH" baselines/sword/sample.sh)

EVAL=$(submit --dependency=afterany:"$FBP":"$SART":"$TV":"$FBPCONV":"$DOLCE":"$SWORD" baselines/evaluate.sh)

cat <<EOF
Submitted (logs: slurm-bl-*.out in the repo root):
  prepare          $PREP
  fbp / sart / tv  $FBP / $SART / $TV
  fbpconvnet       $FBPCONV
  dolce            train $DOLCE_TRAIN, sample $DOLCE
  sword            train full $SWORD_FULL, high $SWORD_HIGH, sample $SWORD
  evaluate         $EVAL
Check the split once prepare finishes: cat data/baselines_cache/*/split.json
EOF
