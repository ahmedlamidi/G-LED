#!/bin/bash -l
#SBATCH --job-name=bl-sword-train
#SBATCH -p general
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ahmedlamidi@usf.edu
#
# One of SWORD's two score models, for TRAIN_HOURS (default 22, fits one job).
# Needs baselines/prepare.sh first. Submit once per band; the two can run at
# the same time. Resubmitting resumes from <band>_last.pt.
#
#   sbatch baselines/sword/train.sh full
#   sbatch baselines/sword/train.sh high

[ -f baselines/cluster/env.sh ] || { echo "Submit from the repo root: sbatch baselines/sword/train.sh full|high"; exit 1; }
BAND="${1:-}"
if [ "$BAND" != "full" ] && [ "$BAND" != "high" ]; then
	echo "usage: sbatch baselines/sword/train.sh full|high [extra args]"
	exit 1
fi
shift
source baselines/cluster/env.sh
run baselines.sword.train --band "$BAND" --hours "${TRAIN_HOURS:-22}" --segment_hours 22.5 "$@"
