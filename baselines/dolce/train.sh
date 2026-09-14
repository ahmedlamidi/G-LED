#!/bin/bash -l
#SBATCH --job-name=bl-dolce-train
#SBATCH -p YES
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ahmedlamidi@usf.edu
#
# DOLCE training for TRAIN_HOURS (default 22, fits one job). Needs
# baselines/prepare.sh first. For longer training, set TRAIN_HOURS above 22.5
# and submit this script again after each job ends; it resumes from last.pt.
#
#   sbatch baselines/dolce/train.sh
#   TRAIN_HOURS=40 sbatch baselines/dolce/train.sh     # then once more after it stops

[ -f baselines/cluster/env.sh ] || { echo "Submit from the repo root: sbatch baselines/dolce/train.sh"; exit 1; }
source baselines/cluster/env.sh
run baselines.dolce.train --hours "${TRAIN_HOURS:-22}" --segment_hours 22.5 "$@"
