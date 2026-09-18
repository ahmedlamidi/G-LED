#!/bin/bash -l
#SBATCH --job-name=bl-dps-prior
#SBATCH -p YES
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ahmedlamidi@usf.edu
#
# DPS's dedicated prior: DOLCE's network and training (Adam 1.5e-4, batch 8,
# EMA, 22 h cap, plateau stop) with the condition always dropped, so an
# unconditional model of the label images. One prior serves every view
# setting; it reads only the labels of the default setting's train split.
# dps/sample.sh and sweep.sh use it once train_done exists. Submitting again
# resumes from last.pt.
#
#   sbatch baselines/dps/train_prior.sh

[ -f baselines/cluster/env.sh ] || { echo "Submit from the repo root: sbatch baselines/dps/train_prior.sh"; exit 1; }
source baselines/cluster/env.sh
run baselines.dolce.train --p_uncond 1 --name dps_prior --hours "${TRAIN_HOURS:-22}" --segment_hours 22.5 "$@"
