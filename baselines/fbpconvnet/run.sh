#!/bin/bash -l
#SBATCH --job-name=bl-fbpconvnet
#SBATCH -p general
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ahmedlamidi@usf.edu
#
# FBPConvNet: trains (up to TRAIN_HOURS, default 12), then reconstructs the
# test patients with the best validation checkpoint. Needs baselines/prepare.sh
# first. Resubmitting resumes training from last.pt.
#
#   sbatch baselines/fbpconvnet/run.sh

[ -f baselines/cluster/env.sh ] || { echo "Submit from the repo root: sbatch baselines/fbpconvnet/run.sh"; exit 1; }
source baselines/cluster/env.sh
run baselines.fbpconvnet.train --hours "${TRAIN_HOURS:-12}" "$@" && run baselines.fbpconvnet.test
