#!/bin/bash -l
#SBATCH --job-name=bl-sword-sample
#SBATCH -p YES
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ahmedlamidi@usf.edu
#
# SWORD reconstruction of the test patients, after both
# `baselines/sword/train.sh full` and `... high` have finished. Resubmitting
# skips slices already done.
#
#   sbatch baselines/sword/sample.sh

[ -f baselines/cluster/env.sh ] || { echo "Submit from the repo root: sbatch baselines/sword/sample.sh"; exit 1; }
source baselines/cluster/env.sh
run baselines.sword.sample "$@"
