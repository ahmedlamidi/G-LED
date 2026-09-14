#!/bin/bash -l
#SBATCH --job-name=bl-prepare
#SBATCH -p YES
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ahmedlamidi@usf.edu
#
# Step 1, before any baseline: sinograms, label and conditioning images for the
# 7 / 1 / 2 patient split. Then check data/baselines_cache/*/split.json.
#
#   sbatch baselines/prepare.sh
#   DATA_ROOT=data/AAPM_dataset sbatch baselines/prepare.sh

[ -f baselines/cluster/env.sh ] || { echo "Submit from the repo root: sbatch baselines/prepare.sh"; exit 1; }
source baselines/cluster/env.sh
run baselines.prepare_data --data_root "${DATA_ROOT:-data}" "$@"
