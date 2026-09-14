#!/bin/bash -l
#SBATCH --job-name=bl-dolce-sample
#SBATCH -p YES
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ahmedlamidi@usf.edu
#
# DOLCE reconstruction of the test patients, after baselines/dolce/train.sh has
# finished. Tunes the data-consistency step on validation slices first if
# tuned.json does not exist yet. Resubmitting skips slices already done.
#
#   sbatch baselines/dolce/sample.sh

[ -f baselines/cluster/env.sh ] || { echo "Submit from the repo root: sbatch baselines/dolce/sample.sh"; exit 1; }
source baselines/cluster/env.sh
run baselines.dolce.sample "$@"
