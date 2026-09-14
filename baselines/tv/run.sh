#!/bin/bash -l
#SBATCH --job-name=bl-tv
#SBATCH -p YES
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ahmedlamidi@usf.edu
#
# TV baseline: tunes the TV weight on the validation patient, then reconstructs
# the test patients. Needs baselines/prepare.sh first.
#
#   sbatch baselines/tv/run.sh

[ -f baselines/cluster/env.sh ] || { echo "Submit from the repo root: sbatch baselines/tv/run.sh"; exit 1; }
source baselines/cluster/env.sh
run baselines.tv.reconstruct "$@"
