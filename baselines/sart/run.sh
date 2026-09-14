#!/bin/bash -l
#SBATCH --job-name=bl-sart
#SBATCH -p YES
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ahmedlamidi@usf.edu
#
# SART baseline: tunes the sweep count on the validation patient, then
# reconstructs the test patients. Needs baselines/prepare.sh first.
#
#   sbatch baselines/sart/run.sh

[ -f baselines/cluster/env.sh ] || { echo "Submit from the repo root: sbatch baselines/sart/run.sh"; exit 1; }
source baselines/cluster/env.sh
run baselines.sart.reconstruct "$@"
