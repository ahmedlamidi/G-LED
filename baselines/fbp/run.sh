#!/bin/bash -l
#SBATCH --job-name=bl-fbp
#SBATCH -p general
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ahmedlamidi@usf.edu
#
# FBP baseline on the test patients. Needs baselines/prepare.sh first.
#
#   sbatch baselines/fbp/run.sh

[ -f baselines/cluster/env.sh ] || { echo "Submit from the repo root: sbatch baselines/fbp/run.sh"; exit 1; }
source baselines/cluster/env.sh
run baselines.fbp.reconstruct "$@"
