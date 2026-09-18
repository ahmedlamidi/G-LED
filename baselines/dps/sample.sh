#!/bin/bash -l
#SBATCH --job-name=bl-dps-sample
#SBATCH -p YES
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ahmedlamidi@usf.edu
#
# DPS reconstruction of the test slices with DOLCE's unconditional branch as
# the prior (no training of its own). Tunes zeta on validation slices first if
# tuned.json does not exist. About 1 min per slice; resubmitting skips slices
# already done.
#
#   sbatch baselines/dps/sample.sh
#   sbatch baselines/dps/sample.sh --prior output/baselines/limited0-45_stride10/dps_prior/last.pt

[ -f baselines/cluster/env.sh ] || { echo "Submit from the repo root: sbatch baselines/dps/sample.sh"; exit 1; }
source baselines/cluster/env.sh
run baselines.dps.sample "$@"
