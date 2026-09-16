#!/bin/bash -l
#SBATCH --job-name=bl-sdflow-start
#SBATCH -p YES
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ahmedlamidi@usf.edu
#
# Score each SD-Flow model with its conditioning window started at every
# --step degrees, to find (or confirm) the window it was trained with.
#
#   sbatch baselines/sdflow/start_sweep.sh --config configurations/Limited45Sparse10.json
#   sbatch baselines/sdflow/start_sweep.sh --config configurations/A.json --config configurations/B.json

[ -f baselines/cluster/env.sh ] || { echo "Submit from the repo root: sbatch baselines/sdflow/start_sweep.sh --config ..."; exit 1; }
source baselines/cluster/env.sh
run baselines.sdflow.start_sweep "$@"
