#!/bin/bash -l
#SBATCH --job-name=bl-evaluate
#SBATCH -p general
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ahmedlamidi@usf.edu
#
# Table and figures for every method that has test reconstructions so far; run
# it again whenever more finish. SD-Flow runs are added with --sdflow (or the
# SDFLOW variable, space-separated NAME=DIR[,DIR] entries).
#
#   sbatch baselines/evaluate.sh
#   sbatch baselines/evaluate.sh --sdflow "SD-Flow=best_model_folder/<run>/diffusion_folder/<series>/contour"

[ -f baselines/cluster/env.sh ] || { echo "Submit from the repo root: sbatch baselines/evaluate.sh"; exit 1; }
source baselines/cluster/env.sh
SDFLOW_ARGS=()
for spec in ${SDFLOW:-}; do
	SDFLOW_ARGS+=(--sdflow "$spec")
done
run baselines.evaluate ${SDFLOW_ARGS[@]+"${SDFLOW_ARGS[@]}"} "$@"
