#!/bin/bash -l
#SBATCH --job-name=bl-sdflow-sample
#SBATCH -p YES
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ahmedlamidi@usf.edu
#
# SD-Flow reconstruction of the same test slices as the baselines, with the
# model and measured views of an SD-Flow config. evaluate.py then scores it
# with the rest. Resubmitting skips slices already done.
#
#   sbatch baselines/sdflow/sample.sh --config configurations/<run>.json
#   sbatch baselines/sdflow/sample.sh --config configurations/<other>.json --name sdflow_other

[ -f baselines/cluster/env.sh ] || { echo "Submit from the repo root: sbatch baselines/sdflow/sample.sh --config ..."; exit 1; }
source baselines/cluster/env.sh
run baselines.sdflow.sample "$@"
