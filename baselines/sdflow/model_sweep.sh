#!/bin/bash -l
#SBATCH --job-name=bl-sdflow-sweep
#SBATCH -p YES
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ahmedlamidi@usf.edu
#
# Every SD-Flow model in configurations/sdflow_models.json on the sweep's 10
# test slices with its trained window; table and grid figure like the SWORD
# sweep. ~5 s per slice at 20 steps, so about 15 min for 16 models.
#
#   sbatch baselines/sdflow/model_sweep.sh
#   sbatch baselines/sdflow/model_sweep.sh --no_sample     # rescore and redraw only

[ -f baselines/cluster/env.sh ] || { echo "Submit from the repo root: sbatch baselines/sdflow/model_sweep.sh"; exit 1; }
source baselines/cluster/env.sh
run baselines.sdflow.model_sweep "$@"
