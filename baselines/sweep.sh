#!/bin/bash -l
#SBATCH --job-name=bl-sweep
#SBATCH -p YES
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ahmedlamidi@usf.edu
#
# The baselines on 8 view settings (45/90/270 deg every row; 360 deg every 2nd,
# 4th, 8th, 10th row; 45 deg every 10th), 10 test slices each: samples SWORD
# per setting (one GPU, one after another) and scores every method that has
# results, then writes the table and strips. Needs the trained SWORD pair and
# baselines/prepare.sh's default setting; DOLCE and FBPConvNet per setting come
# from baselines/train_sweep.sh. Resubmitting skips slices already sampled.
#
#   sbatch baselines/sweep.sh
#   sbatch baselines/sweep.sh --methods fbp,sword,dolce,fbpconvnet
#   sbatch baselines/sweep.sh --no_sample      # rescore and redraw only

[ -f baselines/cluster/env.sh ] || { echo "Submit from the repo root: sbatch baselines/sweep.sh"; exit 1; }
source baselines/cluster/env.sh
run baselines.sweep "$@"
