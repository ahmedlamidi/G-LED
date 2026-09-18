#!/bin/bash -l
#SBATCH --job-name=bl-dudotrans
#SBATCH -p YES
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ahmedlamidi@usf.edu
#
# DuDoTrans for one view setting: trains (validation patience, TRAIN_HOURS cap,
# default 20), then reconstructs the test slices with the best validation
# checkpoint. Needs baselines/prepare.sh first. Resubmitting resumes from last.pt.
# The setting arguments reach both steps; training-only options go in TRAIN_ARGS.
#
#   sbatch baselines/dudotrans/run.sh
#   sbatch baselines/dudotrans/run.sh --angle_start 0 --angle_end 360 --angle_stride 10

[ -f baselines/cluster/env.sh ] || { echo "Submit from the repo root: sbatch baselines/dudotrans/run.sh"; exit 1; }
source baselines/cluster/env.sh
run baselines.dudotrans.train --hours "${TRAIN_HOURS:-20}" ${TRAIN_ARGS:-} "$@" && run baselines.dudotrans.test "$@"
