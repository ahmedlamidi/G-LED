#!/bin/bash -l
#SBATCH --job-name=bl-train-sweep
#SBATCH -p YES
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ahmedlamidi@usf.edu
#
# Train one method (METHOD=dolce or METHOD=fbpconvnet) on the sweep's view
# settings, one after another on one GPU: for each setting, prepare the data
# (skips splits that are up to date), train until the method's own stopping
# rule (DOLCE: loss plateau, FBPConvNet: validation patience) or the cap, then
# reconstruct the test slices. The YES partition has no wall-clock limit, so
# one 7-day job covers the whole sweep; settings already done are skipped and,
# should the job still run out (WALL_H), it resubmits itself and a DOLCE
# setting resumes from last.pt.
# The reference setting (45 deg every 10th row) is assumed trained already.
#
#   METHOD=fbpconvnet sbatch baselines/train_sweep.sh     # ~1 day in total
#   METHOD=dolce sbatch baselines/train_sweep.sh          # several days, one job at a time
#   METHOD=dolce SETTINGS="0 45 1;0 90 1" sbatch baselines/train_sweep.sh   # a subset

[ -f baselines/cluster/env.sh ] || { echo "Submit from the repo root: METHOD=dolce sbatch baselines/train_sweep.sh"; exit 1; }
source baselines/cluster/env.sh
: "${METHOD:?set METHOD=dolce or METHOD=fbpconvnet}"
SETTINGS="${SETTINGS:-0 45 1;0 90 1;0 270 1;0 360 2;0 360 4;0 360 8;0 360 10}"
DOLCE_HOURS="${DOLCE_HOURS:-22}"          # cap; the plateau rule usually stops earlier
FBPCONV_HOURS="${FBPCONV_HOURS:-12}"
MAX_RESUBMITS="${MAX_RESUBMITS:-12}"
WALL_H="${WALL_H:-168}"                    # hours this job may use; matches --time above

hours_left() { echo $(( (WALL_H * 3600 - SECONDS) / 3600 )); }
resubmit() {
	if [ "${RESUBMITS:-0}" -ge "$MAX_RESUBMITS" ]; then
		echo "train_sweep: $MAX_RESUBMITS resubmissions reached, stopping"; exit 1
	fi
	echo "train_sweep: work remains, resubmitting ($((RESUBMITS + 1)))"
	RESUBMITS=$((${RESUBMITS:-0} + 1)) METHOD="$METHOD" SETTINGS="$SETTINGS" \
		sbatch --export=ALL baselines/train_sweep.sh
	exit 0
}

IFS=';' read -ra LIST <<< "$SETTINGS"
for spec in "${LIST[@]}"; do
	read -r A B K <<< "$spec"
	TAG="limited${A}-${B}_stride${K}"
	OUT="output/baselines/$TAG/$METHOD"
	ANGLES=(--angle_start "$A" --angle_end "$B" --angle_stride "$K")
	if [ -f "$OUT/train_done" ] && [ -d "$OUT/recon" ] && [ -n "$(ls "$OUT/recon" 2>/dev/null)" ]; then
		echo "train_sweep: $TAG $METHOD already done"; continue
	fi
	# a fresh setting needs prepare (up to ~1 h) plus a useful stretch of training
	[ "$(hours_left)" -lt 3 ] && resubmit

	run baselines.prepare_data --data_root "${DATA_ROOT:-data/LIDC-IDRI}" "${ANGLES[@]}"
	if [ ! -f "$OUT/train_done" ]; then
		LEFT=$(( $(hours_left) - 1 ))       # keep an hour for the checkpoint and sampling
		case "$METHOD" in
			dolce)      run baselines.dolce.train "${ANGLES[@]}" --hours "$DOLCE_HOURS" --segment_hours "$LEFT" ;;
			fbpconvnet) [ "$LEFT" -lt 3 ] && resubmit
			            run baselines.fbpconvnet.train "${ANGLES[@]}" --hours "$FBPCONV_HOURS" ;;
			*) echo "unknown METHOD $METHOD"; exit 1 ;;
		esac
		[ -f "$OUT/train_done" ] || resubmit     # segment ended before the stopping rule: continue next job
	fi
	case "$METHOD" in
		dolce)      run baselines.dolce.sample "${ANGLES[@]}" ;;
		fbpconvnet) run baselines.fbpconvnet.test "${ANGLES[@]}" ;;
	esac
done
echo "train_sweep: all settings done for $METHOD"
