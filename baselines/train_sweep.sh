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
# Train one method (METHOD=dolce, fbpconvnet or dudotrans) on the sweep's view
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
#   METHOD=dudotrans sbatch baselines/train_sweep.sh      # validation patience, DUDO_HOURS cap per setting
#   METHOD=dolce sbatch baselines/train_sweep.sh          # several days, one job at a time
#   METHOD=dolce SETTINGS="0 45 1;0 90 1" sbatch baselines/train_sweep.sh   # a subset
#
# Disk: each setting needs ~2.8 GB per input array (RLS for DOLCE, FBP for
# FBPConvNet) for the training patients, plus the shared label; the arrays are
# removed again after sampling (CLEAN=0 keeps them). A step that fails aborts
# the job instead of resubmitting.

[ -f baselines/cluster/env.sh ] || { echo "Submit from the repo root: METHOD=dolce sbatch baselines/train_sweep.sh"; exit 1; }
source baselines/cluster/env.sh
: "${METHOD:?set METHOD=dolce, fbpconvnet or dudotrans}"
SETTINGS="${SETTINGS:-0 45 1;0 90 1;0 270 1;0 360 2;0 360 4;0 360 8;0 360 10}"
DOLCE_HOURS="${DOLCE_HOURS:-22}"          # cap; the plateau rule usually stops earlier
FBPCONV_HOURS="${FBPCONV_HOURS:-12}"
DUDO_HOURS="${DUDO_HOURS:-20}"
MAX_RESUBMITS="${MAX_RESUBMITS:-12}"
BASE_SETTING="${BASE_SETTING:-limited0-45_stride10}"
# patients per split, read from the base setting so every setting has the same split
read -r N_TRAIN N_VAL N_TEST < <("$PY" - "data/baselines_cache/$BASE_SETTING/split.json" <<'PYEOF'
import json, sys
p = json.load(open(sys.argv[1]))['patients']
print(len(p['train']), len(p['val']), len(p['test']))
PYEOF
)
[ -n "$N_TEST" ] || { echo "train_sweep: cannot read data/baselines_cache/$BASE_SETTING/split.json"; exit 1; }
echo "train_sweep: split from $BASE_SETTING: $N_TRAIN train / $N_VAL val / $N_TEST test patients"
WALL_H="${WALL_H:-168}"                    # hours this job may use; matches --time above

hours_left() { echo $(( (WALL_H * 3600 - SECONDS) / 3600 )); }
fail() { echo "train_sweep: $1; not resubmitting, fix and resubmit by hand"; exit 1; }
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

	# only the input this method needs; the label is shared with the base setting
	case "$METHOD" in
		dolce)      IMAGES=label,rls ;;
		fbpconvnet) IMAGES=label,fbp_in ;;
		dudotrans)  IMAGES=label ;;          # it reads the cached sinograms; nothing else to write
		*) echo "unknown METHOD $METHOD"; exit 1 ;;
	esac
	# One prepare at a time per setting: two jobs writing the same memory-mapped arrays
	# over NFS crash each other (bus error). mkdir is atomic on NFS, so it serves as the lock.
	LOCK="data/baselines_cache/.prepare_$TAG.lock"
	mkdir -p data/baselines_cache
	WAITED=0
	until mkdir "$LOCK" 2>/dev/null; do
		[ "$WAITED" -ge 7200 ] && fail "another job has held $LOCK for 2 h; remove it if that job is gone"
		[ "$WAITED" -eq 0 ] && echo "train_sweep: waiting for another job's prepare of $TAG ($LOCK)"
		sleep 60; WAITED=$((WAITED + 60))
	done
	trap 'rmdir "$LOCK" 2>/dev/null' EXIT
	# same split as the base setting (its test patients), so labels can be shared and nothing is sampled twice
	run baselines.prepare_data --data_root "${DATA_ROOT:-data/LIDC-IDRI}" "${ANGLES[@]}" \
		--n_train "$N_TRAIN" --n_val "$N_VAL" --n_test "$N_TEST" \
		--images "$IMAGES" --label_from "data/baselines_cache/$BASE_SETTING"
	STATUS=$?
	rmdir "$LOCK" 2>/dev/null; trap - EXIT
	[ "$STATUS" -eq 0 ] || fail "prepare failed for $TAG"
	if [ ! -f "$OUT/train_done" ]; then
		LEFT=$(( $(hours_left) - 1 ))       # keep an hour for the checkpoint and sampling
		case "$METHOD" in
			dolce)      run baselines.dolce.train "${ANGLES[@]}" --hours "$DOLCE_HOURS" --segment_hours "$LEFT" \
			                || fail "training failed for $TAG" ;;
			fbpconvnet) [ "$LEFT" -lt 3 ] && resubmit
			            run baselines.fbpconvnet.train "${ANGLES[@]}" --hours "$FBPCONV_HOURS" \
			                || fail "training failed for $TAG" ;;
			dudotrans)  [ "$LEFT" -lt 3 ] && resubmit
			            run baselines.dudotrans.train "${ANGLES[@]}" --hours "$(( LEFT < DUDO_HOURS ? LEFT : DUDO_HOURS ))" \
			                || fail "training failed for $TAG" ;;
		esac
		[ -f "$OUT/train_done" ] || resubmit     # segment ended before the stopping rule: continue next job
	fi
	case "$METHOD" in
		dolce)      run baselines.dolce.sample "${ANGLES[@]}" || fail "sampling failed for $TAG" ;;
		fbpconvnet) run baselines.fbpconvnet.test "${ANGLES[@]}" || fail "test failed for $TAG" ;;
		dudotrans)  run baselines.dudotrans.test "${ANGLES[@]}" || fail "test failed for $TAG" ;;
	esac
	if [ "${CLEAN:-1}" = "1" ] && [ "$METHOD" != dudotrans ]; then
		# the method's training/validation input arrays (~2.8 GB per setting) are not needed any more;
		# the test split and the shared label stay. prepare rebuilds them if a run is ever repeated.
		INPUT=$([ "$METHOD" = dolce ] && echo rls || echo fbp_in)
		rm -f "data/baselines_cache/$TAG/train/$INPUT.npy" "data/baselines_cache/$TAG/val/$INPUT.npy"
		echo "train_sweep: removed $TAG train/val $INPUT arrays (CLEAN=0 keeps them)"
	fi
done
echo "train_sweep: all settings done for $METHOD"
