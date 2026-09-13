#!/bin/bash -l
#SBATCH --job-name=abl-180k10-nopiml
#SBATCH -p general
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahmedlamidi@usf.edu
#SBATCH --mem=64G
#
# ABLATION of the physics loss at 180 deg coverage. Identical to
# train_180k10_piml.sh except physics_loss_weight = 0.
#
# The 2x2 ablation grid is coverage {45, 180} x physics loss {on, off}.
# Every script in ablations/ is identical apart from the block marked
# "ABLATION KNOBS" below, so any difference in the results is attributable
# to those two variables alone.
#
#   sbatch ablations/train_180k10_nopiml.sh
#   RESUME=1 sbatch ablations/train_180k10_nopiml.sh     # continue after the 24h wall clock
#   INSTALL_DEPS=1 sbatch ablations/train_180k10_nopiml.sh

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source /apps/anaconda3/etc/profile.d/conda.sh
CONDA_ENV=proj
conda activate "$CONDA_ENV"

export TMPDIR=/home/a/ahmedlamidi/tmp
export TEMP=/home/a/ahmedlamidi/tmp
export TMP=/home/a/ahmedlamidi/tmp

# ── ABLATION KNOBS ──────────────────────────────────────────────────────────
# Coverage is a contiguous arc starting at 0 deg; stride 10 means every 10th
# index, and at sample_H = 720 one index is 0.5 deg, so stride 10 = every 5 deg.
RUN_FOLDER=output/720_816_limited180_k10_NONE_PIML
ANGLE_START=0
ANGLE_END=180            # -> 36 of 720 views
ANGLE_STRIDE=10
PHYSICS_LOSS_WEIGHT=0   # 0.1 = conjugate-ray symmetry loss on, 0 = off

# Held fixed across the whole grid, so the runs stay comparable.
BATCH_SIZE=8
EPOCH_NUM=500
UNET_DIM=32
WANDB_PROJECT="sinogram-ablations"
WANDB_RUN_NAME="abl-180k10-nopiml"
# ────────────────────────────────────────────────────────────────────────────

# Call the env's interpreter by absolute path so srun cannot fall back to
# system python if the job step does not inherit the activated environment.
PY="$CONDA_PREFIX/bin/python"

# This GPU (RTX PRO 6000 Blackwell) is sm_120, which only exists in torch's
# CUDA 12.8 builds -> torch >= 2.7. A cu121 wheel imports and reports
# cuda.is_available() == True, then fails every kernel launch with
# "no kernel image is available for execution on the device". So the check
# below is not "is torch installed" but "does this torch have kernels for
# this GPU". torchvision must move in lockstep or imagen_pytorch's import breaks.
TORCH_VERSION=2.8.0
TORCHVISION_VERSION=0.23.0

torch_matches_gpu() {
	"$PY" - <<-'EOF' 2>/dev/null
	import sys
	try:
	    import torch
	except ImportError:
	    sys.exit(1)
	if not torch.cuda.is_available():
	    sys.exit(1)
	arch = 'sm_%d%d' % torch.cuda.get_device_capability()
	sys.exit(0 if arch in torch.cuda.get_arch_list() else 1)
	EOF
}

install_deps() {
	# astra-toolbox comes from requirements.txt via pip; the old
	# `conda install astra-toolbox::astra-toolbox` named a channel that 404s.
	"$PY" -m pip install --no-cache-dir -r requirements.txt
	"$PY" -m pip install --no-cache-dir --upgrade \
		"torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"
}

# Installs run when forced (INSTALL_DEPS=1 sbatch ...) or when the env's torch
# cannot actually run on the allocated GPU.
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
	install_deps
elif ! torch_matches_gpu; then
	echo "torch missing or built without kernels for this GPU - installing"
	install_deps
fi

nvcc --version
nvidia-smi

# Fail here with a clear message rather than deep inside the training script.
"$PY" -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__); print('arch list', torch.cuda.get_arch_list())"
if ! torch_matches_gpu; then
	echo "ERROR: torch has no kernels for this GPU's compute capability."
	echo "       Every CUDA op would fail with 'no kernel image is available'."
	echo "       If pip reported a corrupt dist-info, clear the leftovers first:"
	echo "         ls -d \$CONDA_PREFIX/lib/python3.9/site-packages/~*"
	exit 1
fi

# main_diff_bfs.py reads the sequence-model args from
# <RUN_FOLDER>/logging/args.txt via read_args_txt, which opens the file
# directly and so raises FileNotFoundError on a fresh run folder. Only
# coarse_dim and coarse_mode are ever used downstream, so seed the file from
# the SEQ_ARGS defaults the first time this ablation runs.
SEQ_ARGS_TXT="$RUN_FOLDER/logging/args.txt"
if [ ! -f "$SEQ_ARGS_TXT" ]; then
	echo "Seeding sequence args at $SEQ_ARGS_TXT"
	mkdir -p "$RUN_FOLDER/logging"
	"$PY" - "$SEQ_ARGS_TXT" <<-'EOF'
	import json, sys
	from main_seq_bfs import Args as SEQ_ARGS
	# parse_args(args=[]) takes the defaults without touching sys.argv and
	# without update_args()'s side effect of creating an output/ folder.
	defaults = vars(SEQ_ARGS().parser.parse_args(args=[]))
	with open(sys.argv[1], 'w') as f:
	    json.dump(defaults, f, indent=2)
	EOF
fi

RESUME_FLAG=""
if [ "${RESUME:-0}" = "1" ]; then
	RESUME_FLAG="--resume"
	echo "Resuming from the latest checkpoint in $RUN_FOLDER"
fi

srun --export=ALL "$PY" main_diff_bfs.py \
	--bfs_dynamic_folder "$RUN_FOLDER" \
	--angle_start "$ANGLE_START" \
	--angle_end "$ANGLE_END" \
	--angle_stride "$ANGLE_STRIDE" \
	--physics_loss_weight "$PHYSICS_LOSS_WEIGHT" \
	--batch_size "$BATCH_SIZE" \
	--epoch_num "$EPOCH_NUM" \
	--unet_dim "$UNET_DIM" \
	--wandb_project "$WANDB_PROJECT" \
	--wandb_run_name "$WANDB_RUN_NAME" \
	$RESUME_FLAG
