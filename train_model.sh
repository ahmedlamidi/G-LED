#!/bin/bash -l
#SBATCH --job-name=ddpm-chest
#SBATCH -p YES
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahmedlamidi@usf.edu
#SBATCH --mem=64G
# 激活 conda（按 GAIVI 手册风格)

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source /apps/anaconda3/etc/profile.d/conda.sh
CONDA_ENV=proj
conda activate "$CONDA_ENV"

export TMPDIR=/home/a/ahmedlamidi/tmp
export TEMP=/home/a/ahmedlamidi/tmp
export TMP=/home/a/ahmedlamidi/tmp

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

# Installs run when forced (INSTALL_DEPS=1 sbatch train_model.sh) or when the
# env's torch cannot actually run on the allocated GPU.
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
	install_deps
elif ! torch_matches_gpu; then
	echo "torch missing or built without kernels for this GPU - installing"
	install_deps
fi

nvcc --version
nvidia-smi

# Fail here with a clear message rather than deep inside the eval script.
"$PY" -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__); print('arch list', torch.cuda.get_arch_list())"
if ! torch_matches_gpu; then
	echo "ERROR: torch has no kernels for this GPU's compute capability."
	echo "       Every CUDA op would fail with 'no kernel image is available'."
	echo "       If pip reported a corrupt dist-info, clear the leftovers first:"
	echo "         ls -d \$CONDA_PREFIX/lib/python3.9/site-packages/~*"
	exit 1
fi
#srun --ntasks=1 --cpus-per-task=1 --exact visdom -port 8097 &
# srun python compare_ssim.py
#to start from saved model

# srun python main_diff_bfs.py --resume
# srun python data/dicom_preprocess.py
srun --export=ALL "$PY" main_diff_eval_bfs.py
# srun python validation/compare_sparse_methods_with_1062.py
# srun --export=ALL "$PY" main_diff_bfs.py --resume
# srun --export=ALL "$PY" data/dicom_preprocess.py
# srun --export=ALL "$PY" validation/compare_sparse_methods_with_1062.py
# srun python validation/match_and_compare_with_1062.py \
# 	--model_sinogram output/LimitedView45Sparse10/diffusion_folder/experiment_final_checkpoint_150/contour/batch0/recon_micro_0.npy \
# 	--dicom_path data/extra_data/1-001.dcm \
# 	--window_width 1500 \
# 	--window_level -600
# srun python saved_ground_truth.py
# srun python validation/compare_ssim.py
# srun python validation/convert_to_dicom.py
# srun --export=ALL "$PY" saved_ground_truth.py
# srun --export=ALL "$PY" validation/compare_ssim.py
# srun --export=ALL "$PY" validation/convert_to_dicom.py

# python dicom_fbp_degraded.py \
#     --dicom_folder data/extra_data  \
#     --output_root Visualization \
#     --limit_deg 45 \
#     --index_step 10 \
#     --window_width 1500 \
#     --window_level -600 \
#     --sart_iterations 200 \
#     --tv_weight 0.002