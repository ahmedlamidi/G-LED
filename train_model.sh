#!/bin/bash -l
#SBATCH --job-name=ddpm-chest
#SBATCH -p general
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

# Dependency installs re-solve the env and can bump its python version, which
# orphans pip-installed torch. Run them deliberately, not on every job:
#   INSTALL_DEPS=1 sbatch train_model.sh
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
	conda install -y pip
	conda install -y astra-toolbox::astra-toolbox
	"$PY" -m pip install -r requirements.txt
fi

nvcc --version
nvidia-smi

# Fail here with a clear message rather than deep inside the training script.
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
#srun --ntasks=1 --cpus-per-task=1 --exact visdom -port 8097 &
# srun python compare_ssim.py
#to start from saved model

# srun python main_diff_bfs.py --resume
# srun python data/dicom_preprocess.py
srun python main_diff_eval_bfs.py
# srun python validation/compare_sparse_methods_with_1062.py
# srun --export=ALL "$PY" main_diff_bfs.py --resume
# srun --export=ALL "$PY" data/dicom_preprocess.py
srun --export=ALL "$PY" main_diff_eval_bfs.py
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