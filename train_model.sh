#!/bin/bash -l
#SBATCH --job-name=ddpm-chest
#SBATCH -p Quick
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
conda install -y pip
conda install astra-toolbox::astra-toolbox
export TMPDIR=/home/a/ahmedlamidi/tmp
export TEMP=/home/a/ahmedlamidi/tmp
export TMP=/home/a/ahmedlamidi/tmp
#pip cache dir  
nvcc --version 
pip install -r requirements.txt
nvidia-smi
#srun --ntasks=1 --cpus-per-task=1 --exact visdom -port 8097 &
srun python compare_ssim.py
#to start from saved model

#python main_diff_bfs.py --resume

# srun python saved_ground_truth.py

