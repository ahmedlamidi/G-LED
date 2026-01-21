#!/bin/bash -l
#SBATCH --job-name=ddpm-chest
#SBATCH -p Quick 
#SBATCH --cpus-per-task=2
#SBATCH --time=22:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahmedlamidi@usf.edu

# 激活 conda（按 GAIVI 手册风格)

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source /apps/anaconda3/etc/profile.d/conda.sh
CONDA_ENV=proj
conda activate "$CONDA_ENV"
conda install -y pip
#pip cache dir   
pip install -r requirements.txt
#srun --ntasks=1 --cpus-per-task=1 --exact visdom -port 8097 &
srun python main_diff_bfs.py

