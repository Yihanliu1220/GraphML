#!/bin/bash
#SBATCH --job-name=train_knowddi
#SBATCH --partition=week
#SBATCH --ntasks=1 --nodes=1
#SBATCH --mem-per-cpu=150G
#SBATCH --time=30:00:00
#SBATCH --account=pi_hx235
#SBATCH --mail-type ALL


module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate knowddi_env

export HF_DATASETS_CACHE=/home/zc347/project_pi_hx235/zc347/cache/huggingface/datasets
export TRANSFORMERS_CACHE=/home/zc347/project_pi_hx235/zc347/cache/huggingface/transformers

python /home/zc347/project/KnowDDI/pytorch/train.py -e Drugbank --gpu 0