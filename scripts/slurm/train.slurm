#!/bin/bash
#SBATCH --job-name=train_pepo
#SBATCH --output=outputs/slurm/train/%j.out
#SBATCH --error=outputs/slurm/train/%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
#SBATCH --account=a0136
#SBATCH --exclusive
#SBATCH --mem=450G

cd ~/projects/pepo

# Redirect uv and Hugging Face caches to Lustre scratch directory to avoid quota limits
export UV_CACHE_DIR="/capstor/scratch/cscs/barla/uv_cache"
export HF_HOME="/capstor/scratch/cscs/barla/hf_cache"

uv run scripts/train.py "$@"
