#!/bin/bash
#SBATCH --job-name=pepo_benchmark
#SBATCH --output=out/benchmark-%j.out
#SBATCH --error=out/benchmark-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
#SBATCH --account=a0136
#SBATCH --exclusive --mem=450G

cd ~/projects/pepo

# Redirect uv and Hugging Face caches to Lustre scratch directory to avoid quota limits
export UV_CACHE_DIR="/capstor/scratch/cscs/barla/uv_cache"
export HF_HOME="/capstor/scratch/cscs/barla/hf_cache"

uv run scripts/benchmark.py "$@"
