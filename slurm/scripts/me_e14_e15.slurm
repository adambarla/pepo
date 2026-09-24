#!/bin/bash
#SBATCH --job-name=me_e14_15
#SBATCH --output=outputs/slurm/eval/me_e14_15_%A_%a.out
#SBATCH --error=outputs/slurm/eval/me_e14_15_%A_%a.err
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:4
#SBATCH --account=a0136
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --array=14-15
cd ~/projects/pepo
ulimit -c 0
export UV_CACHE_DIR="/capstor/scratch/cscs/barla/uv_cache"
export HF_HOME="/capstor/scratch/cscs/barla/hf_cache"
EPOCH=$SLURM_ARRAY_TASK_ID
uenv run prgenv-gnu/26.3:v1 -- bash -c '
  export CUDA_HOME=/user-environment/env/default
  export PATH="${CUDA_HOME}/bin:${PATH}"
  .venv/bin/python scripts/eval.py model=deppo L=1 a=0.1 b=0.1 split_mode=disjoint e='$EPOCH' backbone=llama8b model@ref_model=deppo ref_L=1 ref_a=0.0 ref_b=0.1 ref_e=0 sync=true
' -- ""
