#!/bin/bash
#SBATCH --job-name=boot_ev_test
#SBATCH --output=outputs/slurm/eval/boot_ev_test_%j.out
#SBATCH --error=outputs/slurm/eval/boot_ev_test_%j.err
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:4
#SBATCH --account=a0136
#SBATCH --exclusive
#SBATCH --mem=450G
cd ~/projects/pepo
ulimit -c 0
export UV_CACHE_DIR="/capstor/scratch/cscs/barla/uv_cache"
export HF_HOME="/capstor/scratch/cscs/barla/hf_cache"
uenv run prgenv-gnu/26.3:v1 -- bash -c '
  export CUDA_HOME=/user-environment/env/default
  export PATH="${CUDA_HOME}/bin:${PATH}"
  .venv/bin/python scripts/eval.py model=deppo L=4 a=0.1 b=0.1 split_mode=bootstrap e=1 backbone=llama8b model@ref_model=deppo ref_L=1 ref_a=0.0 ref_b=0.1 ref_e=0 sync=true
' -- ""
