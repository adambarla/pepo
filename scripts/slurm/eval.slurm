#!/bin/bash
#SBATCH --job-name=pepo_eval
#SBATCH --partition=normal
#SBATCH --output=outputs/slurm/eval/%j.out
#SBATCH --error=outputs/slurm/eval/%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:4
#SBATCH --account=a0136
#SBATCH --exclusive --mem=450G

cd ~/projects/pepo

# Disable core dump generation
ulimit -c 0

# Mount uenv without loading the full view (avoids NCCL library conflicts).
# Only set CUDA_HOME + nvcc PATH for FlashInfer JIT compilation.
uenv run prgenv-gnu/26.3:v1 -- bash -c '
  export CUDA_HOME=/user-environment/env/default
  export PATH="${CUDA_HOME}/bin:${PATH}"
  
  uv run scripts/eval.py "$@"
' -- "$@"
