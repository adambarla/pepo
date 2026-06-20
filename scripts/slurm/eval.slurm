#!/bin/bash
#SBATCH --job-name=pepo_eval
#SBATCH --partition=normal
#SBATCH --output=outputs/slurm/eval/%j.out
#SBATCH --error=outputs/slurm/eval/%j.err
#SBATCH --time=48:00:00
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
  
  # Prepend virtual environment NVIDIA library paths to LD_LIBRARY_PATH to prevent import errors on misconfigured nodes
  NVIDIA_LIBS=""
  for d in "${HOME}/projects/pepo/.venv/lib/python3.13/site-packages/nvidia"/*/lib; do
    if [ -d "$d" ]; then
      NVIDIA_LIBS="${d}:${NVIDIA_LIBS}"
    fi
  done
  export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH}"

  uv run scripts/eval.py "$@"
' -- "$@"
