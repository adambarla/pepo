#!/bin/bash
#SBATCH --job-name=pepo_score
#SBATCH --partition=normal
#SBATCH --output=outputs/slurm/score/%j.out
#SBATCH --error=outputs/slurm/score/%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:4
#SBATCH --account=a0136
#SBATCH --exclusive --mem=450G

cd ~/projects/pepo
ulimit -c 0

uenv run prgenv-gnu/26.3:v1 -- bash -c '
  export CUDA_HOME=/user-environment/env/default
  export PATH="${CUDA_HOME}/bin:${PATH}"
  .venv/bin/python scripts/score_timing_campaign.py "$@"
' -- "$@"
