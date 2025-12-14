#!/bin/bash
#SBATCH --job-name=pepo_eval
#SBATCH --output=slurm_out/eval/%j.out
#SBATCH --error=slurm_out/eval/%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
#SBATCH --account=infra01
#SBATCH --exclusive --mem=450G

cd ~/PEPO

uv run scripts/eval.py "$@"
