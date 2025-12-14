#!/bin/bash
#SBATCH --job-name=train_pepo
#SBATCH --output=slurm_out/train/%j.out
#SBATCH --error=slurm_out/train/%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
#SBATCH --account=infra01
#SBATCH --exclusive
#SBATCH --mem=450G

cd ~/PEPO

uv run scripts/train.py "$@"
