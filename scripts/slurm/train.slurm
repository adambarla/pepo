#!/bin/bash
#SBATCH --job-name=train_pepo
#SBATCH --output=outputs/slurm/train/%j.out
#SBATCH --error=outputs/slurm/train/%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
#SBATCH --account=infra01
#SBATCH --exclusive
#SBATCH --mem=450G

cd ~/pepo

uv run scripts/train.py "$@"
