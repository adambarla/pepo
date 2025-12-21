#!/bin/bash
#SBATCH --job-name=train_pepo
#SBATCH --partition=long
#SBATCH --output=outputs/slurm/train/%j.out
#SBATCH --error=outputs/slurm/train/%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4
#SBATCH --account=infra01
#SBATCH --exclusive
#SBATCH --mem=450G

cd ~/PEPO

uv  run --no-sync scripts/train.py "$@"
