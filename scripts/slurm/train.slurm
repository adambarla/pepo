#!/bin/bash
#SBATCH --job-name=pepo_train
#SBATCH --output=out/pepo_train-%j.out
#SBATCH --error=out/pepo_train-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --account=infra01

cd ~/PEPO

uv run scripts/train.py "$@"
