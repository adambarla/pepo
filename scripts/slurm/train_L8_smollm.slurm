#!/bin/bash
#SBATCH --job-name=pepo_L8_smollm
#SBATCH --output=out/train_L8-%j.out
#SBATCH --error=out/train_L8-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
#SBATCH --account=infra01
#SBATCH --exclusive --mem=450G

cd ~/PEPO

ALPHA=${1:-0.1}
BETA=${2:-0.1}

uv run scripts/train.py L=8 acc_steps=2 batch_size=4 model=smollm model.alpha=${ALPHA} model.beta=${BETA}
