#!/bin/bash
#SBATCH --job-name=pepo_L4_smollm
#SBATCH --output=out/train_L4-%j.out
#SBATCH --error=out/train_L4-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
#SBATCH --account=infra01
#SBATCH --exclusive --mem=450G

cd ~/PEPO

ALPHA=${1:-0.1}
BETA=${2:-0.1}

uv run scripts/train.py L=4 acc_steps=1 batch_size=8 model=smollm model.alpha=${ALPHA} model.beta=${BETA}
