#!/bin/bash
#SBATCH --job-name=pepo_L1
#SBATCH --output=out/train_L1-%j.out
#SBATCH --error=out/train_L1-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
#SBATCH --account=infra01
#SBATCH --exclusive --mem=450G

cd ~/PEPO

uenv run pytorch/v2.6.0:v1 --view=default -- bash << 'EOF'
source .venv/bin/activate
python scripts/train.py L=1 epochs=8 acc_steps=1 batch_size=8
EOF
