#!/bin/bash
#SBATCH --job-name=pepo_L4
#SBATCH --output=out/pepo_train-%j.out
#SBATCH --error=out/pepo_train-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --account=infra01

cd ~/PEPO

uenv run pytorch/v2.6.0:v1 --view=default -- bash << 'EOF'
source .venv/bin/activate
python scripts/train.py L=4 epochs=8 acc_steps=1 batch_size=8
EOF
