#!/bin/bash
#SBATCH --job-name=pepo_train
#SBATCH --output=out/pepo_train-%j.out
#SBATCH --error=out/pepo_train-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --account=infra01
# Note: Normal partition max is 12:00:00. For 24 hours, you may need a different partition.
# For GH200 nodes, consider: --ntasks-per-node=4 --gpus-per-task=1 (one rank per GPU)

cd ~/PEPO


uenv run pytorch/v2.6.0:v1 --view=default -- bash << 'EOF'
source .venv/bin/activate
python scripts/train.py L=4
EOF
