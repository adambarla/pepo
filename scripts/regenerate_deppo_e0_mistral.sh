#!/bin/bash

# Regenerate the corrupted reference model (deppo epoch 0) responses for mistral backbone
echo "Regenerating deppo e0 mistral responses..."
sbatch --gres=gpu:4 --time=02:00:00 --job-name="deppo_e0_mistral_responses" \
    scripts/slurm/eval.slurm \
    evaluator=alpaca \
    model=deppo \
    backbone=mistral7b \
    e=0 \
    overwrite=true \
    stop_after_generation=true \
    sync=true

echo "Job submitted!"
