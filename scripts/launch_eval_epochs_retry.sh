#!/bin/bash

# Loop over failed/stuck epochs
echo "Submitting retry jobs for chi2po..."
for e in 0 2 3 4 5 7 9 11 12 13 14 16; do
    sbatch --gres=gpu:4 --time=02:00:00 --job-name="e_${e}_chi2po" \
        scripts/slurm/eval.slurm \
        evaluator=mtbench \
        model=chi2po \
        backbone=zephyr7b \
        e=${e} \
        model@ref_model=deppo \
        ref_e=0 \
        ref_L=1 \
        ref_a=0.0 \
        overwrite=false \
        evaluator.overwrite_judgments=true \
        sync=true \
        wandb.mode=offline
done

echo "All retry jobs submitted!"
