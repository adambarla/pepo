#!/bin/bash

# Loop over specifically failed epoch 1
echo "Submitting sftdpo alpaca jobs..."
for e in 1; do
    sbatch --gres=gpu:4 --time=03:00:00 --job-name="e_${e}_sftdpo_alpaca" \
        scripts/slurm/eval.slurm \
        evaluator=alpaca \
        model=sftdpo \
        backbone=zephyr7b \
        e=${e} \
        model.sft_weight=0.005 \
        model@ref_model=deppo \
        ref_e=0 \
        ref_L=1 \
        ref_a=0.0 \
        overwrite=false \
        sync=true
done

echo "All sftdpo alpaca jobs submitted!"
