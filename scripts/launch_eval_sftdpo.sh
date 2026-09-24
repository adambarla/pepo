#!/bin/bash

# Loop over specifically failed epochs 1 and 3
echo "Submitting sftdpo jobs..."
for e in 1 3; do
    sbatch --gres=gpu:4 --time=02:00:00 --job-name="e_${e}_sftdpo" \
        scripts/slurm/eval.slurm \
        evaluator=mtbench \
        model=sftdpo \
        backbone=zephyr7b \
        e=${e} \
        model.sft_weight=0.005 \
        model@ref_model=deppo \
        ref_e=0 \
        ref_L=1 \
        ref_a=0.0 \
        overwrite=false \
        evaluator.overwrite_judgments=false \
        sync=true
done

echo "All jobs submitted!"
