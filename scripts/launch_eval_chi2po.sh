#!/bin/bash

# Loop over specifically failed epochs for chi2po
echo "Submitting chi2po jobs..."
for e in 3 5 9 11 13 14 16; do
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
        evaluator.overwrite_judgments=false \
        sync=true
done

echo "All jobs submitted!"
