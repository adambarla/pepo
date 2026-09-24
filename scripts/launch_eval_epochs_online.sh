#!/bin/bash

# Loop over all epochs 0 to 16 for chi2po
echo "Submitting online sync jobs for all epochs..."
for e in {0..16}; do
    sbatch --gres=gpu:4 --time=00:30:00 --job-name="e_${e}_chi2po" \
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

echo "All online sync jobs submitted!"
