#!/bin/bash

# Loop over only the currently failed epochs to retry online sync
echo "Submitting online sync retry jobs for failed epochs..."
for e in 0 2 3 4 5 6; do
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

echo "All retry sync jobs submitted!"
