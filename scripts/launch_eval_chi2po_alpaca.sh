#!/bin/bash

# Loop over specifically failed epochs 3 and 4 for chi2po using alpaca evaluator
echo "Submitting chi2po alpaca jobs..."
for e in 3 4; do
    sbatch --gres=gpu:4 --time=02:00:00 --job-name="e_${e}_chi2po_alpaca" \
        scripts/slurm/eval.slurm \
        evaluator=alpaca \
        model=chi2po \
        backbone=zephyr7b \
        e=${e} \
        model@ref_model=deppo \
        ref_e=0 \
        ref_L=1 \
        ref_a=0.0 \
        overwrite=false \
        sync=true
done

echo "All chi2po alpaca jobs submitted!"
