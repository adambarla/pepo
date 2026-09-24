#!/bin/bash

# Resubmit failed epochs 2 and 3 for chi2po alpaca mistral
echo "Resubmitting chi2po alpaca mistral jobs for epochs 2 and 3..."
for e in 2 3; do
    sbatch --gres=gpu:4 --time=02:00:00 --job-name="e_${e}_chi2po_alpaca_mistral" \
        scripts/slurm/eval.slurm \
        evaluator=alpaca \
        model=chi2po \
        backbone=mistral7b \
        e=${e} \
        model@ref_model=deppo \
        ref_e=0 \
        ref_L=1 \
        ref_a=0.0 \
        overwrite=false \
        sync=true
done

echo "Jobs resubmitted!"
