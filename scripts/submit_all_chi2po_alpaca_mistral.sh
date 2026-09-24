#!/bin/bash

# Submit all chi2po alpaca mistral jobs (epochs 1-5)
echo "Submitting all chi2po alpaca mistral jobs..."
for e in 1 2 3 4 5; do
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

echo "All jobs submitted!"
