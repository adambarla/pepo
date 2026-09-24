#!/bin/bash
# MT-Bench single-answer grading (FastChat "scores" mode).
# Reuses cached generations under outputs/mt_bench/responses/.
# Writes judgments to outputs/mt_bench/default/ (does not touch pairwise ref folders).
# W&B metrics use score/* keys (win_rate/* from pairwise runs are left intact).

set -euo pipefail
cd "$(dirname "$0")/.."

COMMON=(
  evaluator=mtbench
  backbone=zephyr7b
  overwrite=false
  evaluator.overwrite_judgments=false
  evaluator.judgment_tag=scores
  sync=true
)

echo "Submitting MT-Bench single-score jobs (reuse generations)..."

# Baseline init (deppo e0)
sbatch --gres=gpu:4 --time=01:30:00 --job-name="mtb_sc_deppo_e0" \
  scripts/slurm/eval.slurm \
  "${COMMON[@]}" \
  model=deppo \
  e=0 \
  L=1 \
  a=0.0

# chi2po epochs 0-16
for e in {0..16}; do
  sbatch --gres=gpu:4 --time=01:30:00 --job-name="mtb_sc_chi2po_e${e}" \
    scripts/slurm/eval.slurm \
    "${COMMON[@]}" \
    model=chi2po \
    e="${e}"
done

# sftdpo epochs 0-16
for e in {0..16}; do
  sbatch --gres=gpu:4 --time=01:30:00 --job-name="mtb_sc_sftdpo_e${e}" \
    scripts/slurm/eval.slurm \
    "${COMMON[@]}" \
    model=sftdpo \
    model.sft_weight=0.005 \
    e="${e}"
done

echo "Submitted 35 MT-Bench single-score jobs."
echo "Answers: outputs/mt_bench/responses/*_answers.jsonl (reused)"
echo "Scores:  outputs/mt_bench/default/judgments/*_scores_judgments.jsonl"
