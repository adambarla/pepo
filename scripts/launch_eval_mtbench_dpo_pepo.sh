#!/bin/bash
# MT-Bench single-score eval for deppo (DPO) and reppo (PEPO) across all backbones.
# Subsampled epochs. Reuses cached generations where available.
set -euo pipefail
cd "$(dirname "$0")/.."

COMMON=(
  evaluator=mtbench
  overwrite=false
  evaluator.overwrite_judgments=false
  evaluator.judgment_tag=scores
  sync=true
)

submit_job() {
  local model=$1 bb=$2 e=$3 time=$4
  sbatch --gres=gpu:4 --time="$time" --job-name="mtb_sc_${model}_${bb}_e${e}" \
    scripts/slurm/eval.slurm \
    "${COMMON[@]}" \
    model="$model" backbone="$bb" e="$e"
}

echo "=== deppo (DPO) ==="
# zephyr7b, mistral7b, llama8b: L=4, epochs 0-16 → subsample e0/e4/e8/e12/e16
for bb in zephyr7b mistral7b llama8b; do
  for e in 0 4 8 12 16; do
    submit_job deppo "$bb" "$e" "01:30:00"
  done
done

# llama3b, smollm, yi34b: L=4, epochs 0-8 → subsample e0/e2/e4/e6/e8
for bb in llama3b_instruct smollm yi34b; do
  time="01:30:00"
  [ "$bb" = "yi34b" ] && time="03:00:00"
  for e in 0 2 4 6 8; do
    submit_job deppo "$bb" "$e" "$time"
  done
done

echo ""
echo "=== reppo (PEPO) ==="
# zephyr7b, mistral7b, llama8b, smollm: L=4, epochs 0-8 → subsample e0/e2/e4/e6/e8
for bb in zephyr7b mistral7b llama8b smollm; do
  for e in 0 2 4 6 8; do
    submit_job reppo "$bb" "$e" "01:30:00"
  done
done

# yi34b reppo: only epochs 0,2
for e in 0 2; do
  submit_job reppo yi34b "$e" "03:00:00"
done

echo ""
echo "All deppo + reppo MT-Bench jobs submitted."
