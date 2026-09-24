#!/usr/bin/env bash
# Replace the canceled token-level greedy arm with matched top-p sampling.

set -euo pipefail

ROOT="outputs/timing_campaign_20260923"
HYDRA_ROOT="outputs/hydra/timing_campaign_20260923"
REFERENCE="outputs/alpaca_eval/responses/Llama-3.1-Tulu-3-8B-SFT-a0.0-b0.1-L1-e0_mt1024_responses.json"

submit_common=(
  --partition=normal
  --time=06:00:00
  --gres=gpu:4
  --exclusive
)

declare -A JOB_IDS

for rep in 1 2 3; do
  seed=$((1000 + rep))
  for epoch in 1 2 3 4 5 6; do
    out="${ROOT}/token_e${epoch}_r${rep}"
    hydra="${HYDRA_ROOT}/token_e${epoch}_r${rep}"
    submission=$(sbatch "${submit_common[@]}" \
      --job-name="tim_tokp_e${epoch}_r${rep}" \
      scripts/slurm/eval.slurm \
      model=deppo backbone=llama8b split_mode=disjoint \
      model.shared_backbone=true \
      model.generator.greedy_sampling=false \
      model.generator.temperature=1.0 model.generator.top_p=0.9 \
      e="${epoch}" L=4 a=0.1 b=0.1 \
      ns=805 mt=1024 seed="${seed}" \
      backbone.generator_batch_size=32 backbone.eval_batch_size=32 \
      stop_after_generation=true overwrite=false sync=false \
      evaluator.output_dir="${out}" \
      hydra.run.dir="${hydra}")
    job_id="${submission##* }"
    JOB_IDS["${epoch}_${rep}"]="${job_id}"
    echo "token-top-p e${epoch} r${rep}: ${job_id}"
  done
done

for rep in 1 2 3; do
  for epoch in 1 2 3 4 5 6; do
    gen_job_id="${JOB_IDS["${epoch}_${rep}"]}"
    response_file="${ROOT}/token_e${epoch}_r${rep}/alpaca_eval/responses/Llama-3.1-Tulu-3-8B-SFT-a0.1-b0.1-L4-e${epoch}_ns805_mt1024-top-p-t1.0-p0.9_responses.json"
    score_submission=$(sbatch "${submit_common[@]}" \
      --dependency="afterany:${gen_job_id}" \
      --job-name="score_tokp_e${epoch}_r${rep}" \
      scripts/slurm/score.slurm \
      --model-outputs="${response_file}" \
      --reference-outputs="${REFERENCE}" \
      --output-dir="${ROOT}/scores/token_e${epoch}_r${rep}" \
      --name="token-top-p-e${epoch}-r${rep}-common805" \
      --expected-count=805)
    score_job_id="${score_submission##* }"
    echo "score token-top-p e${epoch} r${rep}: ${score_job_id} after ${gen_job_id}"
  done
done
