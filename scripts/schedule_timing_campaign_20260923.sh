#!/usr/bin/env bash
# Submit the controlled full-AlpacaEval latency campaign.
#
# Protocol:
#   - Tulu-3-8B SFT, L=4, disjoint ensemble, epochs 1--6
#   - all 805 AlpacaEval prompts, max_new_tokens=1024
#   - four GH200 GPUs, bfloat16, fixed generation batch size 32
#   - token-level top-p decoding versus practical mean-minus-std RS
#   - mean-minus-std: eta=.1, max_trials=128
#   - three independent, explicitly seeded repetitions per cell
#
# Outputs are separated by method/epoch/repetition so repetitions cannot
# overwrite one another.  Existing response files are reused. Slurm
# stdout/stderr remain in outputs/slurm/eval and outputs/slurm/score.

set -euo pipefail

ROOT="outputs/timing_campaign_20260923"
HYDRA_ROOT="outputs/hydra/timing_campaign_20260923"
REFERENCE="outputs/alpaca_eval/responses/Llama-3.1-Tulu-3-8B-SFT-a0.0-b0.1-L1-e0_mt1024_responses.json"

token_submit_common=(
  --partition=normal
  --time=06:00:00
  --gres=gpu:4
  --exclusive
)

rs_submit_common=(
  --partition=normal
  --time=12:00:00
  --gres=gpu:4
  --exclusive
)

score_submit_common=(
  --partition=normal
  --time=06:00:00
  --gres=gpu:4
  --exclusive
)

declare -A TOKEN_JOB_IDS
declare -A RS_JOB_IDS

for rep in 1 2 3; do
  seed=$((1000 + rep))
  for epoch in 1 2 3 4 5 6; do
    out="${ROOT}/token_e${epoch}_r${rep}"
    hydra="${HYDRA_ROOT}/token_e${epoch}_r${rep}"
    token_submission=$(sbatch "${token_submit_common[@]}" \
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
    token_job_id="${token_submission##* }"
    TOKEN_JOB_IDS["${epoch}_${rep}"]="${token_job_id}"
    echo "token e${epoch} r${rep}: ${token_job_id}"
  done

  for epoch in 1 2 3 4 5 6; do
    out="${ROOT}/meanstd_e${epoch}_r${rep}"
    hydra="${HYDRA_ROOT}/meanstd_e${epoch}_r${rep}"
    rs_submission=$(sbatch "${rs_submit_common[@]}" \
      --job-name="tim_rs_e${epoch}_r${rep}" \
      scripts/slurm/eval.slurm \
      model=deppo backbone=llama8b split_mode=disjoint \
      model.shared_backbone=true \
      model.generator._target_=pepo.generator.BestOfNGenerator \
      model.generator.greedy_sampling=false \
      model.generator.temperature=1.0 model.generator.top_p=0.9 \
      +model.generator.max_trials=128 \
      +model.generator.sampling_mode=mean_std \
      +model.generator.eta=0.1 \
      e="${epoch}" L=4 a=0.1 b=0.1 \
      ns=805 mt=1024 seed="${seed}" \
      backbone.generator_batch_size=32 backbone.eval_batch_size=32 \
      stop_after_generation=true overwrite=false sync=false \
      evaluator.output_dir="${out}" \
      hydra.run.dir="${hydra}")
    rs_job_id="${rs_submission##* }"
    RS_JOB_IDS["${epoch}_${rep}"]="${rs_job_id}"
    echo "meanstd e${epoch} r${rep}: ${rs_job_id}"
  done
done

echo "=== Submitting full AlpacaEval win-rate jobs ==="
for rep in 1 2 3; do
  for epoch in 1 2 3 4 5 6; do
    for method in token meanstd; do
      if [[ "${method}" == token ]]; then
        gen_job_id="${TOKEN_JOB_IDS["${epoch}_${rep}"]}"
      else
        gen_job_id="${RS_JOB_IDS["${epoch}_${rep}"]}"
      fi
      response_dir="${ROOT}/${method}_e${epoch}_r${rep}/alpaca_eval/responses"
      if [[ "${method}" == token ]]; then
        response_file="${response_dir}/Llama-3.1-Tulu-3-8B-SFT-a0.1-b0.1-L4-e${epoch}_ns805_mt1024-top-p-t1.0-p0.9_responses.json"
      else
        response_file="${response_dir}/Llama-3.1-Tulu-3-8B-SFT-a0.1-b0.1-L4-e${epoch}_ns805_bon-mean_std-eta0.1-n128-mt1024-top-p-t1.0-p0.9_responses.json"
      fi
      score_dir="${ROOT}/scores/${method}_e${epoch}_r${rep}"
      score_name="${method}-e${epoch}-r${rep}-common805"
      score_submission=$(sbatch "${score_submit_common[@]}" \
        --dependency="afterany:${gen_job_id}" \
        --job-name="score_${method:0:3}p_e${epoch}_r${rep}" \
        scripts/slurm/score.slurm \
        --model-outputs="${response_file}" \
        --reference-outputs="${REFERENCE}" \
        --output-dir="${score_dir}" \
        --name="${score_name}" \
        --expected-count=805)
      score_job_id="${score_submission##* }"
      echo "score ${method} e${epoch} r${rep}: ${score_job_id} after ${gen_job_id}"
    done
  done
done
