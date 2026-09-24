#!/usr/bin/env bash
# Extend the 2026-09-23 timing campaign with matched arms.
#
#   smoke        exact mixture-proposal RS (mean_std and min) on 16 prompts, E6
#   dpo          standard DPO (a=0, L=1), top-p decoding, E1--E6 x 3 seeds
#   meanstd_mix  exact mixture-proposal mean-minus-std RS, L=4, E1--E6 x 3 seeds
#
# Every arm uses the protocol of timing_campaign_2026-09-23.md: Tulu-3-8B SFT,
# all 805 AlpacaEval prompts, max_new_tokens=1024, four GH200s, generation
# batch 32, temperature 1, top-p 0.9, seeds 1001--1003, and the same e0 SFT
# reference for scoring. Usage: scripts/schedule_sampling_campaign_20260924.sh ARM

set -euo pipefail

ARM="${1:?usage: $0 smoke|dpo|meanstd_mix}"
ROOT="outputs/timing_campaign_20260923"
HYDRA_ROOT="outputs/hydra/timing_campaign_20260923"
REFERENCE="outputs/alpaca_eval/responses/Llama-3.1-Tulu-3-8B-SFT-a0.0-b0.1-L1-e0_mt1024_responses.json"
MODEL="Llama-3.1-Tulu-3-8B-SFT"

common=(
  model=deppo backbone=llama8b split_mode=disjoint
  model.shared_backbone=true
  model.generator.greedy_sampling=false
  model.generator.temperature=1.0 model.generator.top_p=0.9
  mt=1024 b=0.1
  backbone.generator_batch_size=32 backbone.eval_batch_size=32
  stop_after_generation=true overwrite=false sync=false
)

rs_args() {  # sampling_mode [eta]
  echo model.generator._target_=pepo.generator.BestOfNGenerator \
    +model.generator.max_trials=128 +model.generator.sampling_mode="$1" \
    ${2:+"+model.generator.eta=$2"}
}

submit_score() {  # gen_job_id response_file tag
  local out
  out=$(sbatch --partition=normal --time=06:00:00 --gres=gpu:4 --exclusive \
    --dependency="afterok:$1" --job-name="score_$3" \
    scripts/slurm/score.slurm \
    --model-outputs="$2" --reference-outputs="${REFERENCE}" \
    --output-dir="${ROOT}/scores/$3" --name="$3-common805" --expected-count=805)
  echo "score $3: ${out##* } after $1"
}

case "${ARM}" in
  smoke)
    for mode in mean_std min; do
      tag="smoke_mix_${mode}_e6"
      eta=""; [[ "${mode}" == mean_std ]] && eta=0.1
      # shellcheck disable=SC2046
      out=$(sbatch --partition=debug --time=00:45:00 --gres=gpu:4 --exclusive \
        --job-name="${tag}" scripts/slurm/eval.slurm "${common[@]}" \
        $(rs_args "${mode}" "${eta}") \
        e=6 L=4 a=0.1 ns=16 seed=1001 \
        evaluator.output_dir="${ROOT}/${tag}" hydra.run.dir="${HYDRA_ROOT}/${tag}")
      echo "${tag}: ${out##* }"
    done
    ;;
  dpo)
    for rep in 1 2 3; do
      for epoch in 1 2 3 4 5 6; do
        tag="dpo_e${epoch}_r${rep}"
        out=$(sbatch --partition=normal --time=04:00:00 --gres=gpu:4 --exclusive \
          --job-name="tim_${tag}" scripts/slurm/eval.slurm "${common[@]}" \
          e="${epoch}" L=1 a=0.0 ns=805 seed=$((1000 + rep)) \
          evaluator.output_dir="${ROOT}/${tag}" hydra.run.dir="${HYDRA_ROOT}/${tag}")
        echo "${tag}: ${out##* }"
        submit_score "${out##* }" \
          "${ROOT}/${tag}/alpaca_eval/responses/${MODEL}-a0.0-b0.1-L1-e${epoch}_ns805_mt1024-top-p-t1.0-p0.9_responses.json" \
          "${tag}"
      done
    done
    ;;
  meanstd_mix)
    for rep in 1 2 3; do
      for epoch in 1 2 3 4 5 6; do
        tag="meanstd_mix_e${epoch}_r${rep}"
        # shellcheck disable=SC2046
        out=$(sbatch --partition=normal --time=12:00:00 --gres=gpu:4 --exclusive \
          --job-name="tim_${tag}" scripts/slurm/eval.slurm "${common[@]}" \
          $(rs_args mean_std 0.1) \
          e="${epoch}" L=4 a=0.1 ns=805 seed=$((1000 + rep)) \
          evaluator.output_dir="${ROOT}/${tag}" hydra.run.dir="${HYDRA_ROOT}/${tag}")
        echo "${tag}: ${out##* }"
        submit_score "${out##* }" \
          "${ROOT}/${tag}/alpaca_eval/responses/${MODEL}-a0.1-b0.1-L4-e${epoch}_ns805_bon-mean_std-eta0.1-mix-n128-mt1024-top-p-t1.0-p0.9_responses.json" \
          "${tag}"
      done
    done
    ;;
  *)
    echo "unknown arm: ${ARM}" >&2
    exit 1
    ;;
esac
