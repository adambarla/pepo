#!/usr/bin/env bash
# Quick rejection-sampling diagnostic using existing PEPO checkpoints.
#
# Smoke test, generation only:
#   scripts/launch_rejection_sampling_quick.sh \
#       backbone=mistral7b phase=smoke
#
# Small quality test:
#   scripts/launch_rejection_sampling_quick.sh \
#       backbone=mistral7b phase=quality

set -euo pipefail

BACKBONE=""
PHASE="smoke"
EPOCHS="1,3"
NS=""
TRIALS=""
MT=""
SAMPLING_MODES=""
INCLUDE_TOKEN=""
SYNC="true"

for arg in "$@"; do
    case "$arg" in
        backbone=*) BACKBONE="${arg#backbone=}" ;;
        phase=*) PHASE="${arg#phase=}" ;;
        epochs=*) EPOCHS="${arg#epochs=}" ;;
        ns=*) NS="${arg#ns=}" ;;
        trials=*) TRIALS="${arg#trials=}" ;;
        mt=*) MT="${arg#mt=}" ;;
        sampling_modes=*) SAMPLING_MODES="${arg#sampling_modes=}" ;;
        include_token=*) INCLUDE_TOKEN="${arg#include_token=}" ;;
        sync=*) SYNC="${arg#sync=}" ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$BACKBONE" ]]; then
    echo "Usage: $0 backbone=mistral7b [phase=smoke|quality]" >&2
    exit 1
fi

case "$PHASE" in
    smoke)
        NS="${NS:-8}"
        TRIALS="${TRIALS:-4}"
        MT="${MT:-256}"
        SAMPLING_MODES="${SAMPLING_MODES:-min,mean_std}"
        INCLUDE_TOKEN="${INCLUDE_TOKEN:-false}"
        STOP_AFTER_GENERATION=true
        ;;
    quality)
        NS="${NS:-32}"
        TRIALS="${TRIALS:-16}"
        MT="${MT:-1024}"
        SAMPLING_MODES="${SAMPLING_MODES:-mean_std}"
        INCLUDE_TOKEN="${INCLUDE_TOKEN:-true}"
        STOP_AFTER_GENERATION=false
        ;;
    *)
        echo "phase must be smoke or quality" >&2
        exit 1
        ;;
esac

IFS=',' read -r -a EPOCH_LIST <<< "$EPOCHS"
IFS=',' read -r -a MODE_LIST <<< "$SAMPLING_MODES"

COMMON_ARGS=(
    model=deppo
    backbone="$BACKBONE"
    L=4
    a=0.1
    b=0.1
    split_mode=disjoint
    evaluator=alpaca
    ns="$NS"
    mt="$MT"
    sync="$SYNC"
    stop_after_generation="$STOP_AFTER_GENERATION"
    overwrite=true
)

submit() {
    local name="$1"
    shift
    echo "Submitting $name"
    sbatch --job-name="$name" scripts/slurm/eval.slurm "${COMMON_ARGS[@]}" "$@"
}

for epoch in "${EPOCH_LIST[@]}"; do
    if [[ "$INCLUDE_TOKEN" == "true" ]]; then
        submit "rsq_${BACKBONE}_tok_e${epoch}" e="$epoch"
    fi

    for sampling_mode in "${MODE_LIST[@]}"; do
        submit "rsq_${BACKBONE}_${sampling_mode}_e${epoch}" \
            e="$epoch" \
            model.generator._target_=pepo.generator.BestOfNGenerator \
            model.generator.greedy_sampling=false \
            model.generator.temperature=1.0 \
            model.generator.top_p=0.9 \
            "+model.generator.max_trials=${TRIALS}" \
            "+model.generator.sampling_mode=${sampling_mode}" \
            "+model.generator.eta=0.1"
    done
done

cat <<EOF
Submitted rejection-sampling diagnostic.
Backbone: $BACKBONE
Epochs: ${EPOCHS//,/, }
Phase: $PHASE
Samples per run: $NS
Max trials: $TRIALS
Max new tokens: $MT
Sampling modes: ${SAMPLING_MODES//,/, }
Generation only: $STOP_AFTER_GENERATION
EOF
