#!/bin/bash
# Chained Slurm submission for the margin-only shifted-sigmoid DPO control.
#
# The experiment uses one model (L=1) with a=0.1, compared during evaluation
# against the initial standard-DPO reference (L=1, a=0.0).  Training is
# split into 12-hour rounds so each round can resume from the previous
# checkpoint after a timeout.
#
# Usage:
#   ./scripts/launch_margin.sh backbone=mistral7b
#
# Monitor with: squeue -u "$USER"

set -euo pipefail

BACKBONE=""
for arg in "$@"; do
    if [[ "$arg" == backbone=* ]]; then
        BACKBONE="${arg#backbone=}"
        break
    fi
done

if [[ -z "$BACKBONE" ]]; then
    echo "Usage: $0 backbone=mistral7b [extra Hydra overrides...]" >&2
    exit 1
fi

MAX_ROUNDS=8
TARGET_EPOCH=16
# Override these if the account's partition/QoS associations change.
TRAIN_PARTITION="${PEPO_TRAIN_PARTITION:-normal}"
TRAIN_QOS="${PEPO_TRAIN_QOS:-normal}"
EVAL_PARTITION="${PEPO_EVAL_PARTITION:-normal}"
EVAL_QOS="${PEPO_EVAL_QOS:-normal}"

TRAIN_ARGS=(
    model=deppo
    e="$TARGET_EPOCH"
    L=1
    a=0.1
    b=0.1
    backbone="$BACKBONE"
    split_mode=disjoint
    skip_eval=true
    seed=42
    sync=true
    "$@"
)

EVAL_ARGS=(
    model=deppo
    L=1
    a=0.1
    b=0.1
    backbone="$BACKBONE"
    split_mode=disjoint
    evaluator=alpaca
    model@ref_model=deppo
    ref_e=0
    ref_L=1
    ref_a=0.0
    ref_b=0.1
    seed=42
    sync=true
    "$@"
)

echo "=== Submitting margin training (${MAX_ROUNDS} rounds) ==="
declare -a TRAIN_JOB_IDS=()
PREV_JOB_ID=""

for ROUND in $(seq 1 "$MAX_ROUNDS"); do
    JOB_NAME="margin_tr_${BACKBONE}_r${ROUND}"

    # Continue from the latest checkpoint when one exists; the trainer falls
    # back to fresh initialization for the first round when none exists.
    ROUND_ARGS=("${TRAIN_ARGS[@]}" continue=true)
    if [[ "$ROUND" -eq 1 ]]; then
        DEP_ARGS=()
    else
        DEP_ARGS=(--dependency="afterany:${PREV_JOB_ID}")
    fi

    JOB_ID=$(sbatch --parsable --requeue --partition="$TRAIN_PARTITION" --qos="$TRAIN_QOS" "${DEP_ARGS[@]}" --job-name="$JOB_NAME" \
        scripts/slurm/train.slurm "${ROUND_ARGS[@]}")
    JOB_ID="${JOB_ID%%;*}"
    echo "Submitted $JOB_NAME ($JOB_ID)"
    TRAIN_JOB_IDS+=("$JOB_ID")
    PREV_JOB_ID="$JOB_ID"
done

echo "=== Submitting per-round evaluations ==="
# Round N covers epochs (N-1)*3+1 through N*3, clamped to 1..16.
for ROUND in $(seq 1 "$MAX_ROUNDS"); do
    START=$(( (ROUND - 1) * 3 + 1 ))
    END=$(( ROUND * 3 ))
    [[ "$START" -gt "$TARGET_EPOCH" ]] && continue
    END=$(( END > TARGET_EPOCH ? TARGET_EPOCH : END ))

    for EPOCH in $(seq "$START" "$END"); do
        JOB_NAME="margin_ev_${BACKBONE}_e${EPOCH}"
        sbatch --parsable --requeue --partition="$EVAL_PARTITION" --qos="$EVAL_QOS" --time=04:00:00 \
            --dependency="afterok:${TRAIN_JOB_IDS[$ROUND-1]}" \
            --job-name="$JOB_NAME" \
            scripts/slurm/eval.slurm \
            "${EVAL_ARGS[@]}" e="$EPOCH" >/dev/null
        echo "Submitted $JOB_NAME"
    done
done

echo "=== Submitting final fallback evaluations ==="
# These cover epochs whose per-round evaluation was skipped because a training
# round timed out. Epoch 0 is the reference baseline and is not reevaluated.
LAST_TRAIN=${TRAIN_JOB_IDS[$MAX_ROUNDS-1]}
for EPOCH in $(seq 1 "$TARGET_EPOCH"); do
    JOB_NAME="margin_ev_${BACKBONE}_final_e${EPOCH}"
    sbatch --parsable --requeue --partition="$EVAL_PARTITION" --qos="$EVAL_QOS" --time=04:00:00 \
        --dependency="afterany:${LAST_TRAIN}" \
        --job-name="$JOB_NAME" \
        scripts/slurm/eval.slurm \
        "${EVAL_ARGS[@]}" e="$EPOCH" >/dev/null
    echo "Submitted $JOB_NAME"
done

echo "Done. Monitor with: squeue -u \$USER"
