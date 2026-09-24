#!/bin/bash
# Chained Slurm submission for overlapping-subset ensemble experiment.
#
# Training (L=4, Tulu-3-8B) does ~2-3 epochs per 12h walltime.
# Chains N training rounds with continue=true so each picks up where
# the last one left off, until all 16 epochs are done.
#
# Submits eval jobs per-round so early-epoch results come in days
# sooner than waiting for all 16 epochs.  A catch-all after the final
# round covers any epochs a round didn't get to.
#
# Usage:
#   ./scripts/launch_overlap_subsample.sh backbone=llama8b
#
# Monitor with: squeue -u $USER

set -euo pipefail

BACKBONE=""
for arg in "$@"; do
    if [[ "$arg" == backbone=* ]]; then
        BACKBONE="${arg#backbone=}"
        break
    fi
done
if [[ -z "$BACKBONE" ]]; then
    echo "Usage: $0 backbone=llama8b [extra Hydra overrides...]"
    exit 1
fi

MAX_ROUNDS=8
# Override these if the account's partition/QoS associations change.
TRAIN_PARTITION="${PEPO_TRAIN_PARTITION:-normal}"
TRAIN_QOS="${PEPO_TRAIN_QOS:-normal}"
EVAL_PARTITION="${PEPO_EVAL_PARTITION:-normal}"
EVAL_QOS="${PEPO_EVAL_QOS:-normal}"

COMMON_ARGS=(
    model=deppo
    e=16
    L=4
    a=0.1
    b=0.1
    backbone="$BACKBONE"
    split_mode=overlap_subsample
    model.shared_backbone=false
    # Evaluation is submitted separately below; keep training memory bounded.
    skip_eval=true
    "$@"
)

EVAL_ARGS=(
    model=deppo
    L=4
    a=0.1
    b=0.1
    backbone="$BACKBONE"
    split_mode=overlap_subsample
    model@ref_model=deppo
    ref_e=0
    ref_L=1
    ref_a=0.0
    ref_b=0.1
    "$@"
)

echo "=== Submitting overlap_subsample training (${MAX_ROUNDS} rounds) ==="
declare -a TRAIN_JOB_IDS=()
PREV_JOB_ID=""

for ROUND in $(seq 1 "$MAX_ROUNDS"); do
    JOB_NAME="overlap_tr_${BACKBONE}_r${ROUND}"

    # Continue from the latest checkpoint when one exists; the trainer falls
    # back to fresh initialization for the first round when none exists.
    ROUND_ARGS=("${COMMON_ARGS[@]}" continue=true)
    if [[ "$ROUND" -eq 1 ]]; then
        DEP_ARGS=()
    else
        DEP_ARGS=(--dependency="afterany:${PREV_JOB_ID}")
    fi

    OUT=$(sbatch --requeue --partition="$TRAIN_PARTITION" --qos="$TRAIN_QOS" "${DEP_ARGS[@]}" --job-name="$JOB_NAME" \
        scripts/slurm/train.slurm "${ROUND_ARGS[@]}")
    echo "$OUT"
    JOB_ID=$(echo "$OUT" | grep -oP '\d+')
    TRAIN_JOB_IDS+=("$JOB_ID")
    PREV_JOB_ID="$JOB_ID"
done

echo "=== Submitting per-round evaluations ==="
# Round N → evals for epochs (N-1)*3+1 .. N*3 (clamped to 1..16)
# Each batch chains after the corresponding training round.
for ROUND in $(seq 1 "$MAX_ROUNDS"); do
    START=$(( (ROUND - 1) * 3 + 1 ))
    END=$(( ROUND * 3 ))
    [[ "$START" -gt 16 ]] && continue
    END=$(( END > 16 ? 16 : END ))

    for e in $(seq "$START" "$END"); do
        # Training rounds may end at the wall-time limit after saving a checkpoint.
        sbatch --requeue --partition="$EVAL_PARTITION" --qos="$EVAL_QOS" \
            --dependency="afterany:${TRAIN_JOB_IDS[$ROUND-1]}" \
            --job-name="overlap_ev_${BACKBONE}_e${e}" \
            scripts/slurm/eval.slurm \
            "${EVAL_ARGS[@]}" e="$e"
    done
done

echo "=== Submitting catch-all evaluations (after final training round) ==="
# If a round produced fewer epochs than expected, the per-round eval for the
# final epoch(s) in its range would fail.  This catch-all runs after the
# last training round guarantees every epoch is evaluated.
LAST_TRAIN=${TRAIN_JOB_IDS[$MAX_ROUNDS-1]}
for e in $(seq 1 16); do
    sbatch --requeue --partition="$EVAL_PARTITION" --qos="$EVAL_QOS" \
        --dependency="afterany:${LAST_TRAIN}" \
        --job-name="overlap_ev_${BACKBONE}_e${e}" \
        scripts/slurm/eval.slurm \
        "${EVAL_ARGS[@]}" e="$e"
done

echo "Done. Monitor with: squeue -u \$USER"
