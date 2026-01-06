#!/bin/bash

MODEL=""
for arg in "$@"; do
    if [[ "$arg" == model=* ]]; then
        MODEL="${arg#model=}"
        break
    fi
done

if [[ -z "$MODEL" ]]; then
    echo "Error: model= argument is required"
    exit 1
fi

# get AGAINST from arg, possible values are: [gpt, init]
AGAINST=""
for arg in "$@"; do
    if [[ "$arg" == against=* ]]; then
        AGAINST="${arg#against=}"
        if [[ $AGAINST != "gpt" && $AGAINST != "init" ]]; then
            echo "Error: against= argument must be either gpt or init"
            exit 1
        fi
        break
    fi
done

if [[ -z "$AGAINST" ]]; then
    echo "Error: against= argument is required. Possible values are: [gpt, init]."
    exit 1
fi

# Filter out script-specific arguments before passing to sbatch/Hydra
FILTERED_ARGS=()
for arg in "$@"; do
    if [[ "$arg" != model=* && "$arg" != against=* ]]; then
        FILTERED_ARGS+=("$arg")
    fi
done

# if AGAINST is gpt, loop over epochs and evaluate the L=[1-4] and epoch[0-16]
if [[ $AGAINST == "gpt" ]]; then
    for L in 1 2 3 4; do
    # for L in 4; do
        for e in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16; do
        # for e in 0 1 2 3; do
            # if L is 1 a should be 0.0
            A=0.1
            if [[ $L -eq 1 ]]; then
                A=0.0
            fi
            sbatch --job-name="ev_${MODEL}_l${L}_e${e}" scripts/slurm/eval.slurm model=$MODEL e=${e} L=${L} a=${A} "${FILTERED_ARGS[@]}"
        done
    done
fi

# if AGAINST is init, loop over epochs and L and evaluate agains initial (epoch 0 of L=1)
if [[ $AGAINST == "init" ]]; then
    # for L in 1 2 3 4; do
    for L in 1 4; do
        # for e in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16; do
        for e in 0 1 2 3 4; do
            A=0.1
            if [[ $L -eq 1 ]]; then
                A=0.0
            fi
            sbatch --job-name="ev_${MODEL}_l${L}_e${e}_init" scripts/slurm/eval.slurm model=$MODEL e=${e} L=${L} a=${A} model@ref_model=$MODEL ref_e=0 ref_L=1 ref_a=0.0 "${FILTERED_ARGS[@]}"
        done
    done
fi


# # sbatch --job-name=l4_e0_eval scripts/slurm/eval.slurm model=$MODEL e=0 L=4 a=0.0 $@
# sbatch --job-name="ev_${MODEL}_l4_e1" scripts/slurm/eval.slurm model=$MODEL e=1 L=4 a=0.1 $@ #done
# sbatch --job-name=l4_e2_eval scripts/slurm/eval.slurm model=$MODEL e=2 L=4 a=0.1 $@
# sbatch --job-name=l4_e3_eval scripts/slurm/eval.slurm model=$MODEL e=3 L=4 a=0.1 $@
# sbatch --job-name=l4_e4_eval scripts/slurm/eval.slurm model=$MODEL e=4 L=4 a=0.1 $@
# sbatch --job-name=l4_e5_eval scripts/slurm/eval.slurm model=$MODEL e=5 L=4 a=0.1 $@
# sbatch --job-name=l4_e6_eval scripts/slurm/eval.slurm model=$MODEL e=6 L=4 a=0.1 $@ #done
# sbatch --job-name=l4_e7_eval scripts/slurm/eval.slurm model=$MODEL e=7 L=4 a=0.1 $@
# sbatch --job-name=l4_e8_eval scripts/slurm/eval.slurm model=$MODEL e=8 L=4 a=0.1 $@
# sbatch --job-name=l4_e9_eval scripts/slurm/eval.slurm model=$MODEL e=9 L=4 a=0.1 $@
# sbatch --job-name=l4_e10_eval scripts/slurm/eval.slurm model=$MODEL e=10 L=4 a=0.1 $@
# sbatch --job-name=l4_e11_eval scripts/slurm/eval.slurm model=$MODEL e=11 L=4 a=0.1 $@ #done
# sbatch --job-name=l4_e12_eval scripts/slurm/eval.slurm model=$MODEL e=12 L=4 a=0.1 $@
# sbatch --job-name=l4_e13_eval scripts/slurm/eval.slurm model=$MODEL e=13 L=4 a=0.1 $@
# sbatch --job-name=l4_e14_eval scripts/slurm/eval.slurm model=$MODEL e=14 L=4 a=0.1 $@
# sbatch --job-name=l4_e15_eval scripts/slurm/eval.slurm model=$MODEL e=15 L=4 a=0.1 $@
# sbatch --job-name=l4_e16_eval scripts/slurm/eval.slurm model=$MODEL e=16 L=4 a=0.1 $@

# sbatch --job-name="ev_${MODEL}_l1_e0" scripts/slurm/eval.slurm model=$MODEL e=0 L=1 a=0.0 $@ #done
# sbatch --job-name="ev_${MODEL}_l1_e1" scripts/slurm/eval.slurm model=$MODEL e=1 L=1 a=0.0 $@ #done
# sbatch --job-name=l1_e2_eval scripts/slurm/eval.slurm model=$MODEL e=2 L=1 a=0.0 $@
# sbatch --job-name="ev_${MODEL}_l1_e3" scripts/slurm/eval.slurm model=$MODEL e=3 L=1 a=0.0 $@ #done
# sbatch --job-name=l1_e4_eval scripts/slurm/eval.slurm model=$MODEL e=4 L=1 a=0.0 $@
# sbatch --job-name=l1_e5_eval scripts/slurm/eval.slurm model=$MODEL e=5 L=1 a=0.0 $@
# sbatch --job-name=l1_e6_eval scripts/slurm/eval.slurm model=$MODEL e=6 L=1 a=0.0 $@

# EPOCH=2

# sbatch --job-name="ev_${MODEL}_l4_e${EPOCH}" scripts/slurm/eval.slurm model=$MODEL e=${EPOCH} L=4 a=0.1 $@
# sbatch --job-name="ev_${MODEL}_l3_e${EPOCH}" scripts/slurm/eval.slurm model=$MODEL e=${EPOCH} L=3 a=0.1 $@
# sbatch --job-name="ev_${MODEL}_l2_e${EPOCH}" scripts/slurm/eval.slurm model=$MODEL e=${EPOCH} L=2 a=0.1 $@
# sbatch --job-name="ev_${MODEL}_l1_e${EPOCH}" scripts/slurm/eval.slurm model=$MODEL e=${EPOCH} L=1 a=0.0 $@

# sbatch --job-name=lam_a0.5_l4_e3_eval scripts/slurm/eval.slurm model=$MODEL e=3 L=4 a=0.5 $@
# # # sbatch --job-name=lam_l4_e3_eval scripts/slurm/eval.slurm model=$MODEL e=3 L=4 a=0.1 $@
# sbatch --job-name=lam_a0.1_l4_e3_eval scripts/slurm/eval.slurm model=$MODEL e=3 L=4 a=0.1 $@
# sbatch --job-name=lam_a0.05_l4_e3_eval scripts/slurm/eval.slurm model=$MODEL e=3 L=4 a=0.05 $@
# sbatch --job-name=lam_a0.01_l4_e3_eval scripts/slurm/eval.slurm model=$MODEL e=3 L=4 a=0.01 $@
# sbatch --job-name=lam_a0.005_l4_e3_eval scripts/slurm/eval.slurm model=$MODEL e=3 L=4 a=0.005 $@
# sbatch --job-name=lam_a0.001_l4_e3_eval scripts/slurm/eval.slurm model=$MODEL e=3 L=4 a=0.001 $@
# sbatch --job-name=lam_a0.0_l4_e3_eval scripts/slurm/eval.slurm model=$MODEL e=3 L=4 a=0.0 $@

# sbatch --job-name=dpo_comp_e0 scripts/slurm/eval.slurm model=$MODEL e=0 L=1 a=0.0 model@ref_model=$MODEL ref_e=8 ref_L=4 ref_a=0.1 $@
# sbatch --job-name=dpo_comp_e1 scripts/slurm/eval.slurm model=$MODEL e=1 L=1 a=0.0 model@ref_model=$MODEL ref_e=8 ref_L=4 ref_a=0.1 $@
# sbatch --job-name=dpo_comp_e2 scripts/slurm/eval.slurm model=$MODEL e=2 L=1 a=0.0 model@ref_model=$MODEL ref_e=8 ref_L=4 ref_a=0.1 $@
# sbatch --job-name=dpo_comp_e3 scripts/slurm/eval.slurm model=$MODEL e=3 L=1 a=0.0 model@ref_model=$MODEL ref_e=8 ref_L=4 ref_a=0.1 $@
