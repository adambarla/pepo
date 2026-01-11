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

EPOCHS=""
for arg in "$@"; do
    if [[ "$arg" == e=* ]]; then
        EPOCHS="${arg#e=}"
        break
    fi
done

if [[ -z "$EPOCHS" ]]; then
    echo "Error: e= argument is required"
    exit 1
fi

JOB_NAME="tr_${MODEL}_${EPOCHS}"

# sbatch --job-name="${JOB_NAME}_1" scripts/slurm/train.slurm model=$MODEL e=$EPOCHS L=1 a=0.0 $@
# sbatch --job-name="${JOB_NAME}_4" scripts/slurm/train.slurm model=$MODEL e=$EPOCHS L=4 a=0.1 $@


# for L in 1 2 3 4; do
for L in 2 3; do
    # if L is 1 a should be 0.0
    A=0.1
    if [[ $L -eq 1 ]]; then
        A=0.0
    fi
    sbatch --job-name="${JOB_NAME}_${L}" scripts/slurm/train.slurm model=$MODEL e=$EPOCHS L=$L a=$A $@
done

# sbatch --job-name="tr_${MODEL}_${EPOCHS}_4" scripts/slurm/train.slurm model=$MODEL e=$EPOCHS L=4 a=0.1 $@
# sbatch --job-name="tr_${MODEL}_${EPOCHS}_3" scripts/slurm/train.slurm model=$MODEL e=$EPOCHS L=3 a=0.1 $@
# sbatch --job-name="tr_${MODEL}_${EPOCHS}_2" scripts/slurm/train.slurm model=$MODEL e=$EPOCHS L=2 a=0.1 $@


# sbatch --job-name=dpo_comp_e$EPOCHS scripts/slurm/train.slurm model=$MODEL e=$EPOCHS L=4 a=100 $@
# sbatch --job-name=dpo_comp_e$EPOCHS scripts/slurm/train.slurm model=$MODEL e=$EPOCHS L=4 a=10 $@
# sbatch --job-name=dpo_comp_e$EPOCHS scripts/slurm/train.slurm model=$MODEL e=$EPOCHS L=4 a=1 $@
# # sbatch --job-name=dpo_comp_e$EPOCHS scripts/slurm/train.slurm model=$MODEL e=$EPOCHS L=4 a=0.1 $@
# sbatch --job-name=dpo_comp_e$EPOCHS scripts/slurm/train.slurm model=$MODEL e=$EPOCHS L=4 a=0.01 $@
# sbatch --job-name=dpo_comp_e$EPOCHS scripts/slurm/train.slurm model=$MODEL e=$EPOCHS L=4 a=0.001 $@
# sbatch --job-name=dpo_comp_e$EPOCHS scripts/slurm/train.slurm model=$MODEL e=$EPOCHS L=4 a=0.0 $@
