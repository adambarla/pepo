#!/bin/bash
#SBATCH --job-name=pepo_eval
#SBATCH --partition=long
#SBATCH --output=outputs/slurm/eval/%j.out
#SBATCH --error=outputs/slurm/eval/%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4
#SBATCH --account=infra01
#SBATCH --exclusive --mem=450G

cd ~/pepo

uv run scripts/eval.py "$@"
