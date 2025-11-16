#!/bin/bash
#SBATCH --job-name=pepo_eval
#SBATCH --output=out/pepo_eval-%j.out
#SBATCH --error=out/pepo_eval-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
#SBATCH --account=infra01
#SBATCH --exclusive --mem=450G

cd ~/PEPO

uv run scripts/eval.py "$@"
