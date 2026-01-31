#!/bin/bash
#SBATCH --job-name=pepo_benchmark
#SBATCH --output=out/benchmark-%j.out
#SBATCH --error=out/benchmark-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
#SBATCH --account=infra01
#SBATCH --exclusive --mem=450G

cd ~/pepo

uv run scripts/benchmark.py "$@"
