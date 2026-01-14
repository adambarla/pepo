#!/bin/bash
#SBATCH --job-name=pepo_benchmark
#SBATCH --output=out/slurm-%j.out
#SBATCH --error=out/slurm-%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:3
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Set HuggingFace token
export HF_TOKEN=''

# Print job info
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"

# Run the benchmark
conda run -n mnlp python LLM/run_benchmark.py \
    --evaluation_dataset arena_hard \
    --num_networks 4 \
    --hf_username PessimisticDPO \
    --model HuggingFaceTB/SmolLM2-1.7B \
    --dpo_baseline_model_id PessimisticDPO/SmolLM2-1.7Bdpo_ensemble1_l0 \
    --num_samples 100 \
    --batch_size 10 \
    --pepo_use_idk_token \
    --pepo_cuda_index 0 \
    --dpo_cuda_index 1 \
    --judge_cuda_index 2 \
    --judge_gpu_memory_utilization 0.9

# Print job completion info
echo "Job finished at: $(date)"
