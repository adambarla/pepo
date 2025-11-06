#!/bin/bash

echo "Getting interactive node with 4 GPUs for 1 hour"
srun --account=infra01 --nodes=1 --time=01:00:00 --gpus=4 --pty bash
