#!/bin/bash

echo "Getting interactive node with 4 GPUs for 12 hours"
srun --account=infra01 --nodes=1 --time=12:00:00 --gpus=4 --pty bash
