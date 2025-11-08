#!/bin/bash

echo "Getting interactive node with 4 GPUs for 2 hours"
srun --account=infra01 --nodes=1 --time=02:00:00 --gpus=4 --pty bash
