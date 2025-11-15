#!/bin/bash

echo "Getting interactive node with 4 GPUs for 12 hours"

# Use srun directly - this creates one interactive shell
# To get additional shells, use: ./scripts/slurm/connect_to_node.sh
srun --account=infra01 --nodes=1 --time=12:00:00 --gpus=4 --job-name="pepo_interactive" --pty bash -c "
  # Print connection info
  echo \"\" >&2
  echo \"==========================================\" >&2
  echo \"Job ID: \$SLURM_JOB_ID, Node: \$SLURM_NODELIST\" >&2
  echo \"==========================================\" >&2
  echo \"\" >&2
  echo \"To get another shell on this node in another terminal, run:\" >&2
  echo \"  ./scripts/slurm/connect_to_node.sh\" >&2
  echo \"\" >&2
  echo \"Or manually with:\" >&2
  echo \"  srun --jobid=\$SLURM_JOB_ID --overlap --pty bash\" >&2
  echo \"\" >&2
  echo \"To activate uenv and venv, run:\" >&2
  echo \"  ./scripts/activate.sh\" >&2
  echo \"\" >&2

  # Start interactive bash
  exec bash
"
