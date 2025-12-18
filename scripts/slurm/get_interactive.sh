#!/bin/bash

# Parse arguments
LONG_JOB=false
for arg in "$@"; do
    if [[ "$arg" == "--long" ]]; then
        LONG_JOB=true
        break
    fi
done

# Set partition and time based on --long flag
if [[ "$LONG_JOB" == true ]]; then
    PARTITION="long"
    TIME="48:00:00"
    echo "Getting interactive node with 4 GPUs for 48 hours (long partition)"
else
    PARTITION="normal"
    TIME="12:00:00"
    echo "Getting interactive node with 4 GPUs for 12 hours (normal partition)"
fi

# Use srun directly - this creates one interactive shell
# To get additional shells, use: ./scripts/slurm/connect_to_node.sh
srun --account=infra01 -p $PARTITION --nodes=1 --time=$TIME --gpus=4 \
     --exclusive --job-name="interactive" --pty bash -c "
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
  echo \"Use 'uv run' to execute scripts directly.\" >&2
  echo \"\" >&2

  # Start interactive bash
  exec bash
"
