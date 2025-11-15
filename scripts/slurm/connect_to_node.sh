#!/bin/bash

# Script to connect to an active interactive node
# Uses squeue to detect active interactive jobs (pepo_interactive)
# Supports multiple active sessions

# Query squeue for interactive jobs with the pepo_interactive job name
# Format: JobID NodeName
declare -a JOB_IDS
declare -a NODES

# Get all running interactive jobs for current user
while IFS= read -r line; do
  if [ -z "$line" ]; then
    continue
  fi

  job_id=$(echo "$line" | awk '{print $1}')
  node=$(echo "$line" | awk '{print $2}')

  if [ -n "$job_id" ] && [ -n "$node" ]; then
    JOB_IDS+=("$job_id")
    NODES+=("$node")
  fi
done < <(squeue --user="$USER" --name="pepo_interactive" --format="%i %N" --noheader 2>/dev/null)

if [ ${#JOB_IDS[@]} -eq 0 ]; then
  echo "Error: No active interactive jobs found."
  echo "Please run ./scripts/slurm/get_interactive.sh first to allocate a node."
  exit 1
fi

# If only one job, use it directly
if [ ${#JOB_IDS[@]} -eq 1 ]; then
  JOB_ID="${JOB_IDS[0]}"
  NODE="${NODES[0]}"
  echo "Connecting to job $JOB_ID on node $NODE"
else
  # Show menu for multiple jobs
  echo "Multiple active jobs found:"
  echo ""
  for i in "${!JOB_IDS[@]}"; do
    idx=$((i + 1))
    echo "  [$idx] Job ${JOB_IDS[$i]} on node ${NODES[$i]}"
  done
  echo ""

  # Get user selection with validation loop
  while true; do
    read -p "Select job number (1-${#JOB_IDS[@]}): " selection
    if [[ "$selection" =~ ^[0-9]+$ ]]; then
      if [ "$selection" -ge 1 ] && [ "$selection" -le "${#JOB_IDS[@]}" ]; then
        idx=$((selection - 1))
        JOB_ID="${JOB_IDS[$idx]}"
        NODE="${NODES[$idx]}"
        echo "Connecting to job $JOB_ID on node $NODE"
        break
      else
        echo "Invalid selection. Please enter a number between 1 and ${#JOB_IDS[@]}."
      fi
    else
      echo "Invalid input. Please enter a number."
    fi
  done
fi

# Verify the job is still running before connecting
if ! squeue --job="$JOB_ID" --format="%i" --noheader > /dev/null 2>&1; then
  echo "Error: Job $JOB_ID is no longer running."
  echo "The job may have expired or finished. Please run the script again to see current jobs."
  exit 1
fi

# Get a new shell on the allocated node
echo ""
echo "To activate uenv and venv, run:"
echo "  ./scripts/activate.sh"
echo ""
srun --jobid="$JOB_ID" --overlap --pty bash
