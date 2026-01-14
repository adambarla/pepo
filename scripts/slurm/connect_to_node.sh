#!/bin/bash

# Script to connect to an active interactive node
# Uses squeue to detect active interactive jobs (pepo_interactive)
# Supports multiple active sessions

# Function to truncate string to max length, adding "..." if truncated
truncate_string() {
  local str="$1"
  local max_len="$2"
  if [ ${#str} -le "$max_len" ]; then
    echo "$str"
  else
    echo "${str:0:$((max_len - 3))}..."
  fi
}

# Query squeue for interactive jobs with the pepo_interactive job name
# Format: JobID JobName NodeName StartTime TimeUsed State Partition TimeRemaining
declare -a JOB_IDS
declare -a JOB_NAMES
declare -a NODES
declare -a START_TIMES
declare -a TIME_USED
declare -a STATES
declare -a PARTITIONS
declare -a TIME_REMAINING

# Get all running interactive jobs for current user
while IFS= read -r line; do
  if [ -z "$line" ]; then
    continue
  fi

  job_id=$(echo "$line" | awk '{print $1}')
  job_name=$(echo "$line" | awk '{print $2}')
  node=$(echo "$line" | awk '{print $3}')
  start_time=$(echo "$line" | awk '{print $4}')
  time_used=$(echo "$line" | awk '{print $5}')
  state=$(echo "$line" | awk '{print $6}')
  partition=$(echo "$line" | awk '{print $7}')
  time_remaining=$(echo "$line" | awk '{print $8}')

  if [ -n "$job_id" ] && [ -n "$node" ]; then
    JOB_IDS+=("$job_id")
    JOB_NAMES+=("$job_name")
    NODES+=("$node")
    START_TIMES+=("$start_time")
    TIME_USED+=("$time_used")
    STATES+=("$state")
    PARTITIONS+=("$partition")
    TIME_REMAINING+=("$time_remaining")
  fi
done < <(squeue --user="$USER" --format="%i %j %N %S %M %T %P %L" --noheader 2>/dev/null)

if [ ${#JOB_IDS[@]} -eq 0 ]; then
  echo "Error: No active interactive jobs found."
  echo "Please run ./scripts/slurm/get_interactive.sh first to allocate a node."
  exit 1
fi

# If only one job, use it directly
if [ ${#JOB_IDS[@]} -eq 1 ]; then
  JOB_ID="${JOB_IDS[0]}"
  NODE="${NODES[0]}"
  echo "Connecting to job $JOB_ID (${JOB_NAMES[0]}) on node $NODE"
  echo "  Start Time: ${START_TIMES[0]}"
  echo "  Time Remaining: ${TIME_REMAINING[0]}"
else
  # Show menu for multiple jobs with detailed information
  echo "Multiple active jobs found:"
  echo ""
  printf "  %-4s %-10s %-20s %-10s %-10s %-10s\n" \
    "IDX" "Job ID" "Job Name" "Node" "Start" "Left"
  printf "  %s\n" "---------------------------------------------------------------------"
  for i in "${!JOB_IDS[@]}"; do
    idx=$((i + 1))
    job_id_trunc=$(truncate_string "${JOB_IDS[$i]}" 10)
    job_name_trunc=$(truncate_string "${JOB_NAMES[$i]}" 20)
    node_trunc=$(truncate_string "${NODES[$i]}" 10)
    start_time_trunc=$(truncate_string "${START_TIMES[$i]}" 10)
    time_remaining_trunc=$(truncate_string "${TIME_REMAINING[$i]}" 10)
    printf "  %-4s %-10s %-20s %-10s %-10s %-10s\n" \
      "[$idx]" \
      "$job_id_trunc" \
      "$job_name_trunc" \
      "$node_trunc" \
      "$start_time_trunc" \
      "$time_remaining_trunc"
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
        echo ""
        echo "Connecting to job $JOB_ID (${JOB_NAMES[$idx]}) on node $NODE"
        echo "  Start Time: ${START_TIMES[$idx]}"
        echo "  Time Remaining: ${TIME_REMAINING[$idx]}"
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
echo "Use 'uv run' to execute scripts directly."
echo ""
srun --jobid="$JOB_ID" --overlap --pty bash
