#!/bin/bash
mkdir -p outputs/nohup/train
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="outputs/nohup/train/${TIMESTAMP}.log"

echo "Starting training..."
echo "Logging to $LOG_FILE"

nohup uv run scripts/train.py "$@" > "$LOG_FILE" 2>&1 &
PID=$!
echo "Process ID: $PID"
