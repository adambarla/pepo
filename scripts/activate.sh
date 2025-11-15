#!/bin/bash

# Script to activate uenv and venv environment
# Can be run from any directory, will change to project root

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project directory
cd "$PROJECT_ROOT" || exit 1

# Ensure uenv image is available
uenv image pull pytorch/v2.6.0:v1 2>/dev/null || true

# Start interactive shell with uenv and venv
if [ -f .venv/bin/activate ]; then
  uenv run pytorch/v2.6.0:v1 --view=default -- bash -c 'source .venv/bin/activate && exec bash'
else
  uenv run pytorch/v2.6.0:v1 --view=default -- bash
fi
