#!/bin/bash

set -e

echo "Checking for uv..."

if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    if ! command -v uv &> /dev/null; then
        echo "Error: Failed to install uv. Please install it manually."
        echo "Visit: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
else
    echo "uv already installed"
fi

echo "Checking for pre-commit..."

if ! command -v pre-commit &> /dev/null; then
    echo "pre-commit not found. Installing with uv..."
    uv pip install pre-commit detect-secrets || pip install pre-commit detect-secrets
else
    echo "pre-commit already installed"
fi

echo "Installing git hooks..."
uv run pre-commit install

echo "Running pre-commit on all files..."
uv run pre-commit run --all-files

echo "✔ Pre-commit hooks installed successfully!"
