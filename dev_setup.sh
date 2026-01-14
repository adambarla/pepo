#!/bin/bash

set -e

# Install and configure uv (Python package manager)
echo "Checking for uv..."

CONFIG_UPDATED=false
SHELL_CONFIG=""
if [[ "$SHELL" == *"zsh"* ]]; then
    SHELL_CONFIG="$HOME/.zshrc"
elif [[ "$SHELL" == *"bash"* ]]; then
    SHELL_CONFIG="$HOME/.bashrc"
fi

if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if ! command -v uv &> /dev/null; then
        echo "Error: Failed to install uv. Please install it manually."
        echo "Visit: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi

    # Add uv installation paths to shell config for future sessions
    # Since uv is now available, add both standard installation paths
    if [ -n "$SHELL_CONFIG" ] && [ -f "$SHELL_CONFIG" ]; then
        # Add .local/bin to PATH if not already present
        PATH_LINE_LOCAL='export PATH="$HOME/.local/bin:$PATH"'
        if ! grep -Fxq "$PATH_LINE_LOCAL" "$SHELL_CONFIG"; then
            echo "" >> "$SHELL_CONFIG"
            echo "# Added by dev_setup.sh to make uv available" >> "$SHELL_CONFIG"
            echo "$PATH_LINE_LOCAL" >> "$SHELL_CONFIG"
            echo "Added $HOME/.local/bin to PATH in $SHELL_CONFIG"
            CONFIG_UPDATED=true
        fi

        # Add .cargo/bin to PATH if not already present
        PATH_LINE_CARGO='export PATH="$HOME/.cargo/bin:$PATH"'
        if ! grep -Fxq "$PATH_LINE_CARGO" "$SHELL_CONFIG"; then
            echo "$PATH_LINE_CARGO" >> "$SHELL_CONFIG"
            echo "Added $HOME/.cargo/bin to PATH in $SHELL_CONFIG"
            CONFIG_UPDATED=true
        fi

        # Source the config file to apply changes in current session
        if [ "$CONFIG_UPDATED" = true ]; then
            source "$SHELL_CONFIG" 2>/dev/null || true
        fi
    fi
else
    echo "uv already installed"
fi

# Install pre-commit and detect-secrets if not already installed
echo "Checking for pre-commit..."

if ! command -v pre-commit &> /dev/null; then
    echo "pre-commit not found. Installing with uv..."
    uv pip install pre-commit detect-secrets || pip install pre-commit detect-secrets
else
    echo "pre-commit already installed"
fi

# Install pre-commit git hooks
echo "Installing git hooks..."
uv run pre-commit install

# Optionally run pre-commit on all files (commented out by default)
# echo "Running pre-commit on all files..."
# uv run pre-commit run --all-files

echo "Pre-commit hooks installed successfully!"

# Remind user to source shell config if it was updated
if [ "$CONFIG_UPDATED" = true ] && [ -n "$SHELL_CONFIG" ]; then
    echo ""
    echo "Note: PATH has been updated in $SHELL_CONFIG"
    echo "To use 'uv' in this terminal session, run:"
    echo "  source $SHELL_CONFIG"
    echo "Or start a new terminal session."
fi
