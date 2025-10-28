# PEPO - Preference Optimization Project

A project for preference alignment of language models using techniques like DPO (Direct Preference Optimization), RLHF (Reinforcement Learning from Human Feedback), and PEPO.

## Installation

### Using UV (Recommended)

```bash
uv sync --all-extras
```

### Using Pip

```bash
pip install -e ".[dev]"
```

## Development Setup

To ensure code quality and consistency, run the development setup script:

```bash
chmod +x dev_setup.sh
./dev_setup.sh
```

After setup, you can manually run pre-commit checks:
```bash
uv run pre-commit run --all-files
```

## Running the Project

### Scripts

Run scripts with Hydra:

```bash
uv run scripts/<script_name>
```
