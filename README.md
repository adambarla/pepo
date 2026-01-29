# PEPO - Preference Optimization Project

A project for preference alignment of language models using techniques like DPO (Direct Preference Optimization), RLHF (Reinforcement Learning from Human Feedback), and PEPO.

## Installation

The installation process is simplified and works across all platforms:

1. **Edit `pyproject.toml`** to set the correct CUDA version for your system:
   - Open `pyproject.toml`
   - Update the `url` in the `[tool.uv.index]` section to match your CUDA version:
     ```toml
     [[tool.uv.index]]
     name = "pytorch"
     url = "https://download.pytorch.org/whl/cu126"  # Change cu126 to your CUDA version (e.g., cu118, cu121)
     ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

   This will create a virtual environment and install all required packages, including PyTorch with the specified CUDA version.

### Adding Dependencies

To add new dependencies to the project, use `uv add`:

```bash
uv add package-name
```

This will automatically update `pyproject.toml` and `uv.lock` with the new dependency.

## Environment Variables

The project uses environment variables for configuration. Create a `.env` file in the project root (`.env` is already in `.gitignore`):

```bash
cp .env.example .env
```

Then edit `.env` and set your HuggingFace token:

```
HF_TOKEN=your_huggingface_token_here
```

Get your token from: https://huggingface.co/settings/tokens

**Note**: Make sure your token has WRITE permissions if you plan to push models to the HuggingFace Hub.

## Running the Project

### Basic Usage

Run scripts using `uv run`:

```bash
uv run scripts/eval.py
```

Or run the training script:

```bash
uv run scripts/train.py
```

### SLURM Scripts

For running on SLURM clusters, use the scripts in `scripts/slurm/`:

- **`get_interactive.sh`**: Allocates an interactive node with 4 GPUs for 12 hours. Use this to get a shell on a compute node for development and testing.
- **`*.slurm`**: Batch job scripts for training and evaluation (e.g., `train.slurm`, `eval.slurm`).
- **`connect_to_node.sh`**: Connects to an existing interactive job. Shows a menu of all active jobs with details (job name, node, start time, time remaining) to help you choose which node to connect to. (can connect to batch jobs too)

### GPU Job Queue (Shared Servers)

For shared GPU servers without SLURM, use the GPU job queue scheduler (`scripts/gpu_queue.py`). It automatically runs queued jobs when the required number of GPUs become available.

#### Quick Start

```bash
# Add jobs to the queue (default: 2 GPUs per job)
uv run scripts/gpu_queue.py add "uv run scripts/train.py model=deppo L=1 a=0.0"
uv run scripts/gpu_queue.py add --gpus 4 "uv run scripts/train.py model=deppo L=4 a=0.1"

# Start the daemon (runs in background, polls every 30s)
uv run scripts/gpu_queue.py start

# Check queue status and GPU availability
uv run scripts/gpu_queue.py status
```

#### Commands

| Command | Description |
|---------|-------------|
| `add --gpus N "cmd"` | Add a job requiring N GPUs (default: 2) |
| `start` | Start the background daemon |
| `stop` | Stop the daemon |
| `status` | Show queue, running jobs, and GPU state |
| `cancel <id>` | Remove a pending job |
| `logs <id>` | View job output logs |
| `clear` | Clear completed jobs |

#### How It Works

- Jobs run from the project root (`~/pepo`)
- Queue data stored in `~/.gpu_queue/jobs.json`
- Job logs saved to `~/.gpu_queue/logs/<job_id>.log`
- A GPU is considered "free" if using < 5GB memory
- Jobs execute in FIFO order when enough GPUs are available

### Configuration Management

This project uses [Hydra](https://hydra.cc/) for configuration management. Configuration files are stored in the `configs` directory, with `configs/train.yaml` as the default (specified in `scripts/train.py`).

#### Overriding Parameters

Override any configuration parameter from the command line using dot notation:

```bash
python scripts/train.py hub.push=false L=1 log_level=debug
```

This example:
- Disables pushing models to HuggingFace Hub (`hub.push=false`)
- Sets the number of ensemble networks to 1 (`L=1`)
- Sets the log level to debug (`log_level=debug`)

For more details, see the [Hydra documentation](https://hydra.cc/docs/intro/).

#### Configuration File Structure

The `configs/train.yaml` file uses Hydra's `defaults:` to compose configurations:

```yaml
defaults:
  - model: smollm
  - dataset: ultrafeedback
  - _self_
```

This loads:
- Model configuration from `configs/model/smollm.yaml` (directory name matches the field name)
- Dataset configuration from `configs/dataset/ultrafeedback.yaml`
- The `train.yaml` config itself (via `_self_`), which can override the defaults

For example, `train.yaml` overrides the number of ensemble networks using a variable reference:

```yaml
model:
  num_networks: ${L}
```
The values in the `train.yaml` config file can be overridden by the command line arguments.

## Development

### Pre-commit hooks

This project uses pre-commit hooks to ensure code quality and consistency. **All contributors must set up pre-commit hooks**:

```bash
# Install development dependencies
uv sync --group dev

# Install pre-commit hooks
uv run pre-commit install

# Optional: Install commit-msg hook for conventional commits
uv run pre-commit install --hook-type commit-msg
```

The hooks will automatically:
- **Clean Jupyter notebooks** by removing output cells
- **Format code** with Black and isort
- **Lint code** with Ruff
- **Type check** with mypy
- **Check for common issues** like trailing whitespace, large files, etc.

**Important**: Pre-commit hooks are enforced via GitHub Actions. Pull requests will fail if checks don't pass.
