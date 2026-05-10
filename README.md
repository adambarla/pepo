# PEPO

A project for preference alignment of language models using techniques like DPO (Direct Preference Optimization), RLHF (Reinforcement Learning from Human Feedback), and PEPO.

## Installation

The installation process is simplified for CUDA-enabled systems (Linux/Windows):

1. **Edit `pyproject.toml`** to set the correct CUDA version for your system:
    Open `pyproject.toml`
    Update the `url` in the `[tool.uv.index]` section to match your CUDA version:
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

3. **The `alpaca_eval` library is included in this package**.
    If you cloned this repo, run:
    ```bash
    git submodule update --init --recursive
    ```

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

Then edit `.env` and fill in all the values:

```bash
# HuggingFace Token (needs WRITE permissions for pushing models)
# Get from: https://huggingface.co/settings/tokens
HF_TOKEN=your_huggingface_token_here

# Weights & Biases Configuration (for experiment tracking)
WANDB_API_KEY=your_wandb_token_here
WANDB_ENTITY=your_wandb_entity

# HuggingFace Hub Base Directory (custom cache/storage location)
HF_HUB_BASE_DIR=your_hf_hub_base_dir
```

## Running the Project

### Basic Usage

Run scripts using `uv run`:

```bash
uv run scripts/eval.py
```

Run MT-Bench with PEPO answer generation and managed vLLM judging:

```bash
uv run scripts/eval.py evaluator=mtbench
```

The managed judge uses `evaluator.judge.vllm_executable`, which defaults to `.venv-vllm/bin/vllm` in the repo root. This keeps vLLM in a Python 3.12 environment while PEPO stays on Python 3.13:

```bash
uv venv .venv-vllm --python 3.12 --seed --managed-python --clear
uv pip install --python .venv-vllm/bin/python "vllm==0.10.1" --torch-backend=cu128
uv pip install --python .venv-vllm/bin/python "transformers>=4.55.0,<5"
```

Override `evaluator.judge.vllm_executable=/path/to/vllm` if vLLM is provided by a module or another environment.

For a GPU smoke test, first generate only PEPO answers, then reuse those answers while the evaluator starts and stops the vLLM judge:

```bash
uv run scripts/eval.py evaluator=mtbench ns=1 stop_after_generation=true
uv run scripts/eval.py evaluator=mtbench ns=1 overwrite=false
```

The MT-Bench evaluator unloads the PEPO model before launching the managed vLLM judge server. Judge tensor parallelism defaults to all GPUs visible to the device manager, so `CUDA_VISIBLE_DEVICES=1,2` starts vLLM with `--tensor-parallel-size 2`. If the cluster judge differs from the default `meta-llama/Meta-Llama-3-70B-Instruct`, override it with `evaluator.judge.model_name=...` and, if needed, `evaluator.judge.tensor_parallel_size=...`.

Judging is sequential and shows a temporary `Judging MT-Bench` progress bar over the vLLM HTTP calls. WandB eval runs include the evaluator in the run name and job type, plus `model:<name>` and `evaluator:<name>` tags.

Or run the training script:

```bash
uv run scripts/train.py
```

### SLURM Scripts

For running on SLURM clusters, use the scripts in `scripts/slurm/`:

- **`get_interactive.sh`**: Allocates an interactive node with 4 GPUs for 12 hours. Use this to get a shell on a compute node for development and testing.
- **`*.slurm`**: Batch job scripts for training and evaluation (e.g., `train.slurm`, `eval.slurm`).
- **`connect_to_node.sh`**: Connects to an existing interactive job. Shows a menu of all active jobs with details (job name, node, start time, time remaining) to help you choose which node to connect to. (can connect to batch jobs too)


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

This project uses pre-commit hooks to ensure code quality and consistency.

```bash
# Install development dependencies
uv sync --group dev

# Install pre-commit hooks
uv run pre-commit install
```

**Note for macOS/non-CUDA users**:
While you cannot run the training scripts locally without a CUDA device, you can still contribute to the codebase. The pre-commit hooks are configured to use `uvx`, so they will run in isolated environments without requiring you to install the project's heavy CUDA dependencies.
