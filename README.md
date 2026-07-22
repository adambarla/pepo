# PEPO

A project for preference alignment of language models using techniques like DPO (Direct Preference Optimization), RLHF (Reinforcement Learning from Human Feedback), and PEPO.

---

## Installation

### 1. GPU Installation
1. **Configure CUDA version**: Open `pyproject.toml` and update the PyTorch index URL to match your CUDA version (e.g., `cu121`, `cu126`):
   ```toml
   [[tool.uv.index]]
   name = "pytorch"
   url = "https://download.pytorch.org/whl/cu126"
   ```
2. **Install all dependencies**:
   ```bash
   uv sync
   ```
3. **Initialize submodules** (for `alpaca_eval` integration):
   ```bash
   git submodule update --init --recursive
   ```

### 2. GPU-less Installation (for CPU/Analysis)
If setting up the project on a device without GPU/CUDA support (e.g., for analysis and plotting), you can sync **without** the heavy GPU dependencies:
```bash
uv sync --no-group gpu
```

### Adding New Dependencies
To add new packages to the project:
```bash
uv add package-name
```

---

## Environment Variables

Create a `.env` file in the project root:
```bash
cp .env.example .env
```

Open `.env` and fill in your keys:
* `HF_TOKEN`: HuggingFace token (requires **write** permissions to push models).
* `WANDB_API_KEY` & `WANDB_ENTITY`: Credentials for experiment tracking with Weights & Biases.
* `HF_HUB_BASE_DIR`: Custom cache/storage directory for HF Hub downloads.

---

## Running the Project

### Basic Usage
* **Training**:
  ```bash
  uv run scripts/train.py
  ```
* **Evaluation**:
  ```bash
  uv run scripts/eval.py
  ```

### MT-Bench Evaluation (with managed vLLM Judge)
By default, the evaluator launches a managed vLLM judge server. To keep dependencies separated, vLLM runs in a Python 3.12 virtual environment (`.venv-vllm`) while the core project runs on Python 3.13.

1. **Setup the vLLM Environment**:
   ```bash
   uv venv .venv-vllm --python 3.12 --seed --managed-python --clear
   uv pip install --python .venv-vllm/bin/python "vllm>=0.10.1" --torch-backend=cu128
   uv pip install --python .venv-vllm/bin/python "transformers>=4.55.0,<5" "fastapi<0.137.0"
   ```
2. **Run MT-Bench**:
   ```bash
   uv run scripts/eval.py evaluator=mtbench
   ```
   * *See [docs/mt_bench_compatibility.md](docs/mt_bench_compatibility.md) for reviewer-facing details on the local judge.*

3. **Useful Options & Customization**:
   * **GPU Smoke Test** (separate answer generation from judging):
     ```bash
     uv run scripts/eval.py evaluator=mtbench ns=1 stop_after_generation=true
     uv run scripts/eval.py evaluator=mtbench ns=1 overwrite=false
     ```
   * **vLLM Executable**: Override path with `evaluator.judge.vllm_executable=/path/to/vllm`.
   * **Tensor Parallelism**: Matches visible GPUs (e.g. `CUDA_VISIBLE_DEVICES=1,2` sets `--tensor-parallel-size 2`).
   * **Judge Model / Port**: Override defaults:
     ```bash
     uv run scripts/eval.py evaluator=mtbench evaluator.judge.model_name=meta-llama/Meta-Llama-3-70B-Instruct evaluator.judge.port=8000
     ```

### SLURM Clusters (`scripts/slurm/`)
For cluster runs, use the helper scripts in `scripts/slurm/`:
* `get_interactive.sh`: Allocates an interactive node with 4 GPUs for 12 hours.
* `connect_to_node.sh`: Displays active interactive/batch jobs and connects to your choice.
* `*.slurm`: Batch job submission templates (e.g., `train.slurm`, `eval.slurm`).

---

## Configuration Management (Hydra)

This project uses [Hydra](https://hydra.cc/) to manage configs in `configs/`.

* **Dot Notation Overrides**:
  ```bash
  python scripts/train.py hub.push=false L=1 log_level=debug
  ```
  *(e.g., disables pushing to HF Hub, sets ensemble networks to 1, sets logging to debug).*
* **Composition Structure (`configs/train.yaml`)**:
  Loads model configuration from `configs/model/smollm.yaml` and dataset from `configs/dataset/ultrafeedback.yaml`, then applies local overrides.

---

## Development

Keep the codebase clean with pre-commit hooks:
```bash
uv run pre-commit install
```
