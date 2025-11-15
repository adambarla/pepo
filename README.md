# PEPO - Preference Optimization Project

A project for preference alignment of language models using techniques like DPO (Direct Preference Optimization), RLHF (Reinforcement Learning from Human Feedback), and PEPO.

## Installation

**Important**: Before proceeding with the installation steps below, make sure to run the development setup script first:

```bash
chmod +x dev_setup.sh
./dev_setup.sh
```

### CSCS Cluster (Clariden)

Connect to `clariden.cscs.ch` (may require proxy via `ela.cscs.ch`).

**Get interactive session:**
```bash
./scripts/slurm/get_interactive.sh  # Requests 4 GPUs for 12 hours
./scripts/activate.sh                # Activates uenv and venv
```

**Additional shells on same node:**
```bash
./scripts/slurm/connect_to_node.sh  # Shows menu of active sessions
```

**First-time setup:**
```bash
./scripts/slurm/get_interactive.sh
./scripts/activate.sh
rm -rf .venv
uv venv -p $(which python) --system-site-packages .venv
source .venv/bin/activate
uv pip install -e .
python -c "import torch; print(f'Torch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

The uenv provides PyTorch 2.6.0 with CUDA 12.6 support for GH200 GPUs.

### Other Machines

Create a virtual environment with Python 3.13 and install dependencies:

```bash
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cuXXX # replace XXX with the correct CUDA version
uv pip install -e .
```

Verify installation:
```bash
python -c "import torch; print(f'Torch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

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

Run the training script with the default configuration:

```bash
source .venv/bin/activate
python scripts/train.py
```

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
