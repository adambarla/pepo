# PEPO - Preference Optimization Project

A project for preference alignment of language models using techniques like DPO (Direct Preference Optimization), RLHF (Reinforcement Learning from Human Feedback), and PEPO.

## Installation

**Important**: Before proceeding with the installation steps below, make sure to run the development setup script first:

```bash
chmod +x dev_setup.sh
./dev_setup.sh
```

### CSCS Cluster (Clariden)

Clariden nodes have `aarch64` architecture, which makes it difficult to install PyTorch with CUDA support. Follow the instructions below to install the dependencies.

#### Prerequisites

Connect to Clariden: SSH to `clariden.cscs.ch` (may require proxy via `ela.cscs.ch`)
Request an interactive session with GPUs:
```bash
./scripts/slurm/get_interactive.sh
```
This script requests 4 GPUs for 1 hour. Adjust the script if you need different resources.

#### Installation Steps

Start the PyTorch user environment (uenv) and create a virtual environment with system site packages.
```bash
uenv image pull pytorch/v2.6.0:v1
uenv start pytorch/v2.6.0:v1 --view=default
rm -rf .venv # remove existing virtual environment if present
uv venv -p $(which python) --system-site-packages .venv
```
Using `--system-site-packages` allows access to PyTorch from the uenv environment.
This provides PyTorch 2.6.0 with CUDA support (Python 3.13) pre-installed.

Activate the virtual environment and install the project dependencies:
```bash
source .venv/bin/activate
uv pip install -e .
```
Verify installation:
```bash
python -c "import torch; print(f'Torch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```
You should see `Torch: 2.6.0` and `CUDA: True`.

#### Notes

- The uenv environment must be active before creating the virtual environment to ensure the correct Python interpreter is used.
- The uenv provides PyTorch with CUDA 12.6 support for the GH200 GPUs on Clariden.

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

### Scripts

Run scripts with Hydra:

```bash
source .venv/bin/activate
python scripts/<script_name>
```
