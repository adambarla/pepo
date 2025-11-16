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
