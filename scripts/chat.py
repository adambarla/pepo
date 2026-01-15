#!/usr/bin/env python3
"""
Chat interactively with a trained model.

Usage:
    uv run scripts/chat.py model=deppo backbone=llama8b gpu_ids="[0]"
"""

import logging
import os
import sys

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from pepo.model import BaseModel
from pepo.utils import (
    constants,  # added
    get_device_manager,
    init_device_manager,
    init_hub_manager,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)

try:
    OmegaConf.register_new_resolver(
        "pepo.constants",
        lambda name: getattr(constants, name),
    )
except ValueError:
    pass  # Already registered


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    # 1. Setup Logging (Standardized)
    debug = cfg.get("debug", False)
    log_level_str = cfg.get("log_level", "INFO").upper()
    if debug:
        log_level_str = "DEBUG"
    setup_logging(level=log_level_str)

    # 2. Setup Device Manager
    set_seed(42)

    # 2. Initialize Managers
    init_device_manager(
        gpu_ids=cfg.gpu_ids,
        dtype=cfg.dtype,
    )
    init_hub_manager(base_dir=cfg.get("hub_base_dir", "PessimisticDPO"))
    device_manager = get_device_manager()

    # 3. Initialize Model
    logger.info(f"Initializing model: {cfg.model._target_}")
    model: BaseModel = hydra.utils.instantiate(cfg.model, backbone=cfg.backbone)

    # 4. Load Model (Load latest checkpoint if available, or just base model)
    # We want to load the trained model if possible.
    # Check for "e" argument or find latest.
    epoch = cfg.get("e", None)
    if epoch is None:
        logger.info("No epoch specified (e=...). Attempting to find latest epoch...")
        epoch = model.find_latest_epoch(max_epoch=10)

    if epoch is not None:
        logger.info(f"Loading model from epoch {epoch}...")
        model.load(epoch=epoch)
    else:
        logger.warning(
            "No checkpoint found. Loading base model (untrained/initialized)."
        )
        model.load(init_new=True)

    # 5. Move to GPU
    # Our models lazy load onto CPU. We need to move them to GPU for inference.
    # Generator logic assumes model is on the device returned by device_manager.

    logger.info("Moving models to GPU...")
    # Ensure models are loaded
    if model._models is None:
        model.load()

    # Move each submodel to its assigned device
    if getattr(model, "shared_backbone", False):
        logger.info("Shared backbone enabled: Moving to first allocated device")
        target_device = device_manager.get_device_for_model(0)
        model.models[0].to(target_device)
    else:
        for i in range(model.num_models):
            target_device = device_manager.get_device_for_model(i)
            logger.info(f"Moving submodel {i} to {target_device}")
            model.models[i].to(target_device)

    logger.info("\n" + "=" * 50)
    logger.info("Chat Interface Ready! (Ctrl+C to exit)")
    logger.info("=" * 50 + "\n")

    # 6. Chat Loop
    history = []
    while True:
        try:
            user_input = input("\033[92mUser: \033[0m")
            if user_input.lower() in ("/exit", "/quit"):
                break
            if user_input.lower() == "/reset":
                history = []
                print("\033[93mConversation reset.\033[0m")
                continue

            history.append({"role": "user", "content": user_input})

            print("\033[94mAssistant: \033[0m", end="", flush=True)

            def print_token(token: str):
                print(token, end="", flush=True)

            os.environ["TQDM_DISABLE"] = "1"
            logging.getLogger("pepo.generator").setLevel(logging.WARNING)

            with torch.no_grad():
                results = model.generate_responses(
                    prompts=[history],
                    apply_chat_template=True,
                    token_callback=print_token,
                )
                assistant_response = results[0]["output"]
                history.append({"role": "assistant", "content": assistant_response})

            print("\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception:
            logger.exception("Error during generation")


if __name__ == "__main__":
    main()
