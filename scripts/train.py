import os

# Prevent CUDA memory fragmentation (must be set before any CUDA operations)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import multiprocessing

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    raise RuntimeError(
        "Failed to set multiprocessing start method to 'spawn'. "
        "It's required for DataLoader workers to work correctly "
        "with CUDA and threading."
    )

import warnings
from typing import Any, cast

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, SCMode

from pepo.data import DataManager
from pepo.model import BaseModel
from pepo.utils import (
    WandbManager,
    constants,
    init_device_manager,
    init_hub_manager,
    set_seed,
    setup_logging,
)

warnings.filterwarnings(
    "ignore", message=".*pkg_resources is deprecated.*", category=UserWarning
)

try:
    OmegaConf.register_new_resolver(
        "pepo.constants",
        lambda name: getattr(constants, name),
    )
except ValueError:
    pass  # Already registered


@hydra.main(config_path="../configs", config_name="train", version_base="1.1")  # type: ignore
def main(cfg: DictConfig) -> None:
    # Setup logging
    debug = cfg.get("debug", False)
    log_level_str = cfg.get("log_level", "INFO").upper()
    if debug:
        log_level_str = "DEBUG"
    logger = setup_logging(level=log_level_str)

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    logger.info("PEPO Training Script - Starting")
    logger.debug(f"Configuration:\n{OmegaConf.to_yaml(resolved_cfg)}")

    set_seed(cfg.seed)

    # Initialize global managers
    device_manager = init_device_manager(
        gpu_ids=cfg.get("gpu_ids"),
        dtype=cfg.get("dtype", "bfloat16"),
    )
    init_hub_manager(
        base_dir=cfg.get("hub_base_dir", "PessimisticDPO"),
        push=cfg.get("push_to_hub", cfg.get("sync", True)),
        load_trainable=True,
    )

    model: BaseModel = instantiate(cfg.model)

    if model._trainer is None:
        raise ValueError("Trainer not configured in model config.")

    # Setup Data Manager
    data_manager: DataManager = instantiate(
        cfg.dataset,
        tokenizer=model.get_tokenizer(),
        inference_batch_size=cfg.model.trainer.eval_batch_size,
        device_manager=device_manager,
    )

    # Setup WandB
    wandb_config = cfg.get("wandb", OmegaConf.create({"enabled": False}))
    wandb_manager = None
    if wandb_config.enabled:
        resolved_cfg_plain = OmegaConf.to_container(
            cfg,
            resolve=True,
            structured_config_mode=SCMode.DICT,
        )
        # Merge tags from model config if present
        tags = list(wandb_config.tags)
        model_cfg = cfg.get("model")
        if model_cfg is not None and "wandb" in model_cfg and "tags" in model_cfg.wandb:
            for tag in model_cfg.wandb.tags:
                if tag not in tags:
                    tags.append(tag)

        wandb_manager = WandbManager(
            enabled=True,
            project=wandb_config.project,
            tags=tags,
            notes=wandb_config.notes,
            entity=wandb_config.entity,
            mode=wandb_config.mode,
            cfg=cast("dict[str, Any] | None", resolved_cfg_plain),
        )

    continue_training = cfg.get("continue", False)
    max_epochs = cfg.get("e", 3)

    logger.info(f"Starting training for {max_epochs} epochs...")
    model.train(
        data_manager=data_manager,
        max_epochs=max_epochs,
        wandb_manager=wandb_manager,
        continue_training=continue_training,
    )

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
