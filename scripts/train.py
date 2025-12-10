# script to train the pepo model, using a hydra config file,
# log the training process to wb and save the model to huggingface
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import multiprocessing

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    raise RuntimeError(
        "Failed to set multiprocessing start method to 'spawn'. "
        "It's required for DataLoader workers to work correctly "
        "with CUDA and threading."
    )


from typing import Any, cast

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pepo.utils import WandbManager, constants, set_seed, setup_logging

OmegaConf.register_new_resolver(
    "pepo.constants",
    lambda name: getattr(constants, name),
)


@hydra.main(config_path="../configs", config_name="train.yaml", version_base="1.1")
def main(cfg: DictConfig) -> None:
    log_level_str = cfg.get("log_level", "INFO").upper()
    logger = setup_logging(level=log_level_str)

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    logger.info("PEPO Training - Starting")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(resolved_cfg)}")

    wandb_config = cfg.wandb
    wandb_manager = None
    resolved_cfg_plain = None
    if wandb_config.enabled:
        resolved_cfg_plain = OmegaConf.to_container(
            cfg,
            resolve=True,
            structured_config_mode=False,  # type: ignore[arg-type]
        )
        wandb_manager = WandbManager(
            enabled=True,
            project=wandb_config.project,
            tags=wandb_config.tags,
            notes=wandb_config.notes,
            entity=wandb_config.entity,
            mode=wandb_config.mode,
            cfg=cast("dict[str, Any] | None", resolved_cfg_plain),
        )
        logger.info("Weights & Biases logging enabled")

    set_seed(cfg.seed)
    logger.info(f"Random seed set to: {cfg.seed}")

    device_manager = instantiate(cfg.device)

    hub_manager = instantiate(cfg.hub)

    model = instantiate(
        cfg.model,
        device_manager=device_manager,
        hub_manager=hub_manager,
    )

    generator = None
    if "generator" in cfg.model:
        generator = instantiate(cfg.model.generator)
        model.generator = generator

    data_manager = instantiate(
        cfg.dataset,
        tokenizer=model.get_tokenizer(),
        ref_model_id=model.model_id,
        inference_batch_size=cfg.model.trainer.eval_batch_size,
        device_manager=device_manager,
    )

    if model.trainer is None:
        raise ValueError("Trainer not configured in model config.")

    if cfg.get("continue", False):
        model.trainer.continue_training = True

    max_epochs = cfg.max_epochs
    model.train(
        data_manager=data_manager,
        max_epochs=max_epochs,
        wandb_manager=wandb_manager,
    )


if __name__ == "__main__":
    main()
