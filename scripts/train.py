# script to train the pepo model, using a hydra config file,
# log the training process to wb and save the model to huggingface
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pepo.trainer import Trainer
from pepo.utils import constants, set_seed

OmegaConf.register_new_resolver(
    "pepo.constants",
    lambda name: getattr(constants, name),
)


@hydra.main(config_path="../configs", config_name="train.yaml", version_base="1.1")
def main(cfg: DictConfig) -> None:
    hydra_cfg = HydraConfig.get()
    original_work_dir = Path(hydra_cfg.runtime.cwd)

    log_level_str = cfg.get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    logger = instantiate(
        cfg.logger,
        log_dir=str(original_work_dir / "logs"),
        level=log_level,
    )

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    logger.info("PEPO Training - Starting")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(resolved_cfg)}")

    wandb_config = cfg.wandb
    resolved_cfg_plain = None
    if wandb_config.enabled:
        resolved_cfg_plain = OmegaConf.to_container(
            cfg,
            resolve=True,
            structured_config_mode=False,  # type: ignore[arg-type]
        )
        logger.info("Weights & Biases logging enabled")

    set_seed(cfg.seed)
    logger.info(f"Random seed set to: {cfg.seed}")

    device_manager = instantiate(
        cfg.device,
        logger=logger,
    )

    hub_manager = instantiate(
        cfg.hub,
        logger=logger,
    )

    model = instantiate(
        cfg.model,
        logger=logger,
        device_manager=device_manager,
        hub_manager=hub_manager,
    )

    data_manager = instantiate(
        cfg.dataset,
        tokenizer=model.get_tokenizer(),
        logger=logger,
    )

    trainer = Trainer(
        model=model,
        data_manager=data_manager,
        optimizer_config=cfg.optimizer,
        scheduler_config=cfg.scheduler,
        wandb_config=wandb_config,
        batch_size=cfg.batch_size,
        eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.acc_steps,
        max_epochs=cfg.max_epochs,
        early_stopping_patience=cfg.early_stopping.patience,
        early_stopping_min_delta=cfg.early_stopping.min_delta,
        resolved_cfg_plain=resolved_cfg_plain,  # type: ignore[arg-type]
        logger=logger,
    )
    trainer.train()  # type: ignore[no-untyped-call]


if __name__ == "__main__":
    main()
