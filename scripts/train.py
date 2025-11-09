# script to train the pepo model, using a hydra config file, log the training process to wb and save the model to huggingface

import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from transformers import get_scheduler

from pepo.utils import WandbHandler, set_seed


@hydra.main(config_path="../configs", config_name="train.yaml", version_base="1.1")
def main(cfg: DictConfig):
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
            cfg, resolve=True, structured_config_mode=False
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

    num_networks = model.num_networks
    optimizers = []
    schedulers = []

    for model_idx in range(num_networks):
        model_params = model.models[model_idx].parameters()

        optimizer = instantiate(
            cfg.optimizer,
            params=model_params,
        )
        optimizers.append(optimizer)

        train_loader = data_manager.get_dataloader(
            model_idx=model_idx,
            partition="train",
            batch_size=cfg.training.batch_size,
        )
        num_training_steps = (
            len(train_loader) // cfg.training.gradient_accumulation_steps
        ) * cfg.training.epochs

        scheduler = get_scheduler(
            name=cfg.scheduler.name,
            optimizer=optimizer,
            num_warmup_steps=cfg.scheduler.num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        schedulers.append(scheduler)

        logger.info(
            f"Created optimizer and scheduler for model {model_idx}. "
            f"Training steps: {num_training_steps}"
        )

    wandb_handlers = None
    if wandb_config.enabled:
        base_model_name = model.model_id.rsplit("/", 1)[-1]
        wandb_handlers = []
        for model_idx in range(num_networks):
            run_name = model._get_model_name(model_idx)
            handler = WandbHandler(
                enabled=True,
                project=wandb_config.project,
                name=run_name,
                tags=list(wandb_config.tags) + [base_model_name],
                notes=wandb_config.notes,
                entity=wandb_config.entity,
                mode=wandb_config.mode,
                cfg=resolved_cfg_plain,
                _lazy_init=True,
            )
            wandb_handlers.append(handler)

    model.train(
        data_manager=data_manager,
        optimizers=optimizers,
        schedulers=schedulers,
        batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        epochs=cfg.training.epochs,
        wandb_handlers=wandb_handlers,
    )

    if wandb_handlers is not None:
        for handler in wandb_handlers:
            if handler.enabled:
                handler.finish()
        logger.info("Weights & Biases logging finished")


if __name__ == "__main__":
    main()
