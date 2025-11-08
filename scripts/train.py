# script to train the pepo model, using a hydra config file, log the training process to wb and save the model to huggingface

import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pepo.utils import set_seed


@hydra.main(config_path="../configs", config_name="train.yaml", version_base="1.1")
def main(cfg: DictConfig):
    hydra_cfg = HydraConfig.get()
    original_work_dir = Path(hydra_cfg.runtime.cwd)

    log_level_str = cfg.get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # instantiate from hydra creates a class instance specified in _target_ with the given arguments
    logger = instantiate(
        cfg.logger,
        log_dir=str(original_work_dir / "logs"),
        level=log_level,
    )

    logger.info("PEPO Training - Starting")
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(resolved_cfg)}")

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
        logger=logger,
    )

    model.train(data_manager)


if __name__ == "__main__":
    main()
