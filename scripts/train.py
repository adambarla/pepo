# script to train the pepo model, using a hydra config file, log the training process to wb and save the model to huggingface

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from pepo import PEPOModel
from pepo.utils import DeviceManager, Logger, set_seed


@hydra.main(
    config_path="../configs/train", config_name="pepo_base.yaml", version_base="1.1"
)
def main(cfg: DictConfig):
    hydra_cfg = HydraConfig.get()
    original_work_dir = Path(hydra_cfg.runtime.cwd)

    logger = Logger(
        name="train",
        log_dir=str(original_work_dir / "logs"),
    )

    logger.info("PEPO Training - Starting")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    set_seed(cfg.seed)
    logger.info(f"Random seed set to: {cfg.seed}")

    device_manager = DeviceManager(
        device_config=cfg.device,
        num_models=cfg.ensemble.num_networks,
        logger=logger,
    )

    model = PEPOModel(
        pepo_config=cfg.pepo,
        full_config=cfg,
        logger=logger,
        device_manager=device_manager,
    )

    model.train()


if __name__ == "__main__":
    main()
