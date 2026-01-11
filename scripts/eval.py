import os
import warnings
from typing import Any, Optional, cast

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pepo.evaluator.base import BaseEvaluator
from pepo.model import BaseModel
from pepo.utils import (
    WandbManager,
    WandbRun,
    constants,
    init_device_manager,
    init_hub_manager,
    setup_logging,
    strip_hydra_targets,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress warnings
warnings.filterwarnings(
    "ignore", message=".*pkg_resources is deprecated.*", category=UserWarning
)

# Register resolvers if not already registered
# (benchmark.py does this too, but for cleanliness)
try:
    OmegaConf.register_new_resolver(
        "pepo.constants",
        lambda name: getattr(constants, name),
    )
except ValueError:
    pass  # Already registered


@hydra.main(config_path="../configs", config_name="eval", version_base="1.1")
def main(cfg: DictConfig) -> None:
    # Setup logging
    debug = cfg.get("debug", False)
    log_level_str = cfg.get("log_level", "INFO").upper()
    if debug:
        log_level_str = "DEBUG"
    logger = setup_logging(level=log_level_str)

    logger.debug(f"Config: \n{OmegaConf.to_yaml(cfg, resolve=True)}")
    logger.info("PEPO Evaluation Script - Starting")

    # Initialize global managers
    init_device_manager(
        gpu_ids=cfg.get("gpu_ids"),
        dtype=cfg.get("dtype", "bfloat16"),
    )
    init_hub_manager(
        base_dir=cfg.get("hub_base_dir", "PessimisticDPO"),
        push=cfg.get("push_to_hub", False),  # Eval typically doesn't push
        load_trainable=False,  # Eval loads frozen models
    )

    model: BaseModel = instantiate(cfg.model)
    epoch: Optional[int] = cfg.get("e", None)

    if model.generator is None:
        raise ValueError("Generator not found in model")

    wandb_config = cfg.get("wandb", OmegaConf.create({"enabled": False}))
    wandb_manager: Optional[WandbManager] = None
    wandb_run: Optional[WandbRun] = None
    if wandb_config.enabled:
        resolved_cfg_plain = OmegaConf.to_container(
            cfg,
            resolve=True,
            structured_config_mode=False,  # type: ignore[arg-type]
        )
        # Strip Hydra _target_ keys to prevent recursive instantiation
        resolved_cfg_plain = strip_hydra_targets(resolved_cfg_plain)
        wandb_manager = instantiate(
            wandb_config,
            cfg=cast("dict[str, Any] | None", resolved_cfg_plain),
        )
    if wandb_manager is not None:
        wandb_run = wandb_manager.get_evaluation_handler(
            model=model, generator=model.generator, epoch=epoch
        )
    if wandb_run is not None:
        wandb_run.init_run()
        logger.info(f"WandB evaluation run initialized: {wandb_run.run_id}")

    evaluator: BaseEvaluator = instantiate(cfg.evaluator, wandb_run=wandb_run)

    model_ref: Optional[BaseModel] = None
    epoch_ref: Optional[int] = None
    if cfg.get("ref_model", None) is not None and "_target_" in cfg.ref_model:
        model_ref = instantiate(cfg.ref_model)
        epoch_ref = cfg.get("ref_e", None)

    evaluator.evaluate(
        model=model,
        epoch=epoch,
        ref_model=model_ref,
        ref_epoch=epoch_ref,
        overwrite=cfg.get("overwrite", False),
    )

    if wandb_run is not None:
        wandb_run.finish()

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
