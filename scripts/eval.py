import warnings

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pepo.utils import constants, setup_logging

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


@hydra.main(config_path="../configs", config_name="eval", version_base="1.1")  # type: ignore
def main(cfg: DictConfig) -> None:
    # Setup logging
    debug = cfg.get("debug", False)
    log_level_str = cfg.get("log_level", "INFO").upper()
    if debug:
        log_level_str = "DEBUG"
    logger = setup_logging(level=log_level_str)

    logger.debug(f"Config: \n{OmegaConf.to_yaml(cfg, resolve=True)}")
    logger.info("PEPO Evaluation Script - Starting")

    device_manager = instantiate(cfg.device)
    hub_manager = instantiate(cfg.hub)
    evaluator = instantiate(cfg.evaluator)

    model = instantiate(
        cfg.model,
        device_manager=device_manager,
        hub_manager=hub_manager,
    )
    epoch = cfg.get("e", None)

    model_ref = None
    epoch_ref = None
    if cfg.get("ref_model", None) is not None:
        model_ref = instantiate(
            cfg.ref_model,
            device_manager=device_manager,
            hub_manager=hub_manager,
        )
        epoch_ref = cfg.get("ref_e", None)

    evaluator.evaluate(
        model=model, epoch=epoch, ref_model=model_ref, ref_epoch=epoch_ref
    )
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
