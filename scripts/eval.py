import logging
import os
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pepo.utils import constants, set_seed

OmegaConf.register_new_resolver(
    "pepo.constants",
    lambda name: getattr(constants, name),
)


@hydra.main(config_path="../configs", config_name="eval.yaml", version_base="1.1")
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
    logger.info("PEPO Evaluation - Starting")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(resolved_cfg)}")

    set_seed(cfg.seed)
    logger.info(f"Random seed set to: {cfg.seed}")

    if not os.getenv("HF_TOKEN"):
        logger.warning(
            "HF_TOKEN environment variable not set. Model loading may fail if models are private."
        )

    # Instantiate managers
    device_manager = instantiate(cfg.device, logger=logger)
    hub_manager = instantiate(cfg.hub, logger=logger)

    # Instantiate model (same as chat.py)
    model = instantiate(
        cfg.model,
        logger=logger,
        device_manager=device_manager,
        hub_manager=hub_manager,
    )

    # Resolve output_dir relative to project root
    output_dir = cfg.get("output_dir", "outputs")
    if output_dir:
        output_dir_path = original_work_dir / output_dir
        cfg.evaluator.output_dir = str(output_dir_path)
        logger.info(f"Output directory set to: {output_dir_path}")

    # Instantiate evaluator (generator is automatically instantiated via config)
    evaluator = instantiate(cfg.evaluator, model=model, logger=logger)
    logger.info("Evaluator and generator instantiated from config")

    # Check if responses exist
    responses_exist = evaluator.responses_exist()
    force_generate = cfg.get("force_generate", False)

    # Generate responses if they don't exist or if force_generate is True
    if not responses_exist or force_generate:
        if not responses_exist:
            logger.info("Responses file not found - Generating responses")
        else:
            logger.info("force_generate=True - Regenerating responses")
        evaluator.generate_responses()
    else:
        logger.info("Responses file found - Skipping generation")

    # Evaluate responses
    logger.info("Running evaluation (implementation pending)")
    evaluator.evaluate()


if __name__ == "__main__":
    main()
