import os
from pathlib import Path

import hydra
import pandas as pd
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pepo.utils import constants, set_seed, setup_logging

OmegaConf.register_new_resolver(
    "pepo.constants",
    lambda name: getattr(constants, name),
)

load_dotenv()


@hydra.main(config_path="../configs", config_name="eval.yaml", version_base="1.1")
def main(cfg: DictConfig) -> None:
    hydra_cfg = HydraConfig.get()
    original_work_dir = Path(hydra_cfg.runtime.cwd)

    log_level_str = cfg.get("log_level", "INFO").upper()
    logger = setup_logging(level=log_level_str)

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    logger.info("PEPO Evaluation - Starting")
    logger.debug(f"Configuration:\n{OmegaConf.to_yaml(resolved_cfg)}")

    set_seed(cfg.seed)
    logger.info(f"Random seed set to: {cfg.seed}")

    if not os.getenv("HF_TOKEN"):
        logger.warning(
            "HF_TOKEN environment variable not set. "
            "Model loading may fail if models are private."
        )

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
    else:
        raise ValueError(
            "Generator not configured in model config. "
            "Please add generator section to model config."
        )

    output_dir = cfg.get("output_dir", "outputs")
    if output_dir:
        output_dir_path = original_work_dir / output_dir
        cfg.evaluator.output_dir = str(output_dir_path)
        logger.info(f"Output directory set to: {output_dir_path}")

    evaluator = instantiate(cfg.evaluator, model=model)

    responses_exist = evaluator.responses_exist()
    force_generate = cfg.get("force_generate", False)

    if not responses_exist or force_generate:
        if not responses_exist:
            logger.info("Responses file not found - Generating responses")
        else:
            logger.info("force_generate=True - Regenerating responses")
        evaluator.generate_responses()
        model.unload_models()
    else:
        logger.info("Responses file found - Skipping generation")

    logger.info("Running evaluation")
    evaluator.evaluate()

    logger.info("Consolidating leaderboard files")
    from pepo.evaluator.alpaca import AlpacaEvalEvaluator

    consolidated_path = AlpacaEvalEvaluator.consolidate_leaderboards(
        output_dir=output_dir_path
    )
    logger.info(f"Consolidated leaderboard saved to: {consolidated_path}")

    # Display the full leaderboard
    if consolidated_path.exists():
        df_leaderboard = pd.read_csv(consolidated_path, index_col=0)

        print("\n" + "=" * 80)
        print("CONSOLIDATED LEADERBOARD")
        print("=" * 80)

        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.width",
            None,
            "display.max_colwidth",
            None,
            "display.precision",
            4,
        ):
            print(df_leaderboard.to_string())

        print("=" * 80 + "\n")

        logger.info("Leaderboard displayed above")


if __name__ == "__main__":
    main()
