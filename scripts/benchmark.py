import os

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
from pathlib import Path  # noqa: E402
from typing import Any, cast  # noqa: E402

import hydra  # noqa: E402
import pandas as pd  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from pepo.utils import WandbManager, constants, set_seed, setup_logging  # noqa: E402

warnings.filterwarnings(
    "ignore", message=".*pkg_resources is deprecated.*", category=UserWarning
)

OmegaConf.register_new_resolver(
    "pepo.constants",
    lambda name: getattr(constants, name),
)


@hydra.main(config_path="../configs", config_name="benchmark.yaml", version_base="1.1")
def main(cfg: DictConfig) -> None:
    hydra_cfg = HydraConfig.get()
    original_work_dir = Path(hydra_cfg.runtime.cwd)

    # Setup logging
    debug = cfg.get("debug", False)
    log_level_str = cfg.get("log_level", "INFO").upper()
    if debug:
        log_level_str = "DEBUG"
    logger = setup_logging(level=log_level_str)

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    logger.info("PEPO Benchmark - Starting")
    logger.debug(f"Configuration:\n{OmegaConf.to_yaml(resolved_cfg)}")

    set_seed(cfg.seed)

    # Instantiate managers
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

    overwrite = cfg.get("overwrite", False)
    continue_training = cfg.get("continue", False) if not overwrite else False
    evaluation_epoch = cfg.get("e", None)

    if evaluation_epoch is None:
        raise ValueError(
            "Benchmarking only supported when training to a specific epoch currently."
            "Set `e` in the benchmark config to specify the epoch to train to."
        )

    can_load = model.can_load_from_epoch(evaluation_epoch)

    wandb_config = cfg.get("wandb", OmegaConf.create({"enabled": False}))

    wandb_manager = None
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

    if not can_load or overwrite:
        if overwrite:
            logger.info("Overwrite enabled. Training model...")
        else:
            logger.info("Model not found. Training model...")

        if model.trainer is None:
            raise ValueError("Trainer not configured in model config.")

        data_manager = instantiate(
            cfg.dataset,
            tokenizer=model.get_tokenizer(),
            ref_model_id=model.model_id,
            inference_batch_size=cfg.model.trainer.eval_batch_size,
            device_manager=device_manager,
        )

        model.train(
            data_manager=data_manager,
            max_epochs=evaluation_epoch,
            wandb_manager=wandb_manager,
            continue_training=continue_training,
        )

    # 3. Load model
    # If we trained, models are in memory/on GPU.
    # If we didn't train, we need to load them from the evaluation_epoch.
    if evaluation_epoch is not None:
        try:
            model._check_models_loaded(expected_epoch=evaluation_epoch)
            logger.info(f"Models already loaded from epoch {evaluation_epoch}.")
        except RuntimeError:
            logger.info(f"Loading models from epoch {evaluation_epoch}...")
            model.load_models(init_new=False, epoch=evaluation_epoch)

    # 4. Setup wandb for benchmark if enabled
    wandb_run = None
    if wandb_manager is not None:
        wandb_run = wandb_manager.get_benchmark_handler(model=model)
        if wandb_run is not None:
            wandb_run.init_bench_run()

    output_dir = cfg.evaluator.get("output_dir", "outputs")
    if output_dir:
        output_dir_path = original_work_dir / output_dir
        cfg.evaluator.output_dir = str(output_dir_path)

    evaluator = instantiate(cfg.evaluator, model=model, wandb_run=wandb_run)

    # Debug: Log the expected filename and path
    logger.info(f"Looking for responses file: {evaluator.responses_file}")
    logger.info(f"Output directory: {evaluator.output_dir}")
    logger.info(f"Responses filename: {evaluator.responses_filename}")

    responses_exist = evaluator.responses_exist()

    if not responses_exist or overwrite:
        if overwrite:
            logger.info("Overwrite enabled. Generating responses...")
        else:
            logger.info("Responses not found. Generating responses...")
        evaluator.generate_responses()
    else:
        logger.info("Responses file exists. Skipping generation.")
    model.unload_models()  # Free up memory for judge if needed

    logger.info("Running evaluation to compute leaderboard...")
    evaluator.evaluate()

    if wandb_run is not None:
        wandb_run.finish()

    consolidated_path = output_dir_path / "leaderboard.csv"
    if consolidated_path.exists():
        df_leaderboard = pd.read_csv(consolidated_path, index_col=0)
        logger.info(f"Consolidated Leaderboard:\n{df_leaderboard}")


if __name__ == "__main__":
    main()
