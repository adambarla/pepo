#!/usr/bin/env python3
"""Find optimal batch sizes for training, evaluation, and generation."""

import gc
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

# Prevent CUDA memory fragmentation (must be set before any CUDA operations)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import multiprocessing

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import warnings

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizerBase

from pepo.utils import constants, set_seed, setup_logging

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

try:
    OmegaConf.register_new_resolver(
        "pepo.constants", lambda name: getattr(constants, name)
    )
except ValueError:
    pass

MAX_BATCH_SIZE = 4096


def cleanup_cuda() -> None:
    """Aggressive GPU memory cleanup."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()


def binary_search_batch_size(
    try_fn: Callable[[int], bool],
    start: int,
    logger: logging.Logger,
) -> Optional[int]:
    """Binary search for optimal batch size."""
    batch_size = start
    best = None

    while batch_size <= MAX_BATCH_SIZE:
        logger.info(f"Trying batch_size={batch_size}...")
        if try_fn(batch_size):
            best = batch_size
            logger.info(f"batch_size={batch_size} OK")
            batch_size *= 2
        else:
            logger.info(f"batch_size={batch_size} FAILED")
            if best:
                return best
            break

    if best:
        return best

    batch_size = start // 2
    while batch_size >= 1:
        logger.info(f"Trying batch_size={batch_size}...")
        if try_fn(batch_size):
            logger.info(f"batch_size={batch_size} OK")
            return batch_size
        logger.info(f"batch_size={batch_size} FAILED")
        batch_size //= 2

    return None


def test_train_batch_size(
    model: Any,
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizerBase,
    logger: logging.Logger,
) -> Tuple[Optional[int], Optional[str]]:
    """Test training batch size with real data (without eval)."""
    logger.info("Testing TRAINING batch size...")

    data_manager = instantiate(
        cfg.dataset,
        n_splits=model.num_models,
        tokenizer=tokenizer,
        inference_batch_size=cfg.model.trainer.eval_batch_size,
        device_manager=model.device_manager,
        shuffle_train=False,
    )

    def try_batch_size(batch_size: int) -> bool:
        cleanup_cuda()
        try:
            model.init_trainer()
            model.trainer.batch_size = batch_size
            model.trainer.eval_batch_size = batch_size
            model.trainer.skip_eval = True
            model.trainer.max_batches_per_epoch = 1

            original_push = model.hub_manager.should_push
            model.hub_manager.should_push = False
            try:
                model.train(
                    data_manager=data_manager,
                    max_epochs=1,
                    wandb_manager=None,
                    continue_training=False,
                )
            finally:
                model.hub_manager.should_push = original_push
                model.trainer.optimizers = []
                model.trainer.schedulers = []
                model.epochs_per_model = [0] * model.num_models
            return True
        except (RuntimeError, torch.cuda.OutOfMemoryError, InterruptedError):
            if model.trainer:
                model.trainer.optimizers = []
                model.trainer.schedulers = []
            model.epochs_per_model = [0] * model.num_models
            cleanup_cuda()
            return False

    result = binary_search_batch_size(
        try_batch_size, cfg.model.trainer.train_batch_size, logger
    )
    return (result, None) if result else (None, "OOM at batch_size=1")


def test_eval_batch_size(
    model: Any,
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizerBase,
    logger: logging.Logger,
) -> Tuple[Optional[int], Optional[str]]:
    """Test evaluation batch size with optimizer states loaded (1 batch only).

    Sets up training (to load optimizer states), then tests 1 eval batch.
    """
    logger.info("Testing EVAL batch size...")

    data_manager = instantiate(
        cfg.dataset,
        n_splits=model.num_models,
        tokenizer=tokenizer,
        inference_batch_size=cfg.model.trainer.eval_batch_size,
        device_manager=model.device_manager,
        shuffle_train=False,
    )

    def try_batch_size(batch_size: int) -> bool:
        cleanup_cuda()
        try:
            # Setup trainer to load optimizer states (matches real training)
            model.init_trainer()
            model.trainer.model = model  # _setup_training needs this
            model.trainer._setup_training(
                data_manager, max_epochs=1, wandb_manager=None
            )

            # Get device and model for first network
            device = torch.device(model.device_manager.get_device_for_model(0))
            submodel = model.models[0]
            submodel.eval()

            # Run 1 eval batch
            loader = data_manager.get_dataloader(
                model_idx=0, partition="eval", batch_size=batch_size
            )
            for batch in loader:
                with torch.no_grad():
                    model.loss_fn(batch, submodel, device)
                break  # Only 1 batch

            # Cleanup
            model.trainer.optimizers = []
            model.trainer.schedulers = []
            return True
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            if model.trainer:
                model.trainer.optimizers = []
                model.trainer.schedulers = []
            cleanup_cuda()
            return False

    result = binary_search_batch_size(
        try_batch_size, cfg.model.trainer.eval_batch_size, logger
    )
    return (result, None) if result else (None, "OOM at batch_size=1")


def test_gen_batch_size(
    model: Any,
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizerBase,
    logger: logging.Logger,
) -> Tuple[Optional[int], Optional[str]]:
    """Test generation batch size."""
    logger.info("Testing GENERATION batch size...")

    max_prompt_length = cfg.dataset.max_prompt_length
    max_new_tokens = cfg.model.generator.max_new_tokens
    target_len = max_prompt_length + max_new_tokens - 1

    generator = instantiate(
        cfg.model.generator, max_prompt_length=target_len, max_new_tokens=1
    )
    model.generator = generator

    def try_batch_size(batch_size: int) -> bool:
        cleanup_cuda()
        try:
            input_ids = torch.randint(
                0, len(tokenizer), (batch_size, target_len), dtype=torch.long
            )
            attention_mask = torch.ones_like(input_ids)
            generator.generate(
                model=model, input_ids=input_ids, attention_mask=attention_mask
            )
            return True
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            cleanup_cuda()
            return False

    result = binary_search_batch_size(
        try_batch_size, cfg.model.generator.batch_size, logger
    )
    return (result, None) if result else (None, "OOM at batch_size=1")


def update_model_config(
    path: Path, results: dict[str, Any], logger: logging.Logger
) -> None:
    """Update model config YAML with optimal batch sizes."""
    with open(path) as f:
        lines = f.readlines()

    mapping = {
        "train_batch_size:": ("training", "train_batch_size"),
        "eval_batch_size:": ("evaluation", "eval_batch_size"),
        "generator_batch_size:": ("generation", "generator_batch_size"),
    }

    new_lines = []
    updates = []
    today = datetime.now().strftime("%Y-%m-%d")

    for line in lines:
        stripped = line.strip()
        updated = False

        for prefix, (key, name) in mapping.items():
            if stripped.startswith(prefix) and key in results:
                new_val = results[key].get("optimal_batch_size")
                if not new_val:
                    break
                try:
                    old_val = int(stripped.split(":")[1].split("#")[0].strip())
                except (IndexError, ValueError):
                    break
                if old_val == new_val:
                    break
                indent = len(line) - len(line.lstrip())
                key = prefix.rstrip(":")
                new_lines.append(f"{' ' * indent}{key}: {new_val}  # Updated {today}\n")
                updates.append(f"{name}: {old_val} -> {new_val}")
                updated = True
                break

        if not updated:
            new_lines.append(line)

    if updates:
        with open(path, "w") as f:
            f.writelines(new_lines)
        for u in updates:
            logger.info(f"Updated {u}")


@hydra.main(config_path="../configs", config_name="benchmark.yaml", version_base="1.1")
def main(cfg: DictConfig) -> None:
    from pepo.utils import init_device_manager, init_hub_manager

    hydra_cfg = HydraConfig.get()
    work_dir = Path(hydra_cfg.runtime.cwd)

    debug = cfg.get("debug", False)
    log_level = "DEBUG" if debug else cfg.get("log_level", "INFO").upper()
    logger = setup_logging(level=log_level)

    logger.info("Finding optimal batch sizes...")
    tasks = cfg.get("tasks", ["train", "eval", "gen"])
    if isinstance(tasks, str):
        tasks = [tasks]

    set_seed(cfg.seed)
    init_device_manager(
        gpu_ids=cfg.get("gpu_ids"),
        dtype=cfg.get("dtype", "bfloat16"),
    )
    init_hub_manager(
        base_dir=cfg.get("hub_base_dir", "PessimisticDPO"),
        push=False,
        load_trainable=True,
    )
    model = instantiate(cfg.model)
    tokenizer = model.get_tokenizer()

    results: dict[str, dict[str, Any]] = {}

    # Test eval first (with optimizer states loaded for realistic memory usage)
    if "eval" in tasks:
        if not model._models:
            model.load(init_new=True)
        batch, err = test_eval_batch_size(model, cfg, tokenizer, logger)
        results["evaluation"] = {"optimal_batch_size": batch, "error": err}

    # Test training (without eval)
    if "train" in tasks:
        if not model._models:
            model.load(init_new=True)
        batch, err = test_train_batch_size(model, cfg, tokenizer, logger)
        results["training"] = {"optimal_batch_size": batch, "error": err}

    if "gen" in tasks:
        if not model._models:
            model.load(init_new=True)
        batch, err = test_gen_batch_size(model, cfg, tokenizer, logger)
        results["generation"] = {"optimal_batch_size": batch, "error": err}

    model.unload()

    logger.info("=" * 40)
    logger.info("RESULTS")
    for task, r in results.items():
        if r["optimal_batch_size"]:
            logger.info(f"{task}: {r['optimal_batch_size']}")
        else:
            logger.info(f"{task}: FAILED ({r['error']})")

    for yaml_file in (work_dir / "configs" / "backbone").glob("*.yaml"):
        if cfg.model.model_id in yaml_file.read_text():
            # Assuming backbone config now, structure is flat in the file
            update_model_config(yaml_file, results, logger)
            break


if __name__ == "__main__":
    main()
