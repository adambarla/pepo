#!/usr/bin/env python3
"""
Script to find optimal batch sizes for training and generation tasks.
"""

import os

os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set multiprocessing start method to 'spawn' to avoid deadlocks with CUDA and threading
import multiprocessing

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    raise RuntimeError(
        "Failed to set multiprocessing start method to 'spawn'. "
        "This is required for DataLoader workers to work correctly "
        "with CUDA and threading."
    )

import io
import logging
import sys
import types
from pathlib import Path  # noqa: E402
from typing import Any, Dict, Optional, Tuple  # noqa: E402

import hydra  # noqa: E402
import torch  # noqa: E402
import tqdm  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

import pepo.model  # noqa: E402
from pepo.utils import constants, set_seed, setup_logging  # noqa: E402


class DisabledTqdm:
    def __init__(self, *args, **kwargs):
        self.iterable = args[0] if args else range(0)
        self.total = len(self.iterable) if hasattr(self.iterable, "__len__") else None

    def __iter__(self):
        return iter(self.iterable)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def update(self, n=1):
        pass

    def close(self):
        pass

    def set_description(self, *args, **kwargs):
        pass

    def set_postfix(self, *args, **kwargs):
        pass


setattr(tqdm, "tqdm", DisabledTqdm)
sys.modules["tqdm"].tqdm = DisabledTqdm  # type: ignore[attr-defined]

setattr(pepo.model, "tqdm", DisabledTqdm)
for name in dir(pepo.model):
    obj = getattr(pepo.model, name)
    if isinstance(obj, types.ModuleType) and hasattr(obj, "tqdm"):
        setattr(obj, "tqdm", DisabledTqdm)

OmegaConf.register_new_resolver(
    "pepo.constants",
    lambda name: getattr(constants, name),
)


def test_batch_size(
    cfg: DictConfig,
    task: str,
    max_epochs: int = 1,
) -> Tuple[Optional[int], Optional[str]]:
    """
    Test a task with increasing batch sizes until failure.
    """
    logger = logging.getLogger("pepo")
    batch_size = 1
    optimal_batch_size = None

    device_manager = instantiate(cfg.device)
    hub_manager = instantiate(cfg.hub)

    model = instantiate(
        cfg.model,
        device_manager=device_manager,
        hub_manager=hub_manager,
    )
    if not model._models:
        model.load_models(init_new=True)
    tokenizer = model.get_tokenizer()
    if hasattr(tokenizer, "vocab_size") and tokenizer.vocab_size is not None:
        vocab_size = tokenizer.vocab_size
    else:
        vocab_dict = tokenizer.get_vocab()
        vocab_size = len(vocab_dict)

    # Determine max_length based on model config if not provided in dataset config
    model_config = getattr(model.models[0], "config", None)
    model_max_len = getattr(model_config, "max_position_embeddings", None)

    if task == "training" or task == "evaluation":
        max_length = cfg.dataset.max_length
        if max_length is None:
            if model_max_len is not None:
                max_length = min(model_max_len, 8192)
                logger.info(
                    f"Using model's max_position_embeddings for {task} test: "
                    f"{max_length}"
                )
            else:
                max_length = 2048
                logger.info(
                    f"Using default fallback length for {task} test: {max_length}"
                )
        else:
            logger.info(f"Using configured max_length for {task} test: {max_length}")

    elif task == "generation":
        max_prompt_length = cfg.dataset.max_prompt_length
        if max_prompt_length is None:
            if model_max_len is not None:
                # For generation, we test with half context as prompt
                max_prompt_length = min(model_max_len, 8192) // 2
                logger.info(
                    f"Using half of model's max_position_embeddings for "
                    f"generation prompt: {max_prompt_length}"
                )
            else:
                max_length = cfg.dataset.max_length
                if max_length is not None:
                    max_prompt_length = max_length // 2
                else:
                    max_prompt_length = 1024  # Default fallback
                    logger.info(
                        f"Using default fallback prompt length for "
                        f"generation test: {max_prompt_length}"
                    )

        max_new_tokens = cfg.model.generator.get("max_new_tokens", 1000)
        input_length = max_prompt_length + max_new_tokens - 10
        max_total_length = max_prompt_length + max_new_tokens

    while True:
        try:
            torch.cuda.empty_cache()

            if task == "training":
                effective_batch_size = cfg.get("effective_batch_size")
                if effective_batch_size is None:
                    effective_batch_size = batch_size

                chosen_ids = torch.randint(
                    0, vocab_size, (batch_size, max_length), dtype=torch.long
                )
                reject_ids = torch.randint(
                    0, vocab_size, (batch_size, max_length), dtype=torch.long
                )
                chosen_amask = torch.ones((batch_size, max_length), dtype=torch.long)
                reject_amask = torch.ones((batch_size, max_length), dtype=torch.long)

                prompt_length = max_length // 2
                chosen_rmask = torch.zeros((batch_size, max_length), dtype=torch.long)
                chosen_rmask[:, prompt_length:] = 1
                reject_rmask = torch.zeros((batch_size, max_length), dtype=torch.long)
                reject_rmask[:, prompt_length:] = 1

                device = torch.device(device_manager.get_device_for_model(0))
                chosen_ids = chosen_ids.to(device)
                reject_ids = reject_ids.to(device)
                chosen_amask = chosen_amask.to(device)
                reject_amask = reject_amask.to(device)
                chosen_rmask = chosen_rmask.to(device)
                reject_rmask = reject_rmask.to(device)

                batch = {
                    "chosen_input_ids": chosen_ids,
                    "chosen_attention_mask": chosen_amask,
                    "chosen_response_mask": chosen_rmask,
                    "rejected_input_ids": reject_ids,
                    "rejected_attention_mask": reject_amask,
                    "rejected_response_mask": reject_rmask,
                }

                test_model = model.models[0]
                test_model.train()
                test_model = test_model.to(device)

                loss_fn = model._loss_fn
                loss, _, _ = loss_fn(batch, test_model, device)
                loss.backward()

                del (
                    batch,
                    chosen_ids,
                    reject_ids,
                    chosen_amask,
                    reject_amask,
                    chosen_rmask,
                    reject_rmask,
                )

            elif task == "evaluation":
                effective_batch_size = cfg.get("effective_batch_size")
                if effective_batch_size is None:
                    effective_batch_size = batch_size

                # Initialize trainer to handle partial and ensure optimizer
                # creation logic is consistent
                # Use the method on the model to avoid manual functools handling
                if hasattr(model, "init_trainer"):
                    model.init_trainer()

                # Evaluation needs to account for the fact that during eval,
                # we compute full sequence logits but we do it per batch.
                # The critical factor is (batch_size * sequence_length *
                # vocab_size)
                # The OOM in your logs happened inside `_get_lprobs` ->
                # `log_probs = F.log_softmax(logits, dim=-1)`

                # Also, crucially, the optimizer state occupies significant
                # memory during the training loop.
                # We must simulate this memory usage for the "eval during
                # training" scenario.
                test_model = model.models[0]
                test_model.train()

                # Create optimizer for this model to reserve memory
                if model.trainer and model.trainer.optimizer_factory:
                    try:
                        model.trainer.optimizer_factory(params=test_model.parameters())
                    except Exception:
                        # If optimizer creation fails (e.g. OOM), catch it later
                        # or here. But we are inside the retry loop? No, this is
                        # inside while True. If optimizer creation OOMs, then
                        # even batch_size=0 fails? The surrounding try/except
                        # will catch OOM.
                        pass

                chosen_ids = torch.randint(
                    0, vocab_size, (batch_size, max_length), dtype=torch.long
                )
                reject_ids = torch.randint(
                    0, vocab_size, (batch_size, max_length), dtype=torch.long
                )
                chosen_amask = torch.ones((batch_size, max_length), dtype=torch.long)
                reject_amask = torch.ones((batch_size, max_length), dtype=torch.long)

                prompt_length = max_length // 2
                chosen_rmask = torch.zeros((batch_size, max_length), dtype=torch.long)
                chosen_rmask[:, prompt_length:] = 1
                reject_rmask = torch.zeros((batch_size, max_length), dtype=torch.long)
                reject_rmask[:, prompt_length:] = 1

                device = torch.device(device_manager.get_device_for_model(0))
                chosen_ids = chosen_ids.to(device)
                reject_ids = reject_ids.to(device)
                chosen_amask = chosen_amask.to(device)
                reject_amask = reject_amask.to(device)
                chosen_rmask = chosen_rmask.to(device)
                reject_rmask = reject_rmask.to(device)

                batch = {
                    "chosen_input_ids": chosen_ids,
                    "chosen_attention_mask": chosen_amask,
                    "chosen_response_mask": chosen_rmask,
                    "rejected_input_ids": reject_ids,
                    "rejected_attention_mask": reject_amask,
                    "rejected_response_mask": reject_rmask,
                }

                # Test evaluation by running _loss_fn with no_grad
                # Since L affects VRAM during training/eval, we should ideally
                # test with L models if possible
                # But here we are just testing if a single model can handle the
                # batch size during eval step
                # The trainer runs eval sequentially or in parallel threads
                # per model.
                # Each model is on its own GPU (usually), so testing one model
                # is sufficient for per-GPU VRAM.

                test_model.eval()
                test_model = test_model.to(device)

                loss_fn = model._loss_fn
                with torch.no_grad():
                    loss_fn(batch, test_model, device)

                del (
                    batch,
                    chosen_ids,
                    reject_ids,
                    chosen_amask,
                    reject_amask,
                    chosen_rmask,
                    reject_rmask,
                )

            elif task == "generation":
                if "generator" not in cfg.model:
                    return None, "Generator not configured in model config"

                generator = instantiate(cfg.model.generator)
                model.generator = generator

                input_ids = torch.randint(
                    0, vocab_size, (batch_size, input_length), dtype=torch.long
                )
                attention_mask = torch.ones(
                    (batch_size, input_length), dtype=torch.long
                )

                device = torch.device(device_manager.get_device_for_model(0))
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                old_stderr = sys.stderr
                sys.stderr = io.StringIO()
                try:
                    generator.generate(
                        model=model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=max_total_length,
                        use_ensamble=generator.use_ensamble,
                        top_p_sampling=generator.top_p_sampling,
                        greedy_sampling=generator.greedy_sampling,
                        temperature=generator.temperature,
                        top_p=generator.top_p,
                    )
                except (RuntimeError, torch.cuda.OutOfMemoryError) as gen_error:
                    error_msg = str(gen_error)
                    if (
                        "out of memory" in error_msg.lower()
                        or "cuda" in error_msg.lower()
                        or "OutOfMemoryError" in str(type(gen_error).__name__)
                    ):
                        sys.stderr = old_stderr
                        raise RuntimeError("OOM") from None
                    sys.stderr = old_stderr
                    raise
                finally:
                    sys.stderr = old_stderr

                del input_ids, attention_mask
            else:
                return None, f"Unknown task: {task}"

            optimal_batch_size = batch_size
            batch_size *= 2
            torch.cuda.empty_cache()
            logger.info(f"OK batch_size={optimal_batch_size}")

        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            error_msg = str(e)
            if (
                "out of memory" in error_msg.lower()
                or "cuda" in error_msg.lower()
                or "OutOfMemoryError" in str(type(e).__name__)
            ):
                torch.cuda.empty_cache()
                return optimal_batch_size, "OOM"
            else:
                return optimal_batch_size, error_msg
        except Exception as e:
            error_msg = str(e)
            if "OutOfMemoryError" in error_msg or "out of memory" in error_msg.lower():
                torch.cuda.empty_cache()
                return optimal_batch_size, "OOM"
            return optimal_batch_size, error_msg

    if task in ["training", "generation", "evaluation"]:
        model.unload_models()


@hydra.main(
    config_path="../configs",
    config_name="benchmark.yaml",
    version_base="1.1",
)
def main(cfg: DictConfig) -> None:
    hydra_cfg = HydraConfig.get()
    original_work_dir = Path(hydra_cfg.runtime.cwd)

    test_training = cfg.get("test_training", True)
    test_generation = cfg.get("test_generation", True)
    test_evaluation = cfg.get("test_evaluation", True)
    max_epochs = cfg.get("max_epochs", 1)

    log_level_str = cfg.log_level.upper()
    logger = setup_logging(level=log_level_str)

    set_seed(cfg.seed)

    results: Dict[str, Dict[str, Any]] = {}

    if test_training:
        logger.info("Testing TRAINING (starting from batch_size=1)")
        optimal_batch, error = test_batch_size(
            cfg,
            "training",
            max_epochs=max_epochs,
        )
        results["training"] = {
            "optimal_batch_size": optimal_batch,
            "error": error,
        }
        if error:
            logger.warning(f"Training test failed: {error}")

    if test_generation:
        logger.info("Testing GENERATION (starting from batch_size=1)")
        optimal_batch, error = test_batch_size(
            cfg,
            "generation",
        )
        results["generation"] = {
            "optimal_batch_size": optimal_batch,
            "error": error,
        }
        if error:
            logger.warning(f"Generation test failed: {error}")

    if test_evaluation:
        logger.info("Testing EVALUATION (starting from batch_size=1)")
        optimal_batch, error = test_batch_size(
            cfg,
            "evaluation",
        )
        results["evaluation"] = {
            "optimal_batch_size": optimal_batch,
            "error": error,
        }
        if error:
            logger.warning(f"Evaluation test failed: {error}")

    model_id = cfg.model.get("model_id", "unknown")
    config_info = {
        "model": model_id,
        "L": cfg.L,
        "max_length": cfg.dataset.max_length,
        "max_prompt_length": cfg.dataset.max_prompt_length,
    }
    if test_generation and "generator" in cfg.model:
        config_info["max_new_tokens"] = cfg.model.generator.get("max_new_tokens", 1000)
    if test_training and "effective_batch_size" in cfg:
        config_info["effective_batch_size"] = cfg.effective_batch_size

    logger.info("\nConfiguration:")
    for key, value in config_info.items():
        logger.info(f"  {key}: {value}")

    logger.info("\nOptimal Batch Sizes:")
    for task, result in results.items():
        if result["optimal_batch_size"]:
            logger.info(f"  {task}: {result['optimal_batch_size']}")
        else:
            logger.info(f"  {task}: FAILED ({result['error']})")

    output_file = original_work_dir / "optimal_batch_sizes.txt"
    with open(output_file, "w") as f:
        f.write("Optimal Batch Sizes\n\n")
        f.write(f"Model: {model_id}\n\n")
        f.write("Configuration:\n")
        for key, value in config_info.items():
            if key != "model":
                f.write(f"  {key}: {value}\n")
        f.write("\nOptimal Batch Sizes:\n")
        for task, result in results.items():
            if result["optimal_batch_size"]:
                f.write(f"  {task}: {result['optimal_batch_size']}\n")
            else:
                f.write(f"  {task}: FAILED ({result['error']})\n")

    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
