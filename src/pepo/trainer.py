import logging
import threading
from datetime import datetime
from typing import Any, Callable, Optional

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from .utils import DataManager, WandbManager, WandbRun

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for PEPO ensemble models."""

    def __init__(
        self,
        optimizer: Callable[..., torch.optim.Optimizer],
        scheduler_name: str,
        scheduler_num_warmup_steps: int,
        wandb_config: DictConfig,
        train_batch_size: int,
        eval_batch_size: int,
        gradient_accumulation_steps: int = 1,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 0.0,
        log_interval: int = 100,
    ) -> None:
        """
        Initialize trainer.

        Args:
            optimizer: Callable factory for optimizer instantiation (partial).
            scheduler_name: Name of the scheduler to use.
            scheduler_num_warmup_steps: Number of warmup steps for the scheduler.
            wandb_config: Hydra config for wandb settings.
            train_batch_size: Batch size for training.
            eval_batch_size: Batch size for evaluation and generation
                (found via find_optimal_batch_sizes.py).
            gradient_accumulation_steps: Number of steps to accumulate gradients.
            early_stopping_patience: Number of epochs to wait before stopping
                if no improvement. If None, early stopping is disabled.
            early_stopping_min_delta: Minimum change to qualify as an improvement.
        """
        self.optimizer_factory = optimizer
        self.scheduler_name = scheduler_name
        self.scheduler_num_warmup_steps = scheduler_num_warmup_steps
        self.wandb_config = wandb_config
        self.batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.log_interval = log_interval

        self.optimizers: list[torch.optim.Optimizer] = []
        self.schedulers: list[torch.optim.lr_scheduler._LRScheduler] = []
        self.wandb_manager: Optional[WandbManager] = None

    def _setup_training(
        self,
        data_manager: DataManager,
        max_epochs: int,
        wandb_manager: Optional[WandbManager] = None,
    ) -> None:
        """
        Setup optimizers, schedulers, and wandb handlers.
        Internal helper called by train().

        Args:
            data_manager: Data manager for training data.
            max_epochs: Maximum number of epochs to train for.
            wandb_manager: Optional WandbManager instance for logging.
        """
        if self.optimizers:
            return

        self.data_manager = data_manager

        for model_idx in range(self.model.num_networks):
            model_params = self.model.models[model_idx].parameters()

            optimizer = self.optimizer_factory(params=model_params)
            self.optimizers.append(optimizer)

            train_loader = data_manager.get_dataloader(
                model_idx=model_idx,
                partition="train",
                batch_size=self.batch_size,
            )
            num_training_steps = (
                len(train_loader) // self.gradient_accumulation_steps
            ) * max_epochs

            scheduler = get_scheduler(
                name=self.scheduler_name,
                optimizer=optimizer,
                num_warmup_steps=self.scheduler_num_warmup_steps,
                num_training_steps=num_training_steps,
            )
            self.schedulers.append(scheduler)

            logger.info(
                f"Created optimizer and scheduler for model {model_idx}. "
                f"Training steps: {num_training_steps}"
            )

        if wandb_manager is not None:
            self.wandb_manager = wandb_manager

    def train(
        self,
        model: Any,
        data_manager: DataManager,
        max_epochs: int,
        wandb_manager: Optional[WandbManager] = None,
        continue_training: bool = False,
    ) -> None:
        """
        Train the PEPO ensemble models and save the models to the hub.
        Uses threading to run models in parallel on different GPUs.

        Args:
            model: PEPOModel instance to train.
            data_manager: Data manager for training data.
            max_epochs: Maximum number of epochs to train for.
            wandb_manager: Optional WandbManager instance for logging.
        """
        self.model = model

        # Load models if not already loaded
        if self.model._models is None:
            if continue_training:
                model_name = model._get_model_name()
                latest_epoch = model.hub_manager.find_latest_epoch_for_all_submodels(
                    model_name, model.num_networks, max_epoch=max_epochs
                )
                if latest_epoch is None:
                    logger.warning(
                        "Continue training enabled but no checkpoint found. "
                        "Starting training from scratch with new models."
                    )
                    self.model.load_models(init_new=True)
                elif self.model.can_load_from_epoch(latest_epoch):
                    logger.info(
                        f"Continuing training from checkpoint: epoch {latest_epoch}"
                    )
                    self.model.load_from_epoch(latest_epoch)
                else:
                    logger.warning(
                        f"Continue training enabled but cannot load from "
                        f"epoch {latest_epoch}. "
                        "Starting training from scratch with new models."
                    )
                    self.model.load_models(init_new=True)
            else:
                logger.info("Initializing new models for training...")
                self.model.load_models(init_new=True)
        else:
            logger.info("Models already loaded. Using existing models for training.")

        self._setup_training(data_manager, max_epochs, wandb_manager)

        logger.info("Training PEPO ensemble models...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        group = f"{self.model._get_model_name()}-{timestamp}"

        threads = []
        for model_idx in range(self.model.num_networks):
            train_loader = self.data_manager.get_dataloader(
                model_idx=model_idx,
                partition="train",
                batch_size=self.batch_size,
            )
            eval_loader = self.data_manager.get_dataloader(
                model_idx=model_idx,
                partition="eval",
                batch_size=self.eval_batch_size,
            )

            wandb_run = None
            if self.wandb_manager is not None:
                wandb_run = self.wandb_manager.get_training_wandb_handler(
                    model=self.model,
                    data_manager=data_manager,
                    model_idx=model_idx,
                    group=group,
                )

            thread = threading.Thread(
                target=self._train_model,
                args=(
                    model_idx,
                    train_loader,
                    eval_loader,
                    self.optimizers[model_idx],
                    self.schedulers[model_idx],
                    max_epochs,
                    self.gradient_accumulation_steps,
                    wandb_run,
                    self.early_stopping_patience,
                    self.early_stopping_min_delta,
                ),
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        if self.model.factory:
            self.model.factory.save_model(self.model.models)

    def _train_model(
        self,
        model_idx: int,
        train_loader: DataLoader[dict[str, torch.Tensor]],
        eval_loader: DataLoader[dict[str, torch.Tensor]],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        n_epochs: int,
        grad_acc_steps: int,
        wandb_run: Optional[WandbRun],
        es_patience: Optional[int],
        es_min_delta: float,
    ) -> None:
        """
        Train a single model in a thread. Each thread sets its CUDA device context
        to ensure proper GPU isolation.
        """
        device = torch.device(self.model.device_manager.get_device_for_model(model_idx))
        model = self.model.models[model_idx]

        if wandb_run is not None and wandb_run.enabled:
            wandb_run.init_train_run()

        logger = logging.getLogger(__name__)
        train_size = len(train_loader.dataset)  # type: ignore[arg-type]
        eval_size = len(eval_loader.dataset)  # type: ignore[arg-type]
        logger.info(
            f"Model {model_idx} - Train: size={train_size}, "
            f"batches={len(train_loader)} - Eval: size={eval_size}, "
            f"batches={len(eval_loader)}"
        )
        n_batches = len(train_loader)

        start_epoch = self.model.epochs_per_network[model_idx] or 0
        global_step = (
            0 if start_epoch == 0 else start_epoch * n_batches // grad_acc_steps
        )
        if global_step > 0:
            for _ in range(global_step):
                scheduler.step()

        logger.info(
            f"Model {model_idx} - Running evaluation from epoch {start_epoch}..."
        )

        initial_eval_loss = float("inf")
        initial_eval_loss = self._eval_model(
            model_idx=model_idx,
            model=model,
            eval_loader=eval_loader,
            device=device,
            epoch=0,
            n_epochs=n_epochs,
            global_step=global_step,
            wandb_run=wandb_run,
        )

        is_continuing = start_epoch > 0
        if not is_continuing:
            if self.model.factory:
                self.model.factory.push_submodel(model, model_idx, epochs=0)

        best_eval_loss = initial_eval_loss
        patience_counter = 0
        es_enabled = es_patience is not None

        for epoch in range(start_epoch + 1, n_epochs + 1):
            logger.info(
                f"Model {model_idx} - Starting training epoch {epoch}/{n_epochs}"
            )

            model.train()
            optimizer.zero_grad()
            loss_sum = torch.tensor(0.0, device=device)
            lprob_chosen_sum_tensor = torch.tensor(0.0, device=device)
            lprob_reject_sum_tensor = torch.tensor(0.0, device=device)
            margin_sum_tensor = torch.tensor(0.0, device=device)
            ebatch = 0

            desc = f"Model {model_idx} - Epoch {epoch}/{n_epochs}"
            pbar = tqdm(
                total=n_batches // self.log_interval,
                desc=desc,
                position=model_idx,
                leave=False,
                mininterval=1.0,  # Update at most once per second
            )

            for step, batch in enumerate(train_loader):
                batch_loss, lprobs_ch, lprobs_re = self.model._loss_fn(
                    batch, model, device
                )

                loss_sum += batch_loss.detach()
                lprob_chosen_sum_tensor += lprobs_ch.mean().detach()
                lprob_reject_sum_tensor += lprobs_re.mean().detach()
                margin_sum_tensor += (lprobs_ch - lprobs_re).mean().detach()

                batch_loss = batch_loss / grad_acc_steps
                batch_loss.backward()

                if (step + 1) % grad_acc_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    ebatch += 1

                if (step + 1) % self.log_interval == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    loss_val = loss_sum.item() / (step + 1)
                    margin_val = margin_sum_tensor.item() / (step + 1)

                    pbar.set_postfix(
                        {
                            "loss": f"{loss_val:.4f}",
                            "margin": f"{margin_val:.4f}",
                            "lr": f"{current_lr:.2e}",
                        }
                    )
                    pbar.update(1)

                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/learning_rate": current_lr,
                                "train/step": global_step,
                                "train/curr_avg_loss": loss_val,
                                "train/curr_avg_margin": margin_val,
                            },
                            step=global_step,
                        )

            pbar.close()

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/avg_lprobs_chosen": lprob_chosen_sum_tensor.item()
                        / ebatch,
                        "train/avg_lprobs_reject": lprob_reject_sum_tensor.item()
                        / ebatch,
                        "train/avg_margin": margin_sum_tensor.item() / ebatch,
                        "train/epoch": epoch,
                    },
                    step=global_step,
                )

            self.model.epochs_per_network[model_idx] = epoch
            if self.model.factory:
                self.model.factory.push_submodel(model, model_idx, epochs=epoch)

            eval_loss = self._eval_model(
                model_idx=model_idx,
                model=model,
                eval_loader=eval_loader,
                device=device,
                epoch=epoch,
                n_epochs=n_epochs,
                global_step=global_step,
                wandb_run=wandb_run,
            )

            if es_enabled and es_patience is not None:
                if eval_loss < best_eval_loss - es_min_delta:
                    best_eval_loss = eval_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                assert es_patience is not None
                if patience_counter >= es_patience:
                    logger.info(
                        f"Model {model_idx} - Early stopping triggered "
                        f"after {epoch} epochs "
                        f"Best validation loss: {best_eval_loss:.4f}"
                    )
                    break

        if wandb_run is not None:
            wandb_run.finish()

    def _eval_model(
        self,
        model_idx: int,
        model: torch.nn.Module,
        eval_loader: DataLoader[dict[str, torch.Tensor]],
        device: torch.device,
        epoch: int,
        n_epochs: int,
        global_step: int,
        wandb_run: Optional[WandbRun] = None,
    ) -> float:
        """
        Evaluate the model on the evaluation dataset.
        """
        n_batches = len(eval_loader)
        if n_batches == 0:
            raise ValueError("Evaluation loader is empty")

        model.eval()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        loss_sum = torch.tensor(0.0, device=device)
        b = 0
        lprob_chosen_sum_tensor = torch.tensor(0.0, device=device)
        lprob_reject_sum_tensor = torch.tensor(0.0, device=device)
        margin_sum_tensor = torch.tensor(0.0, device=device)

        desc = f"Model {model_idx} - Eval Epoch {epoch}/{n_epochs}"
        pbar = tqdm(
            eval_loader,
            desc=desc,
            position=model_idx,
            leave=False,
            total=n_batches,
            mininterval=1.0,  # Update at most once per second
        )

        with torch.no_grad():
            for batch in pbar:
                batch_loss, lprobs_ch, lprobs_re = self.model._loss_fn(
                    batch, model, device
                )

                loss_sum += batch_loss
                b += 1

                lprob_chosen_sum_tensor += lprobs_ch.mean()
                lprob_reject_sum_tensor += lprobs_re.mean()
                margin_sum_tensor += (lprobs_ch - lprobs_re).mean()

                # Only update postfix periodically to reduce overhead
                if b % max(1, n_batches // 10) == 0 or b == 1:
                    current_avg_loss = loss_sum.item() / b
                    margin_val = margin_sum_tensor.item() / b

                    pbar.set_postfix(
                        {
                            "loss": f"{current_avg_loss:.4f}",
                            "margin": f"{margin_val:.4f}",
                        }
                    )

        pbar.close()

        if wandb_run is not None:
            wandb_run.log(
                {
                    "eval/loss": loss_sum.item() / b,
                    "eval/avg_lprobs_chosen": lprob_chosen_sum_tensor.item() / b,
                    "eval/avg_lprobs_reject": lprob_reject_sum_tensor.item() / b,
                    "eval/avg_margin": margin_sum_tensor.item() / b,
                    "eval/epoch": epoch,
                },
                step=global_step,
            )

        return loss_sum.item() / b
