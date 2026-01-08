"""Single Model Trainer for sequential training of a single model."""

import logging
from typing import TYPE_CHECKING, Any, Optional, cast

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from ..utils import DataManager, WandbManager, WandbRun
from .base import GenericTrainer

if TYPE_CHECKING:
    from ..model import BaseModel

logger = logging.getLogger(__name__)


class SingleModelTrainer(GenericTrainer):
    """
    Standard sequential trainer for single-model architectures.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None

    def _setup_training(
        self,
        model: "BaseModel",
        data_manager: DataManager,
        max_epochs: int,
        wandb_manager: Optional[WandbManager] = None,
    ) -> None:
        """Setup optimizer, scheduler, and wandb handlers."""
        if self.optimizer is not None:
            return

        self.model = model
        self.data_manager = data_manager

        # Subclasses can override if parameter selection is different.
        # Default to the first model in the ensemble/container.
        model_params = self.model.models[0].parameters()

        self.optimizer = self.optimizer_factory(params=model_params)

        train_loader = data_manager.get_dataloader(
            model_idx=0,
            partition="train",
            batch_size=self.batch_size,
        )
        num_training_steps = (
            len(train_loader) // self.gradient_accumulation_steps
        ) * max_epochs

        self.scheduler = get_scheduler(
            name=self.scheduler_name,
            optimizer=self.optimizer,
            num_warmup_steps=self.scheduler_num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        if wandb_manager is not None:
            self.wandb_manager = wandb_manager

        logger.info(
            f"Created optimizer and scheduler. Training steps: {num_training_steps}"
        )

    def train(
        self,
        model: "BaseModel",
        data_manager: DataManager,
        max_epochs: int,
        wandb_manager: Optional[WandbManager] = None,
        continue_training: bool = False,
    ) -> None:
        """Sequential training loop."""
        self.model = model

        # Initial loading
        if self.model._models is None:
            self.model.load(init_new=not continue_training)

        self._setup_training(model, data_manager, max_epochs, wandb_manager)

        device = torch.device(model.device_manager.get_device_for_model(0))

        group = model.get_name()
        wandb_run = None
        if self.wandb_manager is not None:
            wandb_run = self.wandb_manager.get_training_wandb_handler(
                model=model,
                data_manager=data_manager,
                model_idx=0,
                group=group,
            )
            if wandb_run is not None and wandb_run.enabled:
                wandb_run.init_run()

        # Initial eval
        if not self.skip_eval:
            self._eval_epoch(
                model=model,
                eval_loader=data_manager.get_dataloader(
                    0, "eval", self.eval_batch_size
                ),
                device=device,
                epoch=0,
                global_step=0,
                wandb_run=wandb_run,
            )

        logger.info(f"Starting single model training for {max_epochs} epochs")

        for epoch in range(1, max_epochs + 1):
            if self.scheduler is None:
                raise ValueError("Scheduler not initialized")
            if self.optimizer is None:
                raise ValueError("Optimizer not initialized")

            self._train_epoch(
                model=model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                train_loader=data_manager.get_dataloader(0, "train", self.batch_size),
                device=device,
                epoch=epoch,
                n_epochs=max_epochs,
                grad_acc_steps=self.gradient_accumulation_steps,
                wandb_run=wandb_run,
            )

            if self.early_stopping_patience is not None:
                # For early stopping we use validation loss
                # Note: We can implement validation set if needed
                pass

            # Checkpointing logic
            # We always save the latest model
            model_any = cast(Any, model)
            if model_any.loader:
                model_any.loader.push_model(
                    model=model.models[0],
                    model_name=model.get_name(),
                    tokenizer=model.tokenizer,
                    epochs=epoch,
                )

            if not self.skip_eval:
                n_batches = len(
                    data_manager.get_dataloader(0, "train", self.batch_size)
                )
                global_step = epoch * n_batches // self.gradient_accumulation_steps
                self._eval_epoch(
                    model=model,
                    eval_loader=data_manager.get_dataloader(
                        0, "eval", self.eval_batch_size
                    ),
                    device=device,
                    epoch=epoch,
                    global_step=global_step,
                    wandb_run=wandb_run,
                )

        if wandb_run is not None:
            wandb_run.finish()

    def _eval_epoch(
        self,
        model: "BaseModel",
        eval_loader: DataLoader[dict[str, torch.Tensor]],
        device: torch.device,
        epoch: int,
        global_step: int,
        wandb_run: Optional[WandbRun] = None,
    ) -> float:
        model.models[0].eval()
        metric_sums: dict[str, float] = {}
        n_batches = len(eval_loader)
        if n_batches == 0:
            return 0.0

        pbar = tqdm(
            total=n_batches,
            desc=f"Eval - E{epoch}",
            position=0,
            leave=False,
        )

        with torch.no_grad():
            for batch in eval_loader:
                loss, metrics = model.loss_fn(batch, model.models[0], device)
                for k, v in metrics.items():
                    metric_sums[k] = metric_sums.get(k, 0.0) + v
                pbar.update(1)

        pbar.close()

        avg_metrics = {f"eval/{k}": v / n_batches for k, v in metric_sums.items()}
        avg_metrics["eval/epoch"] = epoch

        if wandb_run:
            wandb_run.log(avg_metrics, step=global_step)

        return avg_metrics.get("eval/loss", 0.0)

    def _train_epoch(
        self,
        model: "BaseModel",
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_loader: DataLoader[dict[str, torch.Tensor]],
        device: torch.device,
        epoch: int,
        n_epochs: int,
        grad_acc_steps: int,
        wandb_run: Optional[WandbRun] = None,
        stop_event: Optional[Any] = None,
    ) -> dict[str, float]:
        model.models[0].train()
        total_loss = 0.0
        num_batches = 0
        global_step = (epoch - 1) * len(train_loader) // grad_acc_steps

        pbar = tqdm(
            train_loader,
            desc=f"Training Epoch {epoch}/{n_epochs}",
            dynamic_ncols=True,
        )

        metrics_tracker: dict[str, float] = {}

        for batch_idx, batch in enumerate(pbar):
            if stop_event is not None and stop_event.is_set():
                break

            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            loss, metrics = model.loss_fn(batch, model.models[0], device)

            # Scale loss for gradient accumulation
            scaled_loss = loss / grad_acc_steps

            # Backward pass
            scaled_loss.backward()  # type: ignore[no-untyped-call]

            total_loss += loss.item()
            num_batches += 1

            # Accumulate metrics
            for k, v in metrics.items():
                if k not in metrics_tracker:
                    metrics_tracker[k] = 0.0
                metrics_tracker[k] += v

            # Update weights if gradient accumulation steps reached
            if (batch_idx + 1) % grad_acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if wandb_run:
                    # Log metrics for the accumulated step
                    avg_metrics = {
                        k: v / grad_acc_steps for k, v in metrics_tracker.items()
                    }
                    prefixed_metrics = {f"train/{k}": v for k, v in avg_metrics.items()}
                    prefixed_metrics["train/step"] = global_step
                    prefixed_metrics["train/lr"] = scheduler.get_last_lr()[0]
                    wandb_run.log(prefixed_metrics, step=global_step)
                    metrics_tracker.clear()  # Reset metrics for next accumulation

        pbar.close()

        # Handle any remaining metrics if the loop finished without a full accumulation step
        if metrics_tracker and wandb_run:
            avg_metrics = {
                k: v / (num_batches % grad_acc_steps or grad_acc_steps)
                for k, v in metrics_tracker.items()
            }
            prefixed_metrics = {f"train/{k}": v for k, v in avg_metrics.items()}
            prefixed_metrics["train/step"] = (
                global_step + 1
            )  # Increment for the last partial step
            prefixed_metrics["train/lr"] = scheduler.get_last_lr()[0]
            wandb_run.log(prefixed_metrics, step=global_step + 1)

        # Return average metrics for the epoch
        if num_batches > 0:
            final_avg_metrics = {k: v / num_batches for k, v in metrics_tracker.items()}
            return final_avg_metrics
        return {}
