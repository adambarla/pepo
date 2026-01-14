"""Single Model Trainer for sequential training of a single model."""

import logging
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from ..data import DataManager
from ..utils import WandbManager, WandbRun
from .base import BaseTrainer

if TYPE_CHECKING:
    from ..model import BaseModel

logger = logging.getLogger(__name__)


class SingleModelTrainer(BaseTrainer):
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
        max_epochs: Optional[int] = None,
        wandb_manager: Optional[WandbManager] = None,
        continue_training: bool = False,
        **kwargs: Any,
    ) -> None:
        """Sequential training loop."""
        self.model = model

        if max_epochs is None:
            max_epochs = self.training_epochs

        if max_epochs is None:
            raise ValueError(
                "max_epochs not provided to train() and not configured in trainer."
            )

        if self.model._models is None:
            load_kwargs = kwargs.get("load_kwargs", {})
            self.model.load(init_new=not continue_training, **load_kwargs)

        self._setup_training(model, data_manager, max_epochs, wandb_manager)

        device = torch.device(model.device_manager.get_device_for_model(0))

        group = model.get_name()
        wandb_run = None
        if self.wandb_manager is not None:
            m_idx = None if "policy" in group.lower() else 0
            wandb_run = self.wandb_manager.get_training_wandb_handler(
                model=model,
                data_manager=data_manager,
                model_idx=m_idx,
                group=group,
                extra_tags=["policy"],
            )
            if wandb_run is not None and wandb_run.enabled:
                wandb_run.init_run()

        # Initial eval
        best_eval_loss = float("inf")
        if not self.skip_eval:
            best_eval_loss = self._eval_epoch(
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

        patience_counter = 0
        es_min_delta = self.early_stopping_min_delta

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

            model.set_epoch(epoch)
            model.save()

            # Run eval and track best model
            eval_loss = best_eval_loss
            if not self.skip_eval:
                n_batches = len(
                    data_manager.get_dataloader(0, "train", self.batch_size)
                )
                global_step = epoch * n_batches // self.gradient_accumulation_steps
                eval_loss = self._eval_epoch(
                    model=model,
                    eval_loader=data_manager.get_dataloader(
                        0, "eval", self.eval_batch_size
                    ),
                    device=device,
                    epoch=epoch,
                    global_step=global_step,
                    wandb_run=wandb_run,
                )

            # Track best model and save without epoch suffix when eval improves
            if eval_loss < best_eval_loss - es_min_delta:
                best_eval_loss = eval_loss
                patience_counter = 0
                # Push best model (no epoch suffix)
                if model.checkpoint_manager:
                    logger.info(
                        f"New best eval loss: {eval_loss:.4f}. "
                        f"Pushing best model checkpoint."
                    )
                    model.checkpoint_manager.push_model(
                        model=model.models[0],
                        model_name=model.get_name(),
                        tokenizer=model.tokenizer,
                        epochs=None,  # No epoch suffix = best model
                    )
            elif self.early_stopping_patience is not None:
                patience_counter += 1

            if (
                self.early_stopping_patience is not None
                and patience_counter >= self.early_stopping_patience
            ):
                logger.info(
                    f"Early stopping triggered after {epoch} epochs. "
                    f"Best validation loss: {best_eval_loss:.4f}"
                )
                break

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
        """Evaluate the model for one epoch."""
        model.models[0].eval()
        accumulated_metrics: dict[str, float] = {}
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
                if "loss" not in metrics:
                    metrics["loss"] = loss.item()

                for k, v in metrics.items():
                    accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + v
                pbar.update(1)

        pbar.close()

        avg_epoch_metrics = self._compute_avg_metrics(accumulated_metrics, n_batches)
        self._log_metrics(
            wandb_run=wandb_run,
            metrics=avg_epoch_metrics,
            step=global_step,
            prefix="eval",
            add_avg_prefix=True,
            additional_log_items={"eval/epoch": epoch},
        )

        return avg_epoch_metrics.get("loss", 0.0)

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
        global_step = (epoch - 1) * len(train_loader) // grad_acc_steps

        pbar = tqdm(
            train_loader,
            desc=f"Training E{epoch}/{n_epochs}",
            dynamic_ncols=True,
        )

        accumulated_metrics: dict[str, float] = {}
        last_logged_metrics: dict[str, float] = {}
        samples_count = 0

        for batch_idx, batch in enumerate(pbar):
            if stop_event is not None and stop_event.is_set():
                break

            loss, metrics = model.loss_fn(batch, model.models[0], device)
            scaled_loss = loss / grad_acc_steps
            scaled_loss.backward()

            if "loss" not in metrics:
                metrics["loss"] = loss.item()

            for k, v in metrics.items():
                accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + v

            samples_count += 1

            if (batch_idx + 1) % grad_acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            if (batch_idx + 1) % self.log_interval == 0:
                interval_metrics = {
                    k: v - last_logged_metrics.get(k, 0.0)
                    for k, v in accumulated_metrics.items()
                }
                last_logged_metrics = accumulated_metrics.copy()

                avg_interval_metrics = self._compute_avg_metrics(
                    interval_metrics, self.log_interval
                )

                avg_loss = avg_interval_metrics.get("loss", 0.0)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

            self._log_metrics(
                wandb_run=wandb_run,
                metrics=avg_interval_metrics,
                step=global_step,
                prefix="train",
                add_avg_prefix=False,
                additional_log_items={"train/lr": scheduler.get_last_lr()[0]},
            )

        pbar.close()

        # Log epoch averages
        if samples_count > 0:
            avg_epoch_metrics = self._compute_avg_metrics(
                accumulated_metrics, samples_count
            )
            self._log_metrics(
                wandb_run=wandb_run,
                metrics=avg_epoch_metrics,
                step=global_step,
                prefix="train",
                add_avg_prefix=True,
                additional_log_items={"train/epoch": epoch},
            )

        # Return average metrics for the epoch
        if samples_count > 0:
            return self._compute_avg_metrics(accumulated_metrics, samples_count)
        return {}
