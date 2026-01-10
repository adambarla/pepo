"""Ensemble Trainer for parallel training of multiple models (DEPPO, REPPOReward)."""

import logging
import threading
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


class EnsembleTrainer(BaseTrainer):
    """
    Trainer for ensemble models that supports parallel training across GPUs.
    Adapted from DEPPOTrainer to be generic for any BaseModel returning (loss, metrics).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.optimizers: list[torch.optim.Optimizer] = []
        self.schedulers: list[torch.optim.lr_scheduler.LRScheduler] = []

    def _setup_training(
        self,
        data_manager: DataManager,
        max_epochs: int,
        wandb_manager: Optional[WandbManager] = None,
    ) -> None:
        """Setup optimizers, schedulers, and wandb handlers."""
        if self.optimizers:
            return

        self.data_manager = data_manager

        for model_idx in range(self.model.num_models):
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
        model: "BaseModel",
        data_manager: DataManager,
        max_epochs: Optional[int] = None,
        wandb_manager: Optional[WandbManager] = None,
        continue_training: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Train the ensemble models using threading for parallel GPU operation.
        """
        self.model = model

        if max_epochs is None:
            max_epochs = self.training_epochs

        if max_epochs is None:
            raise ValueError(
                "max_epochs not provided to train() and not configured in trainer."
            )

        # Initial loading logic
        if self.model._models is None:
            if continue_training:
                # Use find_latest_epoch from BaseModel
                latest_epoch = self.model.find_latest_epoch(max_epoch=max_epochs)
                if latest_epoch is None:
                    logger.warning(
                        "Continue training enabled but no checkpoint found. "
                        "Starting training from scratch with new models."
                    )
                    self.model.load(init_new=True)
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
                    self.model.load(init_new=True)
            else:
                logger.info("Initializing new models for training...")
                self.model.load(init_new=True)
        else:
            logger.info("Models already loaded. Using existing models for training.")

        self._setup_training(data_manager, max_epochs, wandb_manager)

        logger.info(f"Training {self.model.num_models} ensemble models...")

        group = self.model.get_name()
        threads = []
        stop_event = threading.Event()

        def run_training(**kwargs: Any) -> None:
            """Wrapper that catches exceptions from training threads."""
            try:
                self._train_model(**kwargs)
            except InterruptedError:
                pass
            except Exception as e:
                if not stop_event.is_set():
                    logger.error(
                        f"Training thread for model {kwargs['model_idx']} failed: {e}"
                    )
                stop_event.set()

        for model_idx in range(self.model.num_models):
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
                    extra_tags=["reward"],
                )

            thread = threading.Thread(
                target=run_training,
                kwargs=dict(
                    model_idx=model_idx,
                    train_loader=train_loader,
                    eval_loader=eval_loader,
                    optimizer=self.optimizers[model_idx],
                    scheduler=self.schedulers[model_idx],
                    n_epochs=max_epochs,
                    grad_acc_steps=self.gradient_accumulation_steps,
                    wandb_run=wandb_run,
                    stop_event=stop_event,
                ),
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        if stop_event.is_set():
            raise RuntimeError("Training failed - see error log above")

        self.model.save()

    def _train_model(
        self,
        model_idx: int,
        train_loader: DataLoader[dict[str, torch.Tensor]],
        eval_loader: DataLoader[dict[str, torch.Tensor]],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        n_epochs: int,
        grad_acc_steps: int,
        wandb_run: Optional[WandbRun],
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        """Train a single model in a thread."""
        gpu_id = self.model.device_manager._get_gpu_id_for_model(model_idx)
        torch.cuda.set_device(gpu_id)
        device = torch.device(self.model.device_manager.get_device_for_model(model_idx))
        model = self.model.models[model_idx]

        if wandb_run is not None and wandb_run.enabled:
            wandb_run.init_run()

        n_batches = len(train_loader)
        start_epoch = self.model.get_epoch(model_idx=model_idx)
        global_step = (
            0 if start_epoch == 0 else start_epoch * n_batches // grad_acc_steps
        )

        if global_step > 0:
            for _ in range(global_step):
                scheduler.step()

        initial_eval_loss = float("inf")
        if not self.skip_eval:
            initial_eval_loss = self._eval_epoch(
                model_idx=model_idx,
                model=model,
                eval_loader=eval_loader,
                device=device,
                epoch=start_epoch,
                n_epochs=n_epochs,
                global_step=global_step,
                wandb_run=wandb_run,
            )

        if start_epoch == 0:
            self.model.checkpoint_manager.push_model(
                model=model,
                model_name=self.model.get_name(model_idx=model_idx),
                tokenizer=self.model.tokenizer,
                epochs=0,
            )

        best_eval_loss = initial_eval_loss
        patience_counter = 0
        es_patience = self.early_stopping_patience
        es_min_delta = self.early_stopping_min_delta

        for epoch in range(start_epoch + 1, n_epochs + 1):
            if stop_event is not None and stop_event.is_set():
                logger.warning(
                    f"Model {model_idx} stopping early due to error in another thread"
                )
                break

            self._train_epoch(
                model_idx=model_idx,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                device=device,
                epoch=epoch,
                n_epochs=n_epochs,
                grad_acc_steps=grad_acc_steps,
                wandb_run=wandb_run,
                stop_event=stop_event,
            )

            self.model.set_epoch(epoch, model_idx=model_idx)
            if self.model.checkpoint_manager:
                self.model.checkpoint_manager.push_model(
                    model=model,
                    model_name=self.model.get_name(model_idx=model_idx),
                    tokenizer=self.model.tokenizer,
                    epochs=epoch,
                )

            eval_loss = best_eval_loss
            if not self.skip_eval:
                eval_loss = self._eval_epoch(
                    model_idx=model_idx,
                    model=model,
                    eval_loader=eval_loader,
                    device=device,
                    epoch=epoch,
                    n_epochs=n_epochs,
                    global_step=epoch * len(train_loader) // grad_acc_steps,
                    wandb_run=wandb_run,
                )

            # Track best model and save without epoch suffix when eval loss improves
            if eval_loss < best_eval_loss - es_min_delta:
                best_eval_loss = eval_loss
                patience_counter = 0
                # Push best model (no epoch suffix)
                if self.model.checkpoint_manager:
                    logger.info(
                        f"Model {model_idx} - New best eval loss: {eval_loss:.4f}. "
                        f"Pushing best model checkpoint."
                    )
                    self.model.checkpoint_manager.push_model(
                        model=model,
                        model_name=self.model.get_name(model_idx=model_idx),
                        tokenizer=self.model.tokenizer,
                        epochs=None,  # No epoch suffix = best model
                    )
            elif es_patience is not None:
                patience_counter += 1

            if es_patience is not None and patience_counter >= es_patience:
                logger.info(
                    f"Model {model_idx} - Early stopping triggered "
                    f"after {epoch} epochs. "
                    f"Best validation loss: {best_eval_loss:.4f}"
                )
                break

        if wandb_run is not None:
            wandb_run.finish()

        if wandb_run is not None:
            wandb_run.finish()

    def _train_epoch(
        self,
        model_idx: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_loader: DataLoader[dict[str, torch.Tensor]],
        device: torch.device,
        epoch: int,
        n_epochs: int,
        grad_acc_steps: int,
        wandb_run: Optional[WandbRun],
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        logger.info(f"Model {model_idx} - Training epoch {epoch}/{n_epochs}")

        n_batches = len(train_loader)
        global_step = (epoch - 1) * n_batches // grad_acc_steps

        model.train()
        optimizer.zero_grad()

        accumulated_metrics: dict[str, float] = {}
        last_logged_metrics: dict[str, float] = {}
        samples_count = 0

        desc = f"Model {model_idx} - E{epoch}/{n_epochs}"
        pbar = tqdm(
            total=n_batches // self.log_interval,
            desc=desc,
            position=model_idx,
            leave=False,
            mininterval=1.0,
        )

        for step, batch in enumerate(train_loader):
            if stop_event is not None and stop_event.is_set():
                pbar.close()
                raise InterruptedError(
                    "Training stopped due to error in another thread"
                )

            if (
                self.max_batches_per_epoch is not None
                and step >= self.max_batches_per_epoch
            ):
                pbar.close()
                break

            loss, metrics = self.model.loss_fn(batch, model, device)
            loss_val = loss.item()
            if "loss" not in metrics:
                metrics["loss"] = loss_val

            for k, v in metrics.items():
                accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + v

            samples_count += 1
            scaled_loss = loss / grad_acc_steps
            scaled_loss.backward()  # type: ignore[no-untyped-call]

            if (step + 1) % grad_acc_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if (step + 1) % self.log_interval == 0:
                interval_metrics = {
                    k: v - last_logged_metrics.get(k, 0.0)
                    for k, v in accumulated_metrics.items()
                }
                last_logged_metrics = accumulated_metrics.copy()

                avg_interval_metrics = self._compute_avg_metrics(
                    interval_metrics, self.log_interval
                )

                current_loss = avg_interval_metrics.get("loss", 0.0)
                pbar.set_postfix({"loss": f"{current_loss:.4f}"})
                pbar.update(1)

                current_lr = scheduler.get_last_lr()[0]
                self._log_metrics(
                    wandb_run=wandb_run,
                    metrics=avg_interval_metrics,
                    step=global_step,
                    prefix="train",
                    add_avg_prefix=False,  # Raw interval metrics
                    additional_log_items={"train/learning_rate": current_lr},
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

    def _eval_epoch(
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
        logger.info(f"Model {model_idx} - Evaluating epoch {epoch}/{n_epochs}")
        n_batches = len(eval_loader)
        if n_batches == 0:
            raise ValueError("Evaluation loader is empty")

        model.eval()
        accumulated_metrics: dict[str, float] = {}

        desc = f"Model {model_idx} - Eval E{epoch}/{n_epochs}"
        pbar = tqdm(
            eval_loader,
            desc=desc,
            position=model_idx,
            leave=False,
            total=n_batches,
            mininterval=1.0,
        )

        with torch.no_grad():
            for step, batch in enumerate(pbar):
                loss, metrics = self.model.loss_fn(batch, model, device)

                loss_val = loss.item()
                if "loss" not in metrics:
                    metrics["loss"] = loss_val

                for k, v in metrics.items():
                    accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + v

                if (step + 1) % self.log_interval == 0:
                    running_avg_loss = accumulated_metrics["loss"] / (step + 1)

                    postfix_dict = {"loss": f"{running_avg_loss:.4f}"}

                    pbar.set_postfix(postfix_dict)

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
