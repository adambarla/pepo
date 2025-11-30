import threading
from datetime import datetime
from typing import Optional

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import get_scheduler

from .utils import DataManager, Logger, WandbHandler


class Trainer:
    """Trainer class for PEPO ensemble models."""

    def __init__(
        self,
        model,
        data_manager: DataManager,
        optimizer_config: DictConfig,
        scheduler_config: DictConfig,
        wandb_config: DictConfig,
        batch_size: int,
        eval_batch_size: Optional[int] = None,
        gradient_accumulation_steps: int = 1,
        max_epochs: int = 1,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 0.0,
        resolved_cfg_plain: Optional[dict] = None,
        logger: Optional[Logger] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: PEPOModel instance to train.
            data_manager: DataManager instance for getting dataloaders.
            optimizer_config: Hydra config for optimizer instantiation.
            scheduler_config: Hydra config for scheduler (name, num_warmup_steps).
            wandb_config: Hydra config for wandb settings.
            batch_size: Batch size for training.
            eval_batch_size: Batch size for evaluation. If None, uses 4 * batch_size.
            gradient_accumulation_steps: Number of steps to accumulate gradients.
            max_epochs: Maximum number of training epochs.
            early_stopping_patience: Number of epochs to wait before stopping if no improvement.
                                     If None, early stopping is disabled.
            early_stopping_min_delta: Minimum change to qualify as an improvement.
            resolved_cfg_plain: Resolved config dict for wandb logging.
            logger: Optional logger instance.
        """
        from hydra.utils import instantiate

        self.model = model
        self.data_manager = data_manager
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.logger = logger

        if wandb_config.enabled and resolved_cfg_plain is not None:
            dataset_path = data_manager._get_cache_path()
            if dataset_path is not None:
                dataset_hash = dataset_path.split("/")[-1]  # type: ignore[union-attr]
                resolved_cfg_plain["dataset"]["hash"] = dataset_hash

        self.optimizers = []
        self.schedulers = []
        self.wandb_handlers = None

        for model_idx in range(model.num_networks):
            model_params = model.models[model_idx].parameters()

            optimizer = instantiate(
                optimizer_config,
                params=model_params,
            )
            self.optimizers.append(optimizer)

            train_loader = data_manager.get_dataloader(
                model_idx=model_idx,
                partition="train",
                batch_size=batch_size,
            )
            num_training_steps = (
                len(train_loader) // gradient_accumulation_steps
            ) * max_epochs

            scheduler = get_scheduler(
                name=scheduler_config.name,
                optimizer=optimizer,
                num_warmup_steps=scheduler_config.num_warmup_steps,
                num_training_steps=num_training_steps,
            )
            self.schedulers.append(scheduler)

            if logger:
                logger.info(
                    f"Created optimizer and scheduler for model {model_idx}. "
                    f"Training steps: {num_training_steps}"
                )

        if wandb_config.enabled:
            self.wandb_handlers = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for model_idx in range(model.num_networks):
                handler = WandbHandler(
                    enabled=True,
                    project=wandb_config.project,
                    name=model._get_submodel_name(model_idx),
                    tags=list(wandb_config.tags) + [model._get_base_model_name()],
                    notes=wandb_config.notes,
                    entity=wandb_config.entity,
                    mode=wandb_config.mode,
                    cfg=resolved_cfg_plain,
                    group=f"{model._get_model_name()}-{timestamp}",
                    logger=logger,
                    _lazy_init=True,
                )
                self.wandb_handlers.append(handler)

    def train(self):
        """
        Train the PEPO ensemble models and save the models to the hub.
        Uses threading to run models in parallel on different GPUs.
        """
        if self.logger:
            self.logger.info("Training PEPO ensemble models...")

        threads = []
        for model_idx in range(self.model.num_networks):
            train_loader = self.data_manager.get_dataloader(
                model_idx=model_idx,
                partition="train",
                batch_size=self.batch_size,
            )
            eval_bs = (
                self.eval_batch_size
                if self.eval_batch_size is not None
                else 4 * self.batch_size
            )
            eval_loader = self.data_manager.get_dataloader(
                model_idx=model_idx,
                partition="eval",
                batch_size=eval_bs,
            )

            thread = threading.Thread(
                target=self._train_model,
                args=(
                    model_idx,
                    train_loader,
                    eval_loader,
                    self.optimizers[model_idx],
                    self.schedulers[model_idx],
                    self.max_epochs,
                    self.gradient_accumulation_steps,
                    (
                        self.wandb_handlers[model_idx]
                        if self.wandb_handlers is not None
                        else None
                    ),
                    self.early_stopping_patience,
                    self.early_stopping_min_delta,
                ),
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        if self.wandb_handlers is not None:
            for wandb_handler in self.wandb_handlers:
                wandb_handler.finish()

        self.model._push_models()

    def _train_model(
        self,
        model_idx: int,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        n_epochs: int,
        grad_acc_steps: int,
        wandb_handler: Optional[WandbHandler],
        es_patience: Optional[int],
        es_min_delta: float,
    ):
        """
        Train a single model in a thread. Each thread sets its CUDA device context
        to ensure proper GPU isolation.

        Args:
            model_idx: Index of the model in the ensemble.
            train_loader: DataLoader for training data.
            eval_loader: DataLoader for evaluation data.
            optimizer: Optimizer for training.
            scheduler: Learning rate scheduler.
            n_epochs: Maximum number of training epochs.
            grad_acc_steps: Number of steps to accumulate gradients.
            wandb_handler: Optional wandb handler for logging.
            es_patience: Number of epochs to wait before stopping if no improvement.
            es_min_delta: Minimum change to qualify as an improvement.
        """
        device = torch.device(self.model.device_manager.get_device_for_model(model_idx))
        model = self.model.models[model_idx]

        if wandb_handler is not None and wandb_handler.enabled:
            wandb_handler.init_run()

        if self.logger:
            train_size = len(train_loader.dataset)  # type: ignore[arg-type]
            eval_size = len(eval_loader.dataset)  # type: ignore[arg-type]
            self.logger.info(
                f"Model {model_idx} - Train: size={train_size}, batches={len(train_loader)} - Eval: size={eval_size}, batches={len(eval_loader)}"
            )

        global_step = 0

        if self.logger:
            self.logger.info(f"Model {model_idx} - Running initial evaluation...")

        initial_eval_loss = self._eval_model(
            model_idx=model_idx,
            model=model,
            eval_loader=eval_loader,
            device=device,
            epoch=0,
            n_epochs=n_epochs,
            global_step=global_step,
            wandb_handler=wandb_handler,
        )

        best_eval_loss = initial_eval_loss
        patience_counter = 0
        es_enabled = es_patience is not None

        n_batches = len(train_loader)
        n_ebatches = n_batches // grad_acc_steps
        for epoch in range(n_epochs):
            if self.logger:
                self.logger.info(f"Model {model_idx} - Starting training epoch {epoch+1}")

            model.train()
            optimizer.zero_grad()
            loss = 0.0
            lprob_chosen_sum = 0.0
            lprob_reject_sum = 0.0
            margin_sum = 0.0
            ebatch = 0

            for step, batch in enumerate(train_loader):
                if n_batches - step < grad_acc_steps:
                    if self.logger:
                        self.logger.info(
                            f"Model {model_idx} - Epoch {epoch+1} - "
                            f"Not enough batches to accumulate gradients, skipping remaining {n_batches - step} batches out of {n_batches}"
                        )
                    break
                batch_loss, lprobs_ch, lprobs_re = self.model._loss_fn(
                    batch, model, device
                )

                loss += batch_loss.item()
                lprob_chosen_sum += lprobs_ch.mean().item()
                lprob_reject_sum += lprobs_re.mean().item()
                margin_sum += (lprobs_ch - lprobs_re).mean().item()

                batch_loss = batch_loss / grad_acc_steps
                batch_loss.backward()

                if (step + 1) % grad_acc_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    ebatch += 1

                    current_lr = scheduler.get_last_lr()[0]

                    if self.logger and ebatch % max(1, n_ebatches // 100) == 0:
                        e_str_len = len(str(n_epochs))
                        eb_str_len = len(str(n_ebatches))
                        self.logger.info(
                            f"Model {model_idx} - Train Epoch {epoch+1:>{e_str_len}}/{n_epochs} - Step {ebatch:>{eb_str_len}}/{n_ebatches} - "
                            f"Avg Loss: {loss / ebatch:.4f} - Avg Margin: {margin_sum / ebatch:.4f}"
                        )

                    if wandb_handler is not None:
                        wandb_handler.log(
                            {
                                "train/learning_rate": current_lr,
                                "train/step": global_step,
                                "train/curr_avg_loss": loss / ebatch,
                                "train/curr_avg_margin": margin_sum / ebatch,
                            },
                            step=global_step,
                        )

            if wandb_handler is not None:
                wandb_handler.log(
                    {
                        "train/avg_lprobs_chosen": lprob_chosen_sum / ebatch,
                        "train/avg_lprobs_reject": lprob_reject_sum / ebatch,
                        "train/avg_margin": margin_sum / ebatch,
                        "train/epoch": epoch + 1,
                    },
                    step=global_step,
                )

            self.model.epochs_per_network[model_idx] += 1
            self.model._push_model(
                model_idx, epochs=int(self.model.epochs_per_network[model_idx])
            )

            eval_loss = self._eval_model(
                model_idx=model_idx,
                model=model,
                eval_loader=eval_loader,
                device=device,
                epoch=epoch + 1,
                n_epochs=n_epochs,
                global_step=global_step,
                wandb_handler=wandb_handler,
            )

            if es_enabled and es_patience is not None:
                if eval_loss < best_eval_loss - es_min_delta:
                    best_eval_loss = eval_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                assert es_patience is not None
                if patience_counter >= es_patience:
                    if self.logger:
                        self.logger.info(
                            f"Model {model_idx} - Early stopping triggered after {epoch + 1} epochs. "
                            f"Best validation loss: {best_eval_loss:.4f}"
                        )
                    break

    def _eval_model(
        self,
        model_idx: int,
        model: torch.nn.Module,
        eval_loader: DataLoader,
        device: torch.device,
        epoch: int,
        n_epochs: int,
        global_step: int,
        wandb_handler: Optional[WandbHandler] = None,
    ) -> float:
        """
        Evaluate the model on the evaluation dataset.

        Args:
            model_idx: Index of the model in the ensemble.
            model: The model to evaluate.
            eval_loader: DataLoader for evaluation data.
            device: Device to run evaluation on.
            epoch: Current epoch number.
            n_epochs: Total number of epochs.
            global_step: Current global training step.
            wandb_handler: Optional wandb handler for logging.

        Returns:
            Average evaluation loss.
        """
        n_batches = len(eval_loader)
        if n_batches == 0:
            raise ValueError("Evaluation loader is empty")
        model.eval()
        loss = 0.0
        b = 0
        lprob_chosen_sum = 0.0
        lprob_reject_sum = 0.0
        margin_sum = 0.0

        with torch.no_grad():
            for batch in eval_loader:
                batch_loss, lprobs_ch, lprobs_re = self.model._loss_fn(
                    batch, model, device
                )

                loss += batch_loss.item()
                b += 1

                lprob_chosen_sum += lprobs_ch.mean().item()
                lprob_reject_sum += lprobs_re.mean().item()
                margin_sum += (lprobs_ch - lprobs_re).mean().item()

                if self.logger and b % max(1, n_batches // 10) == 0:
                    current_avg_loss = loss / b
                    e_str_len = len(str(n_epochs))
                    b_str_len = len(str(n_batches))
                    self.logger.info(
                        f"Model {model_idx} - Eval. Epoch {epoch:>{e_str_len}}/{n_epochs} - Step {b:>{b_str_len}}/{n_batches} - "
                        f"Avg Loss: {current_avg_loss:.4f} - "
                        f"Avg Margin: {margin_sum / b:.4f}"
                    )

        if wandb_handler is not None:
            wandb_handler.log(
                {
                    "eval/loss": loss / b,
                    "eval/avg_lprobs_chosen": lprob_chosen_sum / b,
                    "eval/avg_lprobs_reject": lprob_reject_sum / b,
                    "eval/avg_margin": margin_sum / b,
                    "eval/epoch": epoch,
                },
                step=global_step,
            )

        return loss / b
