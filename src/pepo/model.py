import math
import threading
from typing import Optional

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import DataManager, DeviceManager, HubManager, Logger, WandbHandler

# TODO(adam): possible abstraction here is to define a PEPOSubModel base class that handles the logic of a single model in the ensemble
#             we could inherit this class to define submodels such as smollm, gemma, etc. which would handle their own chat template, tokenizer, etc.
#             this would allow for more modularity and easier to extend to new models.
#             we would define hydra configs for each submodel we support and in the pepo config just specify the submodel with a single word (smollm) which would load the appropriate config and instantiate the submodel.
#             ISSUE: instantiation of L submodels requires a factory/passing config to pepo


class PEPOModel:
    def __init__(
        self,
        alpha: float,
        beta: float,
        num_networks: int,
        model_id: str,
        device_manager: DeviceManager,
        hub_manager: HubManager,
        tokenizer_id: Optional[str] = None,
        chat_template: Optional[str] = None,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_bias: str = "none",
        lora_task_type: str = "CAUSAL_LM",
        lora_target_modules: str = "all-linear",
        logger: Optional[Logger] = None,
    ):
        """
        Initialize PEPO Model.

        Args:
            alpha: Pessimistic margin alpha parameter.
            beta: DPO beta parameter, controls strength of preference.
            num_networks: Number of networks in the ensemble (L).
            model_id: HuggingFace model ID (e.g., "HuggingFaceTB/SmolLM2-1.7B").
            device_manager: Device manager instance.
            hub_manager: Hub manager instance.
            tokenizer_id: HuggingFace tokenizer ID. If None, uses model_id.
            chat_template: Custom chat template string. If None, uses model's built-in template.
            lora_r: LoRA rank parameter.
            lora_alpha: LoRA alpha parameter.
            lora_dropout: LoRA dropout rate.
            lora_bias: LoRA bias setting.
            lora_task_type: LoRA task type.
            lora_target_modules: LoRA target modules.
            logger: Optional logger instance.
        """
        self.alpha = alpha
        self.beta = beta
        self.num_networks = num_networks
        self.model_id = model_id
        self.tokenizer_id = tokenizer_id
        self.chat_template = chat_template
        self.logger = logger
        self.device_manager = device_manager
        self.hub_manager = hub_manager

        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            task_type=lora_task_type,
            target_modules=lora_target_modules,
        )

        self.tokenizer = self._init_tokenizer()
        self.models = self._load_models()

        if self.logger:
            self.logger.info(
                f"PEPOModel initialized with alpha={self.alpha}, beta={self.beta}, L={self.num_networks}"
            )

    def _init_tokenizer(self):
        """
        Initialize tokenizer for the PEPO ensemble.
        Handles chat template configuration from config or uses model's built-in template.
        """
        tokenizer_id = self.tokenizer_id or self.model_id
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

        # if pad token is not set, add a new one
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            if self.logger:
                self.logger.info("Added new pad token <pad> to tokenizer")

        # Handle chat template
        if self.chat_template is not None:
            tokenizer.chat_template = self.chat_template
            if self.logger:
                self.logger.info("Using custom chat template from config")
                self.logger.debug(f"Chat template: {self.chat_template}")
        elif tokenizer.chat_template is not None:
            if self.logger:
                self.logger.info("Using model's built-in chat template")
        else:
            raise ValueError(
                f"Model {tokenizer_id} has no built-in chat_template. Consider providing one in the model config."
            )

        return tokenizer

    def get_tokenizer(self):
        return self.tokenizer

    def _get_model_name(self) -> str:
        """
        Generate model-specific repository name for the ensemble.
        e.g., "model-name-pepo-a0.1-b0.1-L3".
        """
        model_name = self.model_id.rsplit("/", 1)[-1]
        repo_name = f"{model_name}-pepo-a{self.alpha}-b{self.beta}-L{self.num_networks}"
        return repo_name

    def _get_submodel_name(self, model_idx: int) -> str:
        """
        Generate submodel name for a specific ensemble member.
        e.g., "model-name-pepo-a0.1-b0.1-L3-l0".
        """
        model_name = self._get_model_name()
        return f"{model_name}-l{model_idx}"

    def _get_base_model_name(self) -> str:
        """
        Generate base model name for the ensemble.
        e.g., "SmolLM2-1.7B".
        """
        return self.model_id.rsplit("/", 1)[-1]

    def _load_model(self, model_idx: int, load_from_hub: bool) -> AutoModelForCausalLM:
        device_map = self.device_manager.get_device_for_model(model_idx)
        dtype = self.device_manager.dtype

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=device_map,
        )
        base_model.config.use_cache = False
        base_model.resize_token_embeddings(len(self.tokenizer))
        base_model.config.vocab_size = len(self.tokenizer)
        pad_token_id = self.tokenizer.pad_token_id
        embedding_layer = base_model.get_input_embeddings()
        embedding_layer.weight.data[pad_token_id].zero_()

        output_embeddings = base_model.get_output_embeddings()
        output_embeddings.weight.data[pad_token_id].zero_()

        if load_from_hub:
            repo_id = self.hub_manager.get_repo_id(self._get_submodel_name(model_idx))
            model = PeftModel.from_pretrained(base_model, repo_id, is_trainable=True)
        else:
            model = get_peft_model(base_model, self.lora_config)

        if self.logger:
            trainable, total = model.get_nb_trainable_parameters()
            trainable = trainable / 1000000
            total = total / 1000000
            self.logger.info(
                f"Submodel id:{model_idx} on {device_map} has {trainable:.2f}M trainable parameters out of {total:.2f}M total parameters ({trainable/total*100:.2f}%)"
            )
        return model

    def _load_models(self):
        """
        Load all ensemble models from Hub or initialize them from scratch.
        """
        models = []
        load_from_hub = self.hub_manager.should_load_from_hub
        for model_idx in range(self.num_networks):
            if not load_from_hub:
                break
            if not self.hub_manager.model_exists(self._get_submodel_name(model_idx)):
                if self.logger:
                    self.logger.info(
                        f"Submodel {self._get_submodel_name(model_idx)} does not exist on Hub, loading from scratch"
                    )
                load_from_hub = False

        for model_idx in range(self.num_networks):
            models.append(self._load_model(model_idx, load_from_hub))
        return models

    def _push_models(self):
        """
        Push all ensemble models to Hub.
        """
        for model_idx in range(self.num_networks):
            self.hub_manager.push_model(
                model_name=self._get_submodel_name(model_idx),
                model=self.models[model_idx],
                commit_message=f"Upload PEPO ensemble model {model_idx}",
            )

    def _get_log_probs(
        self,
        model: AutoModelForCausalLM,
        device: torch.device,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the log probabilities of the response tokens.
        Creates a mask of the responses and
        """
        # B, T = input_ids.shape
        # V = model.config.vocab_size

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad() if model.training is False else torch.enable_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (B, T , V); shifted by 1 (removed first, added new)

        logits = logits[:, :-1, :]  # (B, T-1, V); remove the new token
        labels = input_ids[:, 1:]  # (B, T-1); remove the first token
        attn_mask_shifted = attention_mask[:, 1:]

        log_probs = F.log_softmax(
            logits, dim=-1
        )  # (B, T-1, V); log prob over vocab dimension
        # select only the log probs for the labels, (B, T-1)
        log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        # mask out the response probs by creating a mask of the response tokens
        pos = torch.arange(log_probs.shape[1], device=device).view(1, -1)  # (1, T-1)
        response_mask = (pos >= (prompt_len - 1).unsqueeze(1)).float()
        response_mask *= attn_mask_shifted.float()
        response_mask = torch.where(
            labels == self.tokenizer.pad_token_id, 0, response_mask
        )
        log_probs = log_probs * response_mask.float()  # mask

        # use debug_tokens from utils.debug to debug the masks and log probs
        # debug_tokens(input_ids, self.tokenizer, attn_mask_shifted, response_mask, log_probs, self.logger)
        log_probs_sum = log_probs.sum(dim=-1)
        return log_probs_sum

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
        model.eval()
        eval_loss = 0.0
        num_eval_batches = 0
        total_eval_batches = len(eval_loader)
        eval_prob_chosen_sum = 0.0
        eval_prob_rejected_sum = 0.0

        with torch.no_grad():
            for eval_step, eval_batch in enumerate(eval_loader):
                eval_batch = {k: v.to(device) for k, v in eval_batch.items()}

                log_probs_chosen = self._get_log_probs(
                    model,
                    device,
                    eval_batch["chosen_input_ids"],
                    eval_batch["chosen_attention_mask"],
                    eval_batch["prompt_len"],
                )
                log_probs_rejected = self._get_log_probs(
                    model,
                    device,
                    eval_batch["rejected_input_ids"],
                    eval_batch["rejected_attention_mask"],
                    eval_batch["prompt_len"],
                )
                with model.disable_adapter():  # type: ignore[operator]
                    log_probs_chosen_ref = self._get_log_probs(
                        model,
                        device,
                        eval_batch["chosen_input_ids"],
                        eval_batch["chosen_attention_mask"],
                        eval_batch["prompt_len"],
                    )
                    log_probs_rejected_ref = self._get_log_probs(
                        model,
                        device,
                        eval_batch["rejected_input_ids"],
                        eval_batch["rejected_attention_mask"],
                        eval_batch["prompt_len"],
                    )

                pi_log_ratio = log_probs_chosen - log_probs_rejected
                ref_log_ratio = log_probs_chosen_ref - log_probs_rejected_ref
                alpha_offset = math.log(1.0 + self.alpha)
                argument = self.beta * (pi_log_ratio - ref_log_ratio - alpha_offset)
                dpo_loss_components = -F.logsigmoid(argument)
                batch_eval_loss = dpo_loss_components.mean().item()
                eval_loss += batch_eval_loss
                num_eval_batches += 1

                prob_chosen = torch.exp(log_probs_chosen).mean().item()
                prob_rejected = torch.exp(log_probs_rejected).mean().item()
                eval_prob_chosen_sum += prob_chosen
                eval_prob_rejected_sum += prob_rejected

                if (
                    self.logger
                    and (eval_step + 1) % max(1, total_eval_batches // 10) == 0
                ):
                    current_avg_loss = eval_loss / num_eval_batches
                    self.logger.info(
                        f"Model {model_idx} - Epoch {epoch}/{n_epochs} - Eval Step {eval_step + 1}/{total_eval_batches} - "
                        f"Current Avg Loss: {current_avg_loss:.4f}"
                    )

                if (
                    wandb_handler is not None
                    and (eval_step + 1) % max(1, total_eval_batches // 10) == 0
                ):
                    current_avg_prob_chosen = eval_prob_chosen_sum / num_eval_batches
                    current_avg_prob_rejected = eval_prob_rejected_sum / num_eval_batches
                    wandb_handler.log(
                        {
                            "eval/prob_chosen": prob_chosen,
                            "eval/prob_rejected": prob_rejected,
                            "eval/avg_prob_chosen": current_avg_prob_chosen,
                            "eval/avg_prob_rejected": current_avg_prob_rejected,
                            "train/epoch": epoch,
                        },
                        step=global_step,
                    )

        avg_eval_loss = eval_loss / num_eval_batches if num_eval_batches > 0 else 0.0
        avg_eval_prob_chosen = (
            eval_prob_chosen_sum / num_eval_batches if num_eval_batches > 0 else 0.0
        )
        avg_eval_prob_rejected = (
            eval_prob_rejected_sum / num_eval_batches if num_eval_batches > 0 else 0.0
        )

        if self.logger:
            self.logger.info(
                f"Model {model_idx} - Epoch {epoch}/{n_epochs} - "
                f"Average Eval Loss: {avg_eval_loss:.4f}"
            )

        if wandb_handler is not None:
            wandb_handler.log(
                {
                    "eval/loss": avg_eval_loss,
                    "eval/avg_prob_chosen": avg_eval_prob_chosen,
                    "eval/avg_prob_rejected": avg_eval_prob_rejected,
                    "train/epoch": epoch,
                },
                step=global_step,
            )

        return avg_eval_loss

    def _train_model(
        self,
        model_idx: int,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        n_epochs: int = 1,
        gradient_accumulation_steps: int = 1,
        wandb_handler: Optional[WandbHandler] = None,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 0.0,
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
            gradient_accumulation_steps: Number of steps to accumulate gradients.
            wandb_handler: Optional wandb handler for logging.
            early_stopping_patience: Number of epochs to wait before stopping if no improvement.
                                     If None, early stopping is disabled.
            early_stopping_min_delta: Minimum change to qualify as an improvement.
        """
        device = torch.device(self.device_manager.get_device_for_model(model_idx))
        model = self.models[model_idx]

        if wandb_handler is not None and wandb_handler.enabled:
            wandb_handler.init_run()

        # Custom initialization for testing - set LoRA weights to small fixed values
        # debug the ref and model log prop computation by setting the LoRA weights to small fixed values
        # initialize_lora_for_testing(model, std=0.1, logger=self.logger)

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
        early_stopping_enabled = early_stopping_patience is not None

        if early_stopping_enabled and self.logger:
            self.logger.info(
                f"Model {model_idx} - Early stopping enabled with patience={early_stopping_patience}, "
                f"min_delta={early_stopping_min_delta}"
            )

        for epoch in range(n_epochs):
            if self.logger:
                self.logger.info(f"Model {model_idx} - Starting training epoch {epoch+1}")

            model.train()
            optimizer.zero_grad()
            epoch_train_loss = 0.0
            num_train_batches = 0
            steps_per_epoch = len(train_loader) // gradient_accumulation_steps
            epoch_prob_chosen_sum = 0.0
            epoch_prob_rejected_sum = 0.0

            for step, batch in enumerate(train_loader):
                batch = {k: v.to(device) for k, v in batch.items()}

                log_probs_chosen = self._get_log_probs(
                    model,
                    device,
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                    batch["prompt_len"],
                )
                log_probs_rejected = self._get_log_probs(
                    model,
                    device,
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"],
                    batch["prompt_len"],
                )
                with torch.no_grad():
                    with model.disable_adapter():  # type: ignore[operator]
                        log_probs_chosen_ref = self._get_log_probs(
                            model,
                            device,
                            batch["chosen_input_ids"],
                            batch["chosen_attention_mask"],
                            batch["prompt_len"],
                        )
                        log_probs_rejected_ref = self._get_log_probs(
                            model,
                            device,
                            batch["rejected_input_ids"],
                            batch["rejected_attention_mask"],
                            batch["prompt_len"],
                        )

                pi_log_ratio = log_probs_chosen - log_probs_rejected
                ref_log_ratio = log_probs_chosen_ref - log_probs_rejected_ref
                alpha_offset = math.log(1.0 + self.alpha)
                argument = self.beta * (pi_log_ratio - ref_log_ratio - alpha_offset)
                dpo_loss_components = -F.logsigmoid(argument)
                loss = dpo_loss_components.mean()

                loss = loss / gradient_accumulation_steps
                loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    loss_value = loss.item() * gradient_accumulation_steps
                    epoch_train_loss += loss_value
                    num_train_batches += 1
                    current_lr = scheduler.get_last_lr()[0]

                    prob_chosen = torch.exp(log_probs_chosen).mean().item()
                    prob_rejected = torch.exp(log_probs_rejected).mean().item()
                    epoch_prob_chosen_sum += prob_chosen
                    epoch_prob_rejected_sum += prob_rejected

                    if self.logger and global_step % max(1, steps_per_epoch // 10) == 0:
                        epoch_string_length = len(str(n_epochs))
                        step_string_length = len(str(steps_per_epoch))
                        self.logger.info(
                            f"Model {model_idx} - Epoch {epoch+1:>{epoch_string_length}}/{n_epochs} - Train Step {global_step:>{step_string_length}}/{steps_per_epoch} - "
                            f"Loss: {loss_value:.4f} - "
                            f"Avg_loss: {epoch_train_loss / num_train_batches:.4f}"
                        )

                    if wandb_handler is not None:
                        wandb_handler.log(
                            {
                                "train/loss": loss_value,
                                "train/learning_rate": current_lr,
                                "train/epoch": epoch + 1,
                                "train/step": global_step,
                                "train/prob_chosen": prob_chosen,
                                "train/prob_rejected": prob_rejected,
                            },
                            step=global_step,
                        )

            avg_train_loss = (
                epoch_train_loss / num_train_batches if num_train_batches > 0 else 0.0
            )
            avg_prob_chosen = (
                epoch_prob_chosen_sum / num_train_batches
                if num_train_batches > 0
                else 0.0
            )
            avg_prob_rejected = (
                epoch_prob_rejected_sum / num_train_batches
                if num_train_batches > 0
                else 0.0
            )

            if self.logger:
                self.logger.info(
                    f"Model {model_idx} - Epoch {epoch+1}/{n_epochs} - "
                    f"Average Train Loss: {avg_train_loss:.4f}"
                )

            if wandb_handler is not None:
                wandb_handler.log(
                    {
                        "train/avg_loss": avg_train_loss,
                        "train/avg_prob_chosen": avg_prob_chosen,
                        "train/avg_prob_rejected": avg_prob_rejected,
                        "train/epoch": epoch + 1,
                    },
                    step=global_step,
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

            if early_stopping_enabled:
                if eval_loss < best_eval_loss - early_stopping_min_delta:
                    best_eval_loss = eval_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    if self.logger:
                        self.logger.info(
                            f"Model {model_idx} - Early stopping triggered after {epoch + 1} epochs. "
                            f"Best validation loss: {best_eval_loss:.4f}"
                        )
                    break

    def train(
        self,
        data_manager: DataManager,
        optimizers: list[torch.optim.Optimizer],
        schedulers: list[torch.optim.lr_scheduler._LRScheduler],
        batch_size: int,
        gradient_accumulation_steps: int = 1,
        max_epochs: int = 1,
        wandb_handlers: Optional[list[WandbHandler]] = None,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 0.0,
    ):
        """
        Train the PEPO ensemble models and save the models to the hub.
        Uses threading to run models in parallel on different GPUs.

        Args:
            data_manager: DataManager instance for getting dataloaders.
            optimizers: List of optimizers, one per model in the ensemble.
            schedulers: List of schedulers, one per model in the ensemble.
            batch_size: Batch size for training.
            gradient_accumulation_steps: Number of steps to accumulate gradients.
            max_epochs: Maximum number of training epochs.
            wandb_handlers: Optional list of wandb handlers, one per model.
            early_stopping_patience: Number of epochs to wait before stopping if no improvement.
                                     If None, early stopping is disabled.
            early_stopping_min_delta: Minimum change to qualify as an improvement.
        """
        if self.logger:
            self.logger.info("Training PEPO ensemble models...")

        if len(optimizers) != self.num_networks or len(schedulers) != self.num_networks:
            raise ValueError(
                f"Number of optimizers ({len(optimizers)}) and schedulers ({len(schedulers)}) "
                f"must match number of networks ({self.num_networks})"
            )

        if wandb_handlers is not None and len(wandb_handlers) != self.num_networks:
            raise ValueError(
                f"Number of wandb handlers ({len(wandb_handlers)}) "
                f"must match number of networks ({self.num_networks})"
            )

        threads = []
        for model_idx in range(self.num_networks):
            train_loader = data_manager.get_dataloader(
                model_idx=model_idx,
                partition="train",
                batch_size=batch_size,
            )
            eval_loader = data_manager.get_dataloader(
                model_idx=model_idx,
                partition="eval",
                batch_size=batch_size,
            )

            thread = threading.Thread(
                target=self._train_model,
                args=(
                    model_idx,
                    train_loader,
                    eval_loader,
                    optimizers[model_idx],
                    schedulers[model_idx],
                    max_epochs,
                    gradient_accumulation_steps,
                    wandb_handlers[model_idx] if wandb_handlers is not None else None,
                    early_stopping_patience,
                    early_stopping_min_delta,
                ),
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        if wandb_handlers is not None:
            for wandb_handler in wandb_handlers:
                wandb_handler.finish()

        if self.hub_manager.should_push_to_hub:
            self._push_models()

    def _predict(
        self,
        model: AutoModelForCausalLM,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, T, V)
        last_logits = logits[:, -1, :]
        log_probs = F.log_softmax(last_logits, dim=-1)
        return log_probs

    def predict(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
        if len(input_ids.shape) != 2:
            raise ValueError("input_ids must be a 2D tensor")

        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).float()

        log_probs_ensemble = []

        def predict_log_probs(model, input_ids, attention_mask):
            with torch.no_grad():
                model.eval()
                log_probs = self._predict(model, input_ids, attention_mask)
                log_probs_ensemble.append(log_probs.cpu())

        threads = []
        for model_idx in range(self.num_networks):
            thread = threading.Thread(
                target=predict_log_probs,
                args=(
                    self.models[model_idx],
                    input_ids,
                    attention_mask,
                ),
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        log_probs_tensor: torch.Tensor = torch.stack(
            log_probs_ensemble, dim=0
        )  # (L, B, V)

        # eos_probs = log_probs_tensor[:, 0, self.tokenizer.eos_token_id]
        # eos_probs = torch.exp(eos_probs)
        # print(f"\nEOS token prob: min={eos_probs.min().item():.2f}, max={eos_probs.max().item():.2f}, mean={eos_probs.mean().item():.2f}, std={eos_probs.std().item():.2f}")
        min_log_probs, _ = torch.min(log_probs_tensor, dim=0)
        return min_log_probs

    def predict_base_model(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
        if len(input_ids.shape) != 2:
            raise ValueError("input_ids must be a 2D tensor")

        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).float()

        model = self.models[0]
        # disable adapter
        with torch.no_grad():
            with model.disable_adapter():
                model.eval()
                log_probs = self._predict(model, input_ids, attention_mask)  # (B, V)
        # eos_probs = log_probs[0, self.tokenizer.eos_token_id]
        # eos_probs = torch.exp(eos_probs)
        # print(f"EOS token prob: min={eos_probs.min().item():.2f}, max={eos_probs.max().item():.2f}, mean={eos_probs.mean().item():.2f}, std={eos_probs.std().item():.2f}")
        return log_probs

    def generate(
        self,
        prompts: list[str],
        max_length: int = 1024,
        use_ensamble: bool = True,
        apply_chat_template: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.logger:
            self.logger.info(
                f"Generating ensemble={use_ensamble}, template={apply_chat_template}"
            )

        if apply_chat_template:
            formated_prompts = []
            for prompt in prompts:
                formatted_prompt = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": ""},
                ]
                formated_prompt = self.tokenizer.apply_chat_template(
                    formatted_prompt, tokenize=False, add_generation_prompt=True
                )
                formated_prompts.append(formated_prompt)
        else:
            formated_prompts = prompts

        prev_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        inputs = self.tokenizer(
            formated_prompts,
            return_tensors="pt",
            padding=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        stop_signal = torch.zeros(input_ids.shape[0], dtype=torch.bool)

        for i in range(max_length):
            if use_ensamble:
                log_probs = self.predict(input_ids, attention_mask)
            else:
                log_probs = self.predict_base_model(input_ids, attention_mask)
            min_probs = torch.exp(log_probs)

            missing_token_id = len(self.tokenizer)
            missing_probs = torch.clamp(1 - torch.sum(min_probs, dim=-1), min=0.0)
            min_probs = torch.cat([min_probs, missing_probs.unsqueeze(-1)], dim=-1)

            # TODO(adam): handle top k sampling, temperature sampling, etc.
            # TODO(adam): handle resampling max attempts

            # resample where we got missing token until we get a non-missing token
            missing_mask = torch.ones(input_ids.shape[0], dtype=torch.bool)
            sampled_token_ids = torch.zeros(input_ids.shape[0], dtype=torch.long)
            while True:
                new_sampled_token_ids = torch.multinomial(
                    min_probs[missing_mask], num_samples=1
                ).squeeze(-1)
                sampled_token_ids[missing_mask] = new_sampled_token_ids
                missing_mask = sampled_token_ids == missing_token_id
                if not torch.any(missing_mask):
                    break

            input_ids = torch.cat([input_ids, sampled_token_ids.unsqueeze(-1)], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(sampled_token_ids).unsqueeze(-1)], dim=1
            )
            stop_signal = stop_signal | (sampled_token_ids == self.tokenizer.eos_token_id)

            if sampled_token_ids[0] == self.tokenizer.eos_token_id:
                if self.logger:
                    self.logger.debug(f"Generated EOS token at step {i}")
            if torch.all(stop_signal):
                break
        if self.logger:
            self.logger.debug(
                f"Generated sequence idx=0:\n{self.tokenizer.decode(input_ids[0], skip_special_tokens=True)}"
            )

        self.tokenizer.padding_side = prev_padding_side
        return input_ids, attention_mask

    def generate_base_model(
        self, prompts: list[str], max_length: int = 1024, apply_chat_template: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.generate(
            prompts,
            max_length,
            use_ensamble=False,
            apply_chat_template=apply_chat_template,
        )
