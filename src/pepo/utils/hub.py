import os
from typing import TYPE_CHECKING, Optional

import dotenv
from huggingface_hub import HfApi, login
from peft import PeftModel
from transformers import AutoModelForCausalLM

if TYPE_CHECKING:
    from transformers import AutoTokenizer

from .logger import Logger

dotenv.load_dotenv()


class HubManager:
    """
    Generic HuggingFace Hub manager.
    No domain-specific knowledge - just Hub operations.
    """

    def __init__(
        self,
        base_dir: str,
        load: bool = False,
        push: bool = True,
        load_epochs: Optional[int] = None,
        load_trainable: bool = True,
        logger: Optional[Logger] = None,
        hf_token: Optional[str] = None,
    ):
        """
        Initialize HubManager.

        Args:
            base_dir: HuggingFace username/organization (e.g., "PessimisticDPO").
            load: Whether to load models from hub.
            push: Whether to push models to hub.
            load_epochs: Optional number of epochs to load.
                If None, loads the final model.
            logger: Optional logger instance.
            hf_token: HuggingFace token. If None, uses HF_TOKEN env var or cached token.
        """
        self.base_dir = base_dir
        self.should_load = load
        self.should_push = push
        self.load_epochs = load_epochs
        self.load_trainable = load_trainable
        self.logger = logger
        self.api = HfApi()
        self._authenticate(hf_token)

    @property
    def should_push_to_hub(self) -> bool:
        return self.should_push

    @property
    def should_load_from_hub(self) -> bool:
        return self.should_load

    def _authenticate(self, hf_token: Optional[str] = None) -> None:
        """Authenticate with HuggingFace Hub."""
        if hf_token:
            login(token=hf_token)
        elif os.getenv("HF_TOKEN"):
            login(token=os.getenv("HF_TOKEN"))
        else:
            try:
                login()  # Uses cached token
            except Exception as e:
                if self.logger:
                    self.logger.warning(
                        f"Could not login to HuggingFace: {e}. "
                        f"Set HF_TOKEN environment variable or run "
                        f"'huggingface-cli login'"
                    )

    def model_exists(self, model_name: str, epochs: Optional[int] = None) -> bool:
        repo_id = self.get_repo_id(model_name, epochs)
        try:
            self.api.model_info(repo_id)
            return True
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Model {model_name} not found in hub: {e}")
            return False

    def get_repo_id(self, model_name: str, epochs: Optional[int] = None) -> str:
        if epochs is not None:
            return f"{self.base_dir}/{model_name}-e{epochs}"
        else:
            return f"{self.base_dir}/{model_name}"

    def load_model(
        self, base_model: AutoModelForCausalLM, model_name: str
    ) -> AutoModelForCausalLM:
        if not self.model_exists(model_name, self.load_epochs):
            raise ValueError(f"Model {model_name} not found in hub")
        repo_id = self.get_repo_id(model_name, self.load_epochs)
        model = PeftModel.from_pretrained(
            base_model,  # type: ignore[arg-type]
            repo_id,
            is_trainable=self.load_trainable,
        )

        if self.logger and self.load_trainable:
            trainable, total = model.get_nb_trainable_parameters()
            trainable = int(trainable / 1000000)
            total = int(total / 1000000)
            self.logger.info(
                f"Loaded model from {repo_id} with {trainable:.2f}M trainable "
                f"parameters out of {total:.2f}M total parameters "
                f"({trainable / total * 100:.2f}%)"
            )
        elif self.logger:
            self.logger.info(
                f"Loaded model from {repo_id} without trainable parameters "
                f"(set hub.load_trainable=true to load trainable parameters)"
            )
        return model  # type: ignore[return-value]

    def push_model(
        self,
        model_name: str,
        model: AutoModelForCausalLM,
        tokenizer: "AutoTokenizer",
        model_idx: int,
        private: bool = False,
        epochs: Optional[int] = None,
    ) -> None:
        """
        Push model and tokenizer to HuggingFace Hub.

        Args:
            model_name: Base model name (without epoch suffix).
            model: The model to push (PEFT model with LoRA adapters).
            model_idx: Index of the model in the ensemble.
            private: Whether to make the repository private.
            epochs: Optional number of epochs. If provided, appends "-e{epochs}"
                to model_name. Use None for final push without epoch suffix.
        """
        if not self.should_push_to_hub:
            if self.logger:
                self.logger.warning(
                    f"Skipping push of model {model_name} to Hub because "
                    f"push_to_hub is disabled in config."
                )
            return

        # Append epoch suffix if provided
        if epochs is not None:
            model_name = f"{model_name}-e{epochs}"

        repo_id = f"{self.base_dir}/{model_name}"

        # Generate commit message based on epochs
        if epochs is not None:
            commit_message = (
                f"Upload PEPO ensemble model {model_idx} checkpoint after "
                f"{epochs} epochs to {repo_id}"
            )
        else:
            commit_message = (
                f"Upload final PEPO ensemble model {model_idx} to {repo_id}"
            )

        if self.logger:
            self.logger.info(f"Pushing model to {repo_id}...")

        # Push model
        model.push_to_hub(  # type: ignore[attr-defined]
            repo_id,
            commit_message=commit_message,
            private=private,
        )
        tokenizer.push_to_hub(  # type: ignore[attr-defined]
            repo_id,
            commit_message=commit_message,
            private=private,
        )

        if self.logger:
            self.logger.info(
                f"Model and tokenizer successfully pushed to: https://huggingface.co/{repo_id}"
            )
