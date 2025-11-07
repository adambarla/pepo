import os
from typing import Optional

import dotenv
from huggingface_hub import HfApi, login
from omegaconf import DictConfig

from .logger import Logger

dotenv.load_dotenv()


class HubManager:
    """
    Generic HuggingFace Hub manager.
    No domain-specific knowledge - just Hub operations.
    """

    def __init__(
        self,
        config: DictConfig,
        logger: Optional[Logger] = None,
        hf_token: Optional[str] = None,
    ):
        """
        Initialize HubManager.

        Args:
            base_dir: HuggingFace username/organization (e.g., "PessimisticDPO").
            logger: Optional logger instance.
            hf_token: HuggingFace token. If None, uses HF_TOKEN env var or cached token.
        """
        self.base_dir = config.base_dir
        self.load_from_hub = config.load_from_hub
        self.push_to_hub = config.push_to_hub
        self.logger = logger
        self.api = HfApi()
        self._authenticate(hf_token)

    @property
    def should_push_to_hub(self) -> bool:
        return self.push_to_hub

    @property
    def should_load_from_hub(self) -> bool:
        return self.load_from_hub

    def _authenticate(self, hf_token: Optional[str] = None):
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
                        f"Set HF_TOKEN environment variable or run 'huggingface-cli login'"
                    )

    def model_exists(self, model_name: str) -> bool:
        """
        Check if a model repository exists on HuggingFace Hub.

        Args:
            repo_id: Full repository ID (e.g., "username/repo-name").

        Returns:
            True if repository exists, False otherwise.
        """
        repo_id = f"{self.base_dir}/{model_name}"
        try:
            self.api.model_info(repo_id)
            return True
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Model {model_name} not found in hub: {e}")
            return False

    def get_repo_id(self, model_name: str) -> str:
        """
        Get the full repository ID for a model name.

        Args:
            model_name: Model name (without base_dir).

        Returns:
            Full repository ID (e.g., "username/repo-name").
        """
        return f"{self.base_dir}/{model_name}"

    def push_model(
        self,
        model_name: str,
        model,
        commit_message: Optional[str] = None,
        private: bool = True,
    ):
        """
        Push model and tokenizer to HuggingFace Hub.

        Args:
            repo_id: Full repository ID (e.g., "username/repo-name").
            model: The model to push (PEFT model with LoRA adapters).
            tokenizer: The tokenizer to push.
            commit_message: Custom commit message. If None, auto-generates.
            private: Whether to make the repository private.
        """
        if not self.should_push_to_hub:
            if self.logger:
                self.logger.warning(
                    f"Skipping push of model {model_name} to Hub because push_to_hub is disabled in config."
                )
            return

        repo_id = f"{self.base_dir}/{model_name}"
        if commit_message is None:
            commit_message = f"Upload model to {repo_id}"

        if self.logger:
            self.logger.info(f"Pushing model to {repo_id}...")

        # Push model
        model.push_to_hub(
            repo_id,
            commit_message=commit_message,
            private=private,
        )

        if self.logger:
            self.logger.info(
                f"Model successfully pushed to: https://huggingface.co/{repo_id}"
            )
