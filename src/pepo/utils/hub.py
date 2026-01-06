import logging
import os
from typing import TYPE_CHECKING, Optional

import dotenv
from huggingface_hub import HfApi, login
from peft import PeftModel
from transformers import AutoModelForCausalLM, PreTrainedModel

if TYPE_CHECKING:
    from transformers import AutoTokenizer

dotenv.load_dotenv()

logger = logging.getLogger(__name__)


class HubManager:
    """
    Generic HuggingFace Hub manager.
    No domain-specific knowledge - just Hub operations.
    """

    def __init__(
        self,
        base_dir: str,
        push: bool = True,
        load_trainable: bool = True,
        hf_token: Optional[str] = None,
    ):
        """
        Initialize HubManager.

        Args:
            base_dir: HuggingFace username/organization (e.g., "PessimisticDPO").
            push: Whether to push models to hub.
            load_trainable: Whether to load models with trainable parameters.
            hf_token: HuggingFace token. If None, uses HF_TOKEN env var or cached token.
        """
        self.base_dir = base_dir
        self.should_push = push
        self.load_trainable = load_trainable
        self.api = HfApi()
        self._authenticate(hf_token)

    @property
    def should_push_to_hub(self) -> bool:
        return self.should_push

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
                logger.warning(
                    f"Could not login to HuggingFace: {e}. "
                    f"Set HF_TOKEN environment variable or run "
                    f"'huggingface-cli login'"
                )

    def model_exists(self, model_name: str, epoch: Optional[int] = None) -> bool:
        repo_id = self.get_repo_id(model_name, epoch)
        try:
            self.api.model_info(repo_id)
            return True
        except Exception:
            logger.warning(f"Failed to load {repo_id}")
            return False

    def get_repo_id(self, model_name: str, epoch: Optional[int] = None) -> str:
        if epoch is not None:
            return f"{self.base_dir}/{model_name}-e{epoch}"
        else:
            return f"{self.base_dir}/{model_name}"

    def load_model(
        self,
        base_model: PreTrainedModel,
        model_name: str,
        epoch: Optional[int] = None,
    ) -> PeftModel:
        if not self.model_exists(model_name, epoch):
            logger.error(
                f"Tried to load model non existing {model_name} (epoch {epoch})."
            )
            raise ValueError(f"Model {model_name} (epoch {epoch}) not found in hub")
        repo_id = self.get_repo_id(model_name, epoch)
        model = PeftModel.from_pretrained(
            base_model,
            repo_id,
            is_trainable=self.load_trainable,
        )

        if self.load_trainable:
            trainable, total = model.get_nb_trainable_parameters()
            trainable = int(trainable / 1000000)
            total = int(total / 1000000)
            logger.info(
                f"Loaded model from {repo_id} with {trainable:.2f}M trainable "
                f"parameters out of {total:.2f}M total parameters "
                f"({trainable / total * 100:.2f}%)"
            )
        else:
            logger.info(
                f"Loaded model from {repo_id} without trainable parameters "
                f"(set hub.load_trainable=true to load trainable parameters)"
            )
        return model

    def find_latest_epoch_for_all_submodels(
        self, base_model_name: str, num_networks: int, max_epoch: int
    ) -> Optional[int]:
        """
        Find the latest epoch where all submodels have checkpoints.
        Checks backwards from max_epoch until finding an epoch where all
        submodels exist.

        Args:
            base_model_name: Base model name (without submodel suffix).
            num_networks: Number of submodels in the ensemble.
            max_epoch: Maximum epoch to check from. Checks backwards from here.

        Returns:
            The latest epoch where all submodels exist, or None if no common
            epoch exists.
        """
        for epoch in range(max_epoch, -1, -1):
            all_exist = True
            for model_idx in range(num_networks):
                submodel_name = f"{base_model_name}-l{model_idx}"
                if not self.model_exists(submodel_name, epoch):
                    all_exist = False
                    break

            if all_exist:
                logger.info(
                    f"Found latest common epoch {epoch} for all "
                    f"{num_networks} submodels"
                )
                return epoch

        logger.warning(
            f"No common epoch found across all {num_networks} submodels "
            f"(checked from epoch {max_epoch} down to 0)"
        )
        return None

    def push_model(
        self,
        model_name: str,
        model: AutoModelForCausalLM,
        tokenizer: "AutoTokenizer",
        private: bool = False,
        epoch: Optional[int] = None,
    ) -> None:
        """
        Push model and tokenizer to HuggingFace Hub.

        Args:
            model_name: Base model name (without epoch suffix).
            model: The model to push (PEFT model with LoRA adapters).
            model_idx: Index of the model in the ensemble.
            private: Whether to make the repository private.
            epoch: Optional number of epochs. If provided, appends "-e{epoch}"
                to model_name. Use None for final push without epoch suffix.
        """
        if epoch is not None:
            model_name = f"{model_name}-e{epoch}"  # append epoch suffix if provided
        repo_id = f"{self.base_dir}/{model_name}"

        if not self.should_push_to_hub:
            logger.warning(f"Disabled push to Hub. Skipping push of {repo_id}.")
            return

        # Generate commit message based on epoch
        if epoch is not None:
            commit_message = (
                f"Upload PEPO model checkpoint after {epoch} epochs to {repo_id}"
            )
        else:
            commit_message = f"Upload final PEPO model to {repo_id}"

        logger.info(f"Pushing model to {repo_id}...")

        # Push model
        model.push_to_hub(
            repo_id,
            commit_message=commit_message,
            private=private,
        )
        tokenizer.push_to_hub(
            repo_id,
            commit_message=commit_message,
            private=private,
        )

        logger.info(
            f"Model and tokenizer successfully pushed to: https://huggingface.co/{repo_id}"
        )
