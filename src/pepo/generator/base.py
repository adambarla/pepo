"""Abstract base class for generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from ..model import BaseModel


class BaseGenerator(ABC):
    """Abstract base class defining the generator interface."""

    @abstractmethod
    def generate_responses(
        self,
        model: "BaseModel",
        prompts: list[Any],
        apply_chat_template: bool = True,
        token_callback: Optional[Callable[[str], None]] = None,
    ) -> list[dict[str, Any]]:
        """Generate responses for a list of prompts.

        Args:
            model: BaseModel instance.
            prompts: List of prompts.
            apply_chat_template: Whether to apply chat template.
            token_callback: Optional callback for streaming tokens.

        Returns:
            List of dicts with 'prompt' and 'output' keys.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get generator name for file naming."""
        pass
