from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class JudgePrompt:
    system_prompt: str
    user_prompt: str


class BaseJudge(ABC):
    @abstractmethod
    def generate(self, prompts: Sequence[JudgePrompt]) -> list[str]:
        """Generate judge completions for formatted judge prompts."""

    def unload(self) -> None:
        """Release judge resources. Implementations may override this."""
