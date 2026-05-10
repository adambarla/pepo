"""Pinned MT-Bench assets and loaders (FastChat-compatible)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any, Final, cast

# Mirrors fastchat.llm_judge.common.temperature_config (gen_model_answer defaults).
FASTCHAT_MT_BENCH_TEMPERATURE: Final[dict[str, float]] = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
    "arena-hard-200": 0.0,
}

_DEFAULT_CATEGORY_TEMP: Final[float] = 0.7


def default_question_path() -> Path:
    """Path to bundled ``question.jsonl``."""
    root = resources.files("pepo.evaluator.data.mt_bench")
    return Path(str(root / "question.jsonl"))


def default_reference_answer_path() -> Path:
    root = resources.files("pepo.evaluator.data.mt_bench")
    return Path(str(root / "reference_answer" / "gpt-4.jsonl"))


def default_judge_prompts_path() -> Path:
    root = resources.files("pepo.evaluator.data.mt_bench")
    return Path(str(root / "judge_prompts.jsonl"))


def load_mt_bench_questions(path: Path | None = None) -> list[dict[str, Any]]:
    """Load and validate MT-Bench questions (one JSON object per line)."""
    p = path if path is not None else default_question_path()
    questions: list[dict[str, Any]] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "question_id" not in obj or "category" not in obj or "turns" not in obj:
                raise ValueError(f"Invalid MT-Bench question (missing keys): {obj!r}")
            turns = obj["turns"]
            if not isinstance(turns, list) or len(turns) != 2:
                raise ValueError(
                    f"Question {obj.get('question_id')}: expected turns to be "
                    "a list of length 2"
                )
            if not all(isinstance(t, str) for t in turns):
                raise ValueError(
                    f"Question {obj.get('question_id')}: turns must be strings"
                )
            questions.append(obj)
    questions.sort(key=lambda q: int(q["question_id"]))
    return questions


def category_temperature(
    category: str,
    overrides: dict[str, float] | None = None,
) -> float:
    """Sampling temperature for a category (FastChat ``gen_model_answer`` behavior)."""
    merged = {**FASTCHAT_MT_BENCH_TEMPERATURE, **(overrides or {})}
    return float(merged.get(category, _DEFAULT_CATEGORY_TEMP))


def validate_fastchat_answer_record(obj: dict[str, Any]) -> None:
    """Assert a single line matches FastChat model answer JSONL schema."""
    for key in ("question_id", "answer_id", "model_id", "choices", "tstamp"):
        if key not in obj:
            raise ValueError(f"Missing key {key!r}")
    choices = obj["choices"]
    if not isinstance(choices, list) or not choices:
        raise ValueError("choices must be a non-empty list")
    for i, choice_obj in enumerate(choices):
        if not isinstance(choice_obj, Mapping):
            raise ValueError(f"choices[{i}] must have index and turns")
        choice = cast(Mapping[str, Any], choice_obj)
        if "turns" not in choice or "index" not in choice:
            raise ValueError(f"choices[{i}] must have index and turns")
        turns = choice["turns"]
        if not isinstance(turns, list) or len(turns) != 2:
            raise ValueError(f"choices[{i}].turns must be a list of length 2")
