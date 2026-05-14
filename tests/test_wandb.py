from typing import Any, cast

import pytest

from pepo.generator import Generator
from pepo.model import BaseModel
from pepo.utils.wandb import WandbManager


def test_generate_eval_run_name_includes_evaluator_name() -> None:
    run_name = WandbManager._generate_eval_run_name(
        model_name="model",
        generator_name="gen",
        epoch=3,
        evaluator_name="mtbench",
    )

    assert run_name == "model-gen-e3-mtbench-eval"


class FakeGenerator:
    def get_name(self) -> str:
        return "gen"


class FakeModel:
    def get_name(self) -> str:
        return "full-model"

    def _get_base_model_name(self) -> str:
        return "base-model"


def test_eval_handler_tags_model_and_evaluator(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WandbManager(
        enabled=True,
        project="project",
        tags=["eval"],
        mode="disabled",
    )
    monkeypatch.setattr(
        manager, "find_training_run_group", lambda model, model_idx=0: None
    )
    monkeypatch.setattr(manager, "find_run_by_name", lambda run_name: None)

    handler = manager.get_evaluation_handler(
        model=cast(BaseModel, cast(Any, FakeModel())),
        generator=cast(Generator, cast(Any, FakeGenerator())),
        epoch=1,
        evaluator_name="mtbench",
    )

    assert handler is not None
    assert handler.tags == [
        "eval",
        "base-model",
        "model:full-model",
        "evaluator:mtbench",
    ]
    assert handler.job_type == "evaluation/mtbench"
