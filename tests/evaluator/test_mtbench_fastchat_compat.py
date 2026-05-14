from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import pandas as pd
import pytest

from pepo.evaluator.judges import BaseJudge, JudgePrompt
from pepo.evaluator.mt_bench_data import FASTCHAT_MT_BENCH_TEMPERATURE
from pepo.evaluator.mtbench import MTBenchEvaluator

REPO_ROOT = Path(__file__).resolve().parents[2]
FASTCHAT_ROOT = REPO_ROOT / "FastChat"
FASTCHAT_LLM_JUDGE = FASTCHAT_ROOT / "fastchat" / "llm_judge"
FASTCHAT_DATA = FASTCHAT_LLM_JUDGE / "data"
PEPO_MT_BENCH_DATA = REPO_ROOT / "src" / "pepo" / "evaluator" / "data" / "mt_bench"
EXPECTED_FASTCHAT_COMMIT = "587d5cfa1609a43d192cedb8441cac3c17db105d"


class NoOpJudge(BaseJudge):
    def generate(self, prompts: Sequence[JudgePrompt]) -> list[str]:
        return []


class SequenceJudge(BaseJudge):
    def __init__(self, completions: Sequence[str]) -> None:
        self.completions = list(completions)
        self.prompts: list[JudgePrompt] = []

    def generate(self, prompts: Sequence[JudgePrompt]) -> list[str]:
        self.prompts = list(prompts)
        return self.completions


class FakeConversation:
    roles = ("user", "assistant")

    def __init__(self) -> None:
        self.system_prompt = ""
        self.messages: list[tuple[str, str | None]] = []

    def set_system_message(self, message: str) -> None:
        self.system_prompt = message

    def append_message(self, role: str, message: str | None) -> None:
        self.messages.append((role, message))


def _assert_fastchat_file(path: Path) -> None:
    assert path.is_file(), (
        "FastChat submodule is required for MT-Bench compatibility tests; "
        f"missing {path}"
    )


def _write_jsonl(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _load_fastchat_common(monkeypatch: pytest.MonkeyPatch):
    common_path = FASTCHAT_LLM_JUDGE / "common.py"
    _assert_fastchat_file(common_path)

    openai_module = types.ModuleType("openai")
    anthropic_module = types.ModuleType("anthropic")
    fastchat_module = types.ModuleType("fastchat")
    fastchat_model_module = types.ModuleType("fastchat.model")
    model_adapter_module = types.ModuleType("fastchat.model.model_adapter")
    setattr(model_adapter_module, "OPENAI_MODEL_LIST", ["gpt-4"])
    setattr(model_adapter_module, "ANTHROPIC_MODEL_LIST", [])
    setattr(
        model_adapter_module,
        "get_conversation_template",
        lambda _model: FakeConversation(),
    )

    monkeypatch.setitem(sys.modules, "openai", openai_module)
    monkeypatch.setitem(sys.modules, "anthropic", anthropic_module)
    monkeypatch.setitem(sys.modules, "fastchat", fastchat_module)
    monkeypatch.setitem(sys.modules, "fastchat.model", fastchat_model_module)
    monkeypatch.setitem(
        sys.modules, "fastchat.model.model_adapter", model_adapter_module
    )

    module_name = "_fastchat_llm_judge_common_for_tests"
    spec = importlib.util.spec_from_file_location(module_name, common_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _load_fastchat_show_result(monkeypatch: pytest.MonkeyPatch):
    show_result_path = FASTCHAT_LLM_JUDGE / "show_result.py"
    _assert_fastchat_file(show_result_path)

    module_name = "_fastchat_llm_judge_show_result_for_tests"
    spec = importlib.util.spec_from_file_location(module_name, show_result_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_fastchat_submodule_matches_documented_pin() -> None:
    _assert_fastchat_file(FASTCHAT_ROOT / "fastchat" / "llm_judge" / "common.py")

    result = subprocess.run(
        ["git", "-C", str(FASTCHAT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    version_text = (PEPO_MT_BENCH_DATA / "VERSION").read_text(encoding="utf-8")

    assert result.stdout.strip() == EXPECTED_FASTCHAT_COMMIT
    assert EXPECTED_FASTCHAT_COMMIT in version_text


def test_fastchat_submodule_uses_https_url() -> None:
    gitmodules = (REPO_ROOT / ".gitmodules").read_text(encoding="utf-8")

    assert "url = https://github.com/lm-sys/FastChat.git" in gitmodules
    assert "url = git@github.com:lm-sys/FastChat.git" not in gitmodules


@pytest.mark.parametrize(
    ("local_path", "fastchat_path"),
    [
        (
            PEPO_MT_BENCH_DATA / "question.jsonl",
            FASTCHAT_DATA / "mt_bench" / "question.jsonl",
        ),
        (
            PEPO_MT_BENCH_DATA / "judge_prompts.jsonl",
            FASTCHAT_DATA / "judge_prompts.jsonl",
        ),
        (
            PEPO_MT_BENCH_DATA / "reference_answer" / "gpt-4.jsonl",
            FASTCHAT_DATA / "mt_bench" / "reference_answer" / "gpt-4.jsonl",
        ),
    ],
)
def test_vendored_mtbench_assets_match_pinned_fastchat(
    local_path: Path,
    fastchat_path: Path,
) -> None:
    _assert_fastchat_file(fastchat_path)

    assert local_path.read_bytes() == fastchat_path.read_bytes()


def test_category_temperatures_match_fastchat_common(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common = _load_fastchat_common(monkeypatch)

    assert FASTCHAT_MT_BENCH_TEMPERATURE == common.temperature_config


def test_single_score_parsing_matches_fastchat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common = _load_fastchat_common(monkeypatch)
    judge_prompts = common.load_judge_prompts(
        str(FASTCHAT_DATA / "judge_prompts.jsonl")
    )
    question = {"question_id": 81, "category": "writing", "turns": ["Q1", "Q2"]}
    answer = {"choices": [{"index": 0, "turns": ["A1", "A2"]}]}
    judge = common.Judge("gpt-4", judge_prompts["single-v1"])

    for completion in (
        "Rating: [[8.5]]",
        "Backup format [7.5]",
        "Rating: 8/10",
        "Rating: [[ 8 ]]",
        "No parseable score",
    ):
        common.chat_completion_openai = lambda *args, **kwargs: completion
        fastchat_score, _prompt, _judgment = common.run_judge_single(
            question,
            answer,
            judge,
            ref_answer=None,
            multi_turn=False,
        )

        assert MTBenchEvaluator.parse_score(completion) == float(fastchat_score)


def test_pairwise_verdict_parsing_matches_fastchat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common = _load_fastchat_common(monkeypatch)
    judge_prompts = common.load_judge_prompts(
        str(FASTCHAT_DATA / "judge_prompts.jsonl")
    )
    question = {"question_id": 81, "category": "writing", "turns": ["Q1", "Q2"]}
    answer_a = {"choices": [{"index": 0, "turns": ["A1", "A2"]}]}
    answer_b = {"choices": [{"index": 0, "turns": ["B1", "B2"]}]}
    judge = common.Judge("gpt-4", judge_prompts["pair-v2"])

    for completion in (
        "Assistant A wins. [[A]]",
        "Assistant B wins. [[B]]",
        "Tie. [[C]]",
        "lowercase is not FastChat-compatible [[c]]",
        "whitespace is not FastChat-compatible [[ A ]]",
        "No verdict",
    ):
        common.chat_completion_openai = lambda *args, **kwargs: completion
        fastchat_winner, _prompt, _judgment = common.run_judge_pair(
            question,
            answer_a,
            answer_b,
            judge,
            ref_answer=None,
            multi_turn=False,
        )
        verdict = MTBenchEvaluator.parse_pairwise_verdict(completion)
        our_winner = MTBenchEvaluator._pairwise_game_winner(
            verdict,
            assistant_a="A",
            assistant_b="B",
        )
        expected_winner = {"A": "A", "B": "B"}.get(fastchat_winner, fastchat_winner)

        assert our_winner == expected_winner


@pytest.fixture()
def prompt_fixture(tmp_path: Path):
    questions = [
        {
            "question_id": 1,
            "category": "writing",
            "turns": ["General question one?", "General question two?"],
        },
        {
            "question_id": 101,
            "category": "reasoning",
            "turns": ["Reasoning question one?", "Reasoning question two?"],
        },
    ]
    references = [
        {
            "question_id": 101,
            "choices": [
                {
                    "index": 0,
                    "turns": ["Reference answer one", "Reference answer two"],
                }
            ],
        }
    ]
    answers = {
        1: {
            "question_id": 1,
            "choices": [
                {"index": 0, "turns": ["General answer one", "General answer two"]}
            ],
        },
        101: {
            "question_id": 101,
            "choices": [
                {"index": 0, "turns": ["Reasoning answer one", "Reasoning answer two"]}
            ],
        },
    }
    ref_answers = {
        101: {
            "question_id": 101,
            "choices": [
                {"index": 0, "turns": ["Reference answer one", "Reference answer two"]}
            ],
        }
    }
    baseline_answers = {
        1: {
            "question_id": 1,
            "choices": [
                {
                    "index": 0,
                    "turns": ["General baseline one", "General baseline two"],
                }
            ],
        },
        101: {
            "question_id": 101,
            "choices": [
                {
                    "index": 0,
                    "turns": ["Reasoning baseline one", "Reasoning baseline two"],
                }
            ],
        },
    }

    questions_file = tmp_path / "questions.jsonl"
    references_file = tmp_path / "references.jsonl"
    _write_jsonl(questions_file, questions)
    _write_jsonl(references_file, references)

    evaluator = MTBenchEvaluator(
        questions_file=str(questions_file),
        judge_prompts_file=str(FASTCHAT_DATA / "judge_prompts.jsonl"),
        reference_answers_file=str(references_file),
        output_dir=str(tmp_path / "outputs"),
        judge=NoOpJudge(),
    )
    return SimpleNamespace(
        evaluator=evaluator,
        question_records={record["question_id"]: record for record in questions},
        answer_records=answers,
        baseline_answer_records=baseline_answers,
        reference_records=ref_answers,
    )


@pytest.fixture()
def bundled_evaluator(tmp_path: Path) -> MTBenchEvaluator:
    return MTBenchEvaluator(
        output_dir=str(tmp_path / "outputs"),
        judge=NoOpJudge(),
    )


def _question_by_id(evaluator: MTBenchEvaluator, question_id: int):
    return next(
        question
        for question in evaluator.questions
        if question.question_id == question_id
    )


def _fastchat_question_by_id(question_id: int) -> dict:
    with open(
        FASTCHAT_DATA / "mt_bench" / "question.jsonl", encoding="utf-8"
    ) as handle:
        for line in handle:
            record = json.loads(line)
            if int(record["question_id"]) == question_id:
                return record
    raise AssertionError(f"FastChat question {question_id} not found")


def _fastchat_ref_by_id(question_id: int) -> dict | None:
    ref_path = FASTCHAT_DATA / "mt_bench" / "reference_answer" / "gpt-4.jsonl"
    with open(ref_path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if int(record["question_id"]) == question_id:
                return record
    return None


@pytest.mark.parametrize(
    ("question_id", "turn_index", "single_template", "pairwise_template"),
    [
        (81, 0, "single-v1", "pair-v2"),
        (81, 1, "single-v1-multi-turn", "pair-v2-multi-turn"),
        (91, 0, "single-v1", "pair-v2"),
        (91, 1, "single-v1-multi-turn", "pair-v2-multi-turn"),
        (101, 0, "single-math-v1", "pair-math-v1"),
        (101, 1, "single-math-v1-multi-turn", "pair-math-v1-multi-turn"),
        (111, 0, "single-math-v1", "pair-math-v1"),
        (111, 1, "single-math-v1-multi-turn", "pair-math-v1-multi-turn"),
        (121, 0, "single-math-v1", "pair-math-v1"),
        (121, 1, "single-math-v1-multi-turn", "pair-math-v1-multi-turn"),
    ],
)
def test_template_selection_matches_fastchat_categories(
    bundled_evaluator: MTBenchEvaluator,
    question_id: int,
    turn_index: int,
    single_template: str,
    pairwise_template: str,
) -> None:
    question = _question_by_id(bundled_evaluator, question_id)

    assert (
        bundled_evaluator._select_judge_template(question, turn_index).name
        == single_template
    )
    assert (
        bundled_evaluator._select_pairwise_judge_template(question, turn_index).name
        == pairwise_template
    )


@pytest.mark.parametrize(
    ("question_id", "turn_index"),
    [
        (1, 0),
        (1, 1),
        (101, 0),
        (101, 1),
    ],
)
def test_single_judge_prompts_match_fastchat_formatting(
    monkeypatch: pytest.MonkeyPatch,
    prompt_fixture,
    question_id: int,
    turn_index: int,
) -> None:
    common = _load_fastchat_common(monkeypatch)
    common.chat_completion_openai = lambda *args, **kwargs: "Rating: [[8]]"
    fastchat_prompts = common.load_judge_prompts(
        str(FASTCHAT_DATA / "judge_prompts.jsonl")
    )

    evaluator = prompt_fixture.evaluator
    question = next(q for q in evaluator.questions if q.question_id == question_id)
    template = evaluator._select_judge_template(question, turn_index)
    our_prompt = evaluator._format_judge_prompt(
        template=template,
        question=question,
        answer_turns=prompt_fixture.answer_records[question_id]["choices"][0]["turns"],
        turn_index=turn_index,
    )

    fastchat_judge = common.Judge(
        "gpt-4",
        fastchat_prompts[template.name],
        ref_based=question_id in prompt_fixture.reference_records,
        multi_turn=turn_index == 1,
    )
    _score, fastchat_user_prompt, _judgment = common.run_judge_single(
        prompt_fixture.question_records[question_id],
        prompt_fixture.answer_records[question_id],
        fastchat_judge,
        prompt_fixture.reference_records.get(question_id),
        multi_turn=turn_index == 1,
    )

    assert our_prompt.system_prompt == fastchat_prompts[template.name]["system_prompt"]
    assert our_prompt.user_prompt == fastchat_user_prompt


@pytest.mark.parametrize(
    ("question_id", "turn_index"),
    [
        (81, 0),
        (81, 1),
        (101, 0),
        (101, 1),
        (111, 0),
        (111, 1),
        (121, 0),
        (121, 1),
    ],
)
def test_real_mtbench_single_prompts_match_fastchat_formatting(
    monkeypatch: pytest.MonkeyPatch,
    bundled_evaluator: MTBenchEvaluator,
    question_id: int,
    turn_index: int,
) -> None:
    common = _load_fastchat_common(monkeypatch)
    common.chat_completion_openai = lambda *args, **kwargs: "Rating: [[8]]"
    fastchat_prompts = common.load_judge_prompts(
        str(FASTCHAT_DATA / "judge_prompts.jsonl")
    )
    question = _question_by_id(bundled_evaluator, question_id)
    question_record = _fastchat_question_by_id(question_id)
    answer_turns = ["model answer one", "model answer two"]
    answer = {
        "question_id": question_id,
        "choices": [{"index": 0, "turns": answer_turns}],
    }
    reference_answer = _fastchat_ref_by_id(question_id)
    template = bundled_evaluator._select_judge_template(question, turn_index)

    our_prompt = bundled_evaluator._format_judge_prompt(
        template=template,
        question=question,
        answer_turns=answer_turns,
        turn_index=turn_index,
    )
    fastchat_judge = common.Judge(
        "gpt-4",
        fastchat_prompts[template.name],
        ref_based=reference_answer is not None,
        multi_turn=turn_index == 1,
    )
    _score, fastchat_user_prompt, _judgment = common.run_judge_single(
        question_record,
        answer,
        fastchat_judge,
        reference_answer,
        multi_turn=turn_index == 1,
    )

    assert our_prompt.system_prompt == fastchat_prompts[template.name]["system_prompt"]
    assert our_prompt.user_prompt == fastchat_user_prompt


@pytest.mark.parametrize(
    ("question_id", "turn_index"),
    [
        (1, 0),
        (1, 1),
        (101, 0),
        (101, 1),
    ],
)
def test_pairwise_judge_prompts_match_fastchat_formatting(
    monkeypatch: pytest.MonkeyPatch,
    prompt_fixture,
    question_id: int,
    turn_index: int,
) -> None:
    common = _load_fastchat_common(monkeypatch)
    common.chat_completion_openai = lambda *args, **kwargs: "Assistant A wins. [[A]]"
    fastchat_prompts = common.load_judge_prompts(
        str(FASTCHAT_DATA / "judge_prompts.jsonl")
    )

    evaluator = prompt_fixture.evaluator
    question = next(q for q in evaluator.questions if q.question_id == question_id)
    template = evaluator._select_pairwise_judge_template(question, turn_index)
    answer = prompt_fixture.answer_records[question_id]
    baseline_answer = prompt_fixture.baseline_answer_records[question_id]
    reference_answer = prompt_fixture.reference_records.get(question_id)
    fastchat_judge = common.Judge(
        "gpt-4",
        fastchat_prompts[template.name],
        ref_based=reference_answer is not None,
        multi_turn=turn_index == 1,
    )

    our_g1_prompt = evaluator._format_pairwise_judge_prompt(
        template=template,
        question=question,
        answer_a_turns=answer["choices"][0]["turns"],
        answer_b_turns=baseline_answer["choices"][0]["turns"],
        turn_index=turn_index,
    )
    _g1_winner, fastchat_g1_user_prompt, _g1_judgment = common.run_judge_pair(
        prompt_fixture.question_records[question_id],
        answer,
        baseline_answer,
        fastchat_judge,
        reference_answer,
        multi_turn=turn_index == 1,
    )
    our_g2_prompt = evaluator._format_pairwise_judge_prompt(
        template=template,
        question=question,
        answer_a_turns=baseline_answer["choices"][0]["turns"],
        answer_b_turns=answer["choices"][0]["turns"],
        turn_index=turn_index,
    )
    _g2_winner, fastchat_g2_user_prompt, _g2_judgment = common.run_judge_pair(
        prompt_fixture.question_records[question_id],
        baseline_answer,
        answer,
        fastchat_judge,
        reference_answer,
        multi_turn=turn_index == 1,
    )

    assert (
        our_g1_prompt.system_prompt == fastchat_prompts[template.name]["system_prompt"]
    )
    assert (
        our_g2_prompt.system_prompt == fastchat_prompts[template.name]["system_prompt"]
    )
    assert our_g1_prompt.user_prompt == fastchat_g1_user_prompt
    assert our_g2_prompt.user_prompt == fastchat_g2_user_prompt


@pytest.mark.parametrize(
    ("question_id", "turn_index"),
    [
        (81, 0),
        (81, 1),
        (101, 0),
        (101, 1),
        (111, 0),
        (111, 1),
        (121, 0),
        (121, 1),
    ],
)
def test_real_mtbench_pairwise_prompts_match_fastchat_formatting(
    monkeypatch: pytest.MonkeyPatch,
    bundled_evaluator: MTBenchEvaluator,
    question_id: int,
    turn_index: int,
) -> None:
    common = _load_fastchat_common(monkeypatch)
    common.chat_completion_openai = lambda *args, **kwargs: "Assistant A wins. [[A]]"
    fastchat_prompts = common.load_judge_prompts(
        str(FASTCHAT_DATA / "judge_prompts.jsonl")
    )
    question = _question_by_id(bundled_evaluator, question_id)
    question_record = _fastchat_question_by_id(question_id)
    answer_turns = ["model answer one", "model answer two"]
    baseline_turns = ["baseline answer one", "baseline answer two"]
    answer = {
        "question_id": question_id,
        "choices": [{"index": 0, "turns": answer_turns}],
    }
    baseline_answer = {
        "question_id": question_id,
        "choices": [{"index": 0, "turns": baseline_turns}],
    }
    reference_answer = _fastchat_ref_by_id(question_id)
    template = bundled_evaluator._select_pairwise_judge_template(question, turn_index)
    fastchat_judge = common.Judge(
        "gpt-4",
        fastchat_prompts[template.name],
        ref_based=reference_answer is not None,
        multi_turn=turn_index == 1,
    )

    our_prompt = bundled_evaluator._format_pairwise_judge_prompt(
        template=template,
        question=question,
        answer_a_turns=answer_turns,
        answer_b_turns=baseline_turns,
        turn_index=turn_index,
    )
    _winner, fastchat_user_prompt, _judgment = common.run_judge_pair(
        question_record,
        answer,
        baseline_answer,
        fastchat_judge,
        reference_answer,
        multi_turn=turn_index == 1,
    )

    assert our_prompt.system_prompt == fastchat_prompts[template.name]["system_prompt"]
    assert our_prompt.user_prompt == fastchat_user_prompt


@pytest.mark.parametrize(
    "completion",
    [
        "Assistant A wins. [[A]]",
        "Assistant B wins. [[B]]",
        "Tie. [[C]]",
        "No parseable verdict",
    ],
)
def test_pairwise_judge_mapping_matches_fastchat_two_game_logic(
    monkeypatch: pytest.MonkeyPatch,
    prompt_fixture,
    completion: str,
) -> None:
    common = _load_fastchat_common(monkeypatch)
    common.chat_completion_openai = lambda *args, **kwargs: completion
    fastchat_prompts = common.load_judge_prompts(
        str(FASTCHAT_DATA / "judge_prompts.jsonl")
    )

    question_id = 1
    question = prompt_fixture.evaluator.questions[0]
    template = prompt_fixture.evaluator._select_pairwise_judge_template(question, 0)
    fastchat_judge = common.Judge("gpt-4", fastchat_prompts[template.name])
    answer = prompt_fixture.answer_records[question_id]
    ref_turns = ["Reference model one", "Reference model two"]
    ref_answer = {
        "question_id": question_id,
        "choices": [{"index": 0, "turns": ref_turns}],
    }

    g1_winner, g1_user_prompt, _g1_judgment = common.run_judge_pair(
        prompt_fixture.question_records[question_id],
        answer,
        ref_answer,
        fastchat_judge,
        ref_answer=None,
        multi_turn=False,
    )
    g2_winner, g2_user_prompt, _g2_judgment = common.run_judge_pair(
        prompt_fixture.question_records[question_id],
        ref_answer,
        answer,
        fastchat_judge,
        ref_answer=None,
        multi_turn=False,
    )
    fastchat_g1 = {"A": "model_1", "B": "model_2"}.get(g1_winner, g1_winner)
    fastchat_g2 = {"A": "model_2", "B": "model_1"}.get(g2_winner, g2_winner)

    our_g1_prompt = prompt_fixture.evaluator._format_pairwise_judge_prompt(
        template=template,
        question=question,
        answer_a_turns=answer["choices"][0]["turns"],
        answer_b_turns=ref_turns,
        turn_index=0,
    )
    our_g2_prompt = prompt_fixture.evaluator._format_pairwise_judge_prompt(
        template=template,
        question=question,
        answer_a_turns=ref_turns,
        answer_b_turns=answer["choices"][0]["turns"],
        turn_index=0,
    )
    verdict = MTBenchEvaluator.parse_pairwise_verdict(completion)

    assert our_g1_prompt.user_prompt == g1_user_prompt
    assert our_g2_prompt.user_prompt == g2_user_prompt
    assert (
        MTBenchEvaluator._pairwise_game_winner(verdict, "model_1", "model_2")
        == fastchat_g1
    )
    assert (
        MTBenchEvaluator._pairwise_game_winner(verdict, "model_2", "model_1")
        == fastchat_g2
    )


def test_pairwise_orchestration_matches_fastchat_play_match_rows(
    monkeypatch: pytest.MonkeyPatch,
    prompt_fixture,
) -> None:
    common = _load_fastchat_common(monkeypatch)
    fastchat_prompts = common.load_judge_prompts(
        str(FASTCHAT_DATA / "judge_prompts.jsonl")
    )
    completions = [
        "General turn 1 says A. [[A]]",
        "General turn 1 swapped says B. [[B]]",
        "General turn 2 says B. [[B]]",
        "General turn 2 swapped says A. [[A]]",
        "Reasoning turn 1 tie. [[C]]",
        "Reasoning turn 1 swapped tie. [[C]]",
        "Reasoning turn 2 parse error.",
        "Reasoning turn 2 swapped says A. [[A]]",
    ]
    completion_iter = iter(completions)
    common.chat_completion_openai = lambda *args, **kwargs: next(completion_iter)
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    fastchat_rows = []
    for question in prompt_fixture.evaluator.questions:
        question_record = prompt_fixture.question_records[question.question_id]
        answer = prompt_fixture.answer_records[question.question_id]
        baseline_answer = prompt_fixture.baseline_answer_records[question.question_id]
        reference_answer = prompt_fixture.reference_records.get(question.question_id)
        for turn_index in (0, 1):
            template = prompt_fixture.evaluator._select_pairwise_judge_template(
                question,
                turn_index,
            )
            fastchat_judge = common.Judge(
                "gpt-4",
                fastchat_prompts[template.name],
                ref_based=reference_answer is not None,
                multi_turn=turn_index == 1,
            )
            match = common.MatchPair(
                question_record,
                "model",
                "ref",
                answer,
                baseline_answer,
                fastchat_judge,
                ref_answer=reference_answer,
                multi_turn=turn_index == 1,
            )
            fastchat_rows.append(common.play_a_match_pair(match, output_file=None))

    evaluator = prompt_fixture.evaluator
    evaluator.judge = SequenceJudge(completions)
    our_rows = evaluator._judge_pairwise_answers(
        answers=list(prompt_fixture.answer_records.values()),
        ref_answers=list(prompt_fixture.baseline_answer_records.values()),
        model_name="model",
        ref_model_name="ref",
    )

    assert len(our_rows) == len(fastchat_rows)
    for our_row, fastchat_row in zip(our_rows, fastchat_rows, strict=True):
        shared_fields = set(fastchat_row) - {"judge", "tstamp"}
        assert shared_fields <= set(our_row)
        for field in shared_fields:
            assert our_row[field] == fastchat_row[field]
        assert isinstance(our_row["tstamp"], float)
        assert isinstance(fastchat_row["tstamp"], float)
        assert our_row["judge"][1] == fastchat_row["judge"][1]
        assert our_row["judge"][0] == "local"
        assert fastchat_row["judge"][0] == "gpt-4"
        assert "category" in our_row


def test_pairwise_leaderboard_matches_fastchat_show_result_logic(
    monkeypatch: pytest.MonkeyPatch,
    prompt_fixture,
    tmp_path: Path,
) -> None:
    show_result = _load_fastchat_show_result(monkeypatch)
    judgments = [
        {
            "question_id": 1,
            "category": "writing",
            "turn": 1,
            "model_1": "model",
            "model_2": "ref",
            "g1_winner": "model_1",
            "g2_winner": "model_1",
            "g1_judgment": "[[A]]",
            "g2_judgment": "[[B]]",
        },
        {
            "question_id": 1,
            "category": "writing",
            "turn": 2,
            "model_1": "model",
            "model_2": "ref",
            "g1_winner": "model_2",
            "g2_winner": "model_2",
            "g1_judgment": "[[B]]",
            "g2_judgment": "[[A]]",
        },
        {
            "question_id": 101,
            "category": "reasoning",
            "turn": 1,
            "model_1": "model",
            "model_2": "ref",
            "g1_winner": "tie",
            "g2_winner": "tie",
            "g1_judgment": "[[C]]",
            "g2_judgment": "[[C]]",
        },
        {
            "question_id": 101,
            "category": "reasoning",
            "turn": 2,
            "model_1": "model",
            "model_2": "ref",
            "g1_winner": "model_1",
            "g2_winner": "model_2",
            "g1_judgment": "[[A]]",
            "g2_judgment": "[[A]]",
        },
        {
            "question_id": 102,
            "category": "reasoning",
            "turn": 1,
            "model_1": "model",
            "model_2": "ref",
            "g1_winner": "error",
            "g2_winner": "model_1",
            "g1_judgment": "missing verdict",
            "g2_judgment": "[[B]]",
        },
    ]
    answers = [
        {"question_id": 1, "choices": [{"index": 0, "turns": ["a", "b"]}]},
        {"question_id": 101, "choices": [{"index": 0, "turns": ["c", "d"]}]},
    ]

    input_file = tmp_path / "fastchat_pairwise.jsonl"
    _write_jsonl(input_file, judgments)
    printed: list[object] = []
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: printed.extend(args))
    show_result.display_result_pairwise(
        SimpleNamespace(
            input_file=str(input_file),
            bench_name="mt_bench",
            judge_model="gpt-4",
            baseline_model="ref",
            model_list=None,
        )
    )
    fastchat_df = next(item for item in printed if isinstance(item, pd.DataFrame))
    fastchat_row = fastchat_df.loc["model"]

    our_leaderboard = prompt_fixture.evaluator._build_pairwise_leaderboard(
        judgments=judgments,
        answers=answers,
        model_name="model",
        ref_model_name="ref",
    )
    our_row = our_leaderboard.loc["model"]

    assert our_row["win_rate"] == pytest.approx(fastchat_row["win_rate"])
    assert our_row["loss_rate"] == pytest.approx(fastchat_row["loss_rate"])
    fastchat_total = fastchat_row["win"] + fastchat_row["loss"] + fastchat_row["tie"]
    assert our_row["tie_rate"] == pytest.approx(fastchat_row["tie"] / fastchat_total)
    assert our_row["win_rate_adjusted"] == pytest.approx(
        fastchat_row["win_rate_adjusted"]
    )
    assert our_row["n_judgments"] == len(judgments)
    assert our_row["n_judge_games"] == len(judgments) * 2
    assert our_row["n_valid_judgments"] == fastchat_total
    assert our_row["writing_win_rate"] == pytest.approx(0.5)
    assert our_row["writing_win_rate_adjusted"] == pytest.approx(0.5)
    assert our_row["reasoning_win_rate"] == pytest.approx(0.0)
    assert our_row["reasoning_win_rate_adjusted"] == pytest.approx(0.5)
    assert our_row["parse_error_rate"] == pytest.approx(1 / len(judgments))


def test_pairwise_leaderboard_handles_all_invalid_rows(prompt_fixture) -> None:
    judgments = [
        {
            "question_id": 1,
            "category": "writing",
            "turn": 1,
            "model_1": "model",
            "model_2": "ref",
            "g1_winner": "error",
            "g2_winner": "model_1",
            "g1_judgment": "missing verdict",
            "g2_judgment": "[[B]]",
        },
        {
            "question_id": 1,
            "category": "writing",
            "turn": 2,
            "model_1": "model",
            "model_2": "ref",
            "g1_winner": "model_2",
            "g2_winner": "error",
            "g1_judgment": "[[B]]",
            "g2_judgment": "missing verdict",
        },
    ]
    answers = [{"question_id": 1, "choices": [{"index": 0, "turns": ["a", "b"]}]}]

    leaderboard = prompt_fixture.evaluator._build_pairwise_leaderboard(
        judgments=judgments,
        answers=answers,
        model_name="model",
        ref_model_name="ref",
    )
    row = leaderboard.loc["model"]

    assert row["win_rate"] == 0.0
    assert row["loss_rate"] == 0.0
    assert row["tie_rate"] == 0.0
    assert row["win_rate_adjusted"] == 0.0
    assert row["parse_error_rate"] == 1.0
    assert row["n_valid_judgments"] == 0
