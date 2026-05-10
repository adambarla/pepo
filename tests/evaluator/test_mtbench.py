import json
from pathlib import Path
from typing import Sequence, cast

import pandas as pd

from pepo.evaluator.judges import BaseJudge, JudgePrompt
from pepo.evaluator.mtbench import MTBenchEvaluator
from pepo.model import BaseModel


class FakeGenerator:
    def get_name(self) -> str:
        return "mt64"

    def get_short_name(self) -> str:
        return "mt64"


class FakeModel:
    generator = FakeGenerator()

    def __init__(self) -> None:
        self.loaded_epochs = []
        self.unloaded = False

    def get_name(self, epoch=None) -> str:
        if epoch is None:
            return "fake-model"
        return f"fake-model-e{epoch}"

    def load(self, epoch=None) -> None:
        self.loaded_epochs.append(epoch)

    def unload(self) -> None:
        self.unloaded = True

    def generate_responses(self, prompts, **kwargs):
        outputs = []
        for prompt in prompts:
            if isinstance(prompt, list):
                text = "second answer"
            else:
                text = "first answer"
            outputs.append({"prompt": prompt, "output": text})
        return outputs, {}


class FakeJudge(BaseJudge):
    def __init__(self, completions: Sequence[str]) -> None:
        self.completions = list(completions)
        self.prompts: list[JudgePrompt] = []
        self.unloaded = False

    def generate(self, prompts: Sequence[JudgePrompt]) -> list[str]:
        self.prompts = list(prompts)
        return self.completions

    def unload(self) -> None:
        self.unloaded = True


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_parse_score_extracts_mtbench_rating() -> None:
    assert MTBenchEvaluator.parse_score("Looks good. Rating: [[8.5]]") == 8.5
    assert MTBenchEvaluator.parse_score("Rating: 8/10") == 8.0
    assert MTBenchEvaluator.parse_score("Score = 7.5") == 7.5
    assert MTBenchEvaluator.parse_score("No bracketed score") is None
    assert MTBenchEvaluator.parse_score("Rating: 12/10") is None


def test_mtbench_evaluator_generates_answers_and_scores(tmp_path: Path) -> None:
    questions_file = tmp_path / "questions.jsonl"
    _write_jsonl(
        questions_file,
        [
            {
                "question_id": 1,
                "category": "writing",
                "turns": ["Question one?", "Question two?"],
            }
        ],
    )
    judge = FakeJudge(["First turn is fine. [[7]]", "Second turn is better. [[9]]"])
    evaluator = MTBenchEvaluator(
        questions_file=str(questions_file),
        output_dir=str(tmp_path / "outputs"),
        judge=judge,
    )
    model = FakeModel()

    judgments_file = evaluator.evaluate(
        model=cast(BaseModel, model), epoch=3, overwrite=True
    )

    assert model.loaded_epochs == [3]
    assert model.unloaded is True
    assert judge.unloaded is True
    assert len(judge.prompts) == 2

    answer_files = list(
        (tmp_path / "outputs" / "mt_bench" / "responses").glob("*.jsonl")
    )
    assert len(answer_files) == 1
    answers = MTBenchEvaluator._load_jsonl(answer_files[0])
    assert answers[0]["choices"][0]["turns"] == ["first answer", "second answer"]

    judgments = MTBenchEvaluator._load_jsonl(judgments_file)
    assert [judgment["score"] for judgment in judgments] == [7.0, 9.0]
    assert [judgment["turn"] for judgment in judgments] == [1, 2]

    leaderboard_file = next(
        (tmp_path / "outputs" / "mt_bench" / "default" / "leaderboards").glob(
            "*_leaderboard.csv"
        )
    )
    leaderboard = pd.read_csv(leaderboard_file, index_col=0)
    assert leaderboard.iloc[0]["score"] == 8.0
    assert leaderboard.iloc[0]["turn_1_score"] == 7.0
    assert leaderboard.iloc[0]["turn_2_score"] == 9.0


def test_mtbench_uses_reference_prompt_for_reference_categories(tmp_path: Path) -> None:
    questions_file = tmp_path / "questions.jsonl"
    references_file = tmp_path / "references.jsonl"
    _write_jsonl(
        questions_file,
        [
            {
                "question_id": 101,
                "category": "reasoning",
                "turns": ["Question one?", "Question two?"],
            }
        ],
    )
    _write_jsonl(
        references_file,
        [
            {
                "question_id": 101,
                "choices": [{"index": 0, "turns": ["Reference one", "Reference two"]}],
            }
        ],
    )
    evaluator = MTBenchEvaluator(
        questions_file=str(questions_file),
        reference_answers_file=str(references_file),
        output_dir=str(tmp_path / "outputs"),
        judge=FakeJudge([]),
    )

    template = evaluator._select_judge_template(evaluator.questions[0], 0)
    prompt = evaluator._format_judge_prompt(
        template=template,
        question=evaluator.questions[0],
        answer_turns=["Answer one", "Answer two"],
        turn_index=0,
    )

    assert template.name == "single-math-v1"
    assert "Reference one" in prompt.user_prompt
