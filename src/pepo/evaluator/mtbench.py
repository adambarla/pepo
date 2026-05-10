from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from ..model import BaseModel
from ..utils import WandbRun
from .base import BaseEvaluator
from .judges import BaseJudge, JudgePrompt

logger = logging.getLogger(__name__)

_SCORE_PATTERN = re.compile(r"\[\[(\d+(?:\.\d+)?)\]\]")
_REFERENCE_CATEGORIES = {"math", "reasoning", "coding"}


@dataclass(frozen=True)
class MTBenchQuestion:
    question_id: int
    category: str
    turns: list[str]


@dataclass(frozen=True)
class JudgePromptTemplate:
    name: str
    system_prompt: str
    prompt_template: str


class MTBenchEvaluator(BaseEvaluator):
    """MT-Bench evaluator using PEPO generation and a configurable local judge."""

    def __init__(
        self,
        questions_file: str,
        output_dir: str,
        judge: BaseJudge | DictConfig | Mapping[str, Any],
        judge_prompts_file: Optional[str] = None,
        reference_answers_file: Optional[str] = None,
        dataset_id: str = "mt_bench",
        dataset_split: str = "default",
        num_samples: Optional[int] = None,
        stop_after_generation: bool = False,
        wandb_run: Optional[WandbRun] = None,
    ) -> None:
        super().__init__(
            dataset_id=dataset_id,
            dataset_split=dataset_split,
            output_dir=f"{output_dir}/mt_bench/",
            num_samples=num_samples,
            stop_after_generation=stop_after_generation,
        )
        self.questions_file = Path(questions_file)
        self.judge_prompts_file = (
            Path(judge_prompts_file) if judge_prompts_file is not None else None
        )
        self.reference_answers_file = (
            Path(reference_answers_file)
            if reference_answers_file is not None
            else None
        )
        self.judge = self._init_judge(judge)
        self.wandb_run = wandb_run
        self.questions = self._load_questions()
        self.judge_prompts = self._load_judge_prompts()
        self.reference_answers = self._load_reference_answers()

    def evaluate(
        self,
        model: BaseModel,
        epoch: Optional[int] = None,
        ref_model: Optional[BaseModel] = None,
        ref_epoch: Optional[int] = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> Path:
        if ref_model is not None:
            raise ValueError("MTBenchEvaluator currently supports single-model grading.")

        answers_file = self._get_or_generate(model, epoch=epoch, overwrite=overwrite)
        logger.info("MT-Bench answers file: %s", answers_file)

        if self.stop_after_generation:
            logger.info("Stopping MT-Bench evaluation after response generation.")
            return answers_file

        folder = self._get_folder()
        judgments_folder = folder / "judgments"
        judgments_folder.mkdir(parents=True, exist_ok=True)
        leaderboards_folder = folder / "leaderboards"
        leaderboards_folder.mkdir(parents=True, exist_ok=True)

        filename = self._get_filename(model, epoch)
        judgments_file = judgments_folder / f"{filename}_judgments.jsonl"
        leaderboard_file = leaderboards_folder / f"{filename}_leaderboard.csv"

        if overwrite or not judgments_file.exists():
            judgments = self._judge_answers(
                answers=self._load_jsonl(answers_file),
                model_name=model.get_name(epoch=epoch),
            )
            self._write_jsonl(judgments_file, judgments)
        else:
            judgments = self._load_jsonl(judgments_file)

        leaderboard = self._build_leaderboard(
            judgments=judgments,
            answers=self._load_jsonl(answers_file),
            model_name=self._get_filename(model, epoch),
        )
        leaderboard.to_csv(leaderboard_file, index=True)
        self._log_metrics(leaderboard=leaderboard, model=model, epoch=epoch)
        return judgments_file

    def _get_or_generate(
        self,
        model: BaseModel,
        epoch: Optional[int],
        overwrite: bool,
    ) -> Path:
        responses_folder = self.output_dir / "responses"
        responses_folder.mkdir(parents=True, exist_ok=True)
        filename = self._get_filename(model, epoch)
        answers_file = responses_folder / f"{filename}_answers.jsonl"
        if overwrite or not answers_file.exists():
            self._generate_answers(answers_file, model, epoch=epoch)
        return answers_file

    def _generate_answers(
        self,
        answers_file: Path,
        model: BaseModel,
        epoch: Optional[int],
    ) -> None:
        logger.info(
            "Generating MT-Bench answers for %d questions with %s",
            len(self.questions),
            model.get_name(epoch=epoch),
        )
        model.load(epoch=epoch)
        try:
            turn_1_prompts = [question.turns[0] for question in self.questions]
            turn_1_outputs, turn_1_metrics = model.generate_responses(
                prompts=turn_1_prompts
            )
            turn_1_by_prompt = self._outputs_by_prompt(turn_1_outputs)
            turn_1_answers = [
                turn_1_by_prompt[self._prompt_key(question.turns[0])]
                for question in self.questions
            ]

            turn_2_prompts = [
                [
                    {"role": "user", "content": question.turns[0]},
                    {"role": "assistant", "content": turn_1_answer},
                    {"role": "user", "content": question.turns[1]},
                ]
                for question, turn_1_answer in zip(
                    self.questions, turn_1_answers, strict=True
                )
            ]
            turn_2_outputs, turn_2_metrics = model.generate_responses(
                prompts=turn_2_prompts
            )
            turn_2_by_prompt = self._outputs_by_prompt(turn_2_outputs)
            turn_2_answers = [
                turn_2_by_prompt[self._prompt_key(prompt)]
                for prompt in turn_2_prompts
            ]
        finally:
            model.unload()

        self._log_generation_metrics(
            metrics={**turn_1_metrics, **turn_2_metrics},
            model=model,
            epoch=epoch,
        )

        model_name = model.get_name(epoch=epoch)
        answer_records = []
        for question, turn_1_answer, turn_2_answer in zip(
            self.questions, turn_1_answers, turn_2_answers, strict=True
        ):
            turns = [turn_1_answer, turn_2_answer]
            answer_records.append(
                {
                    "question_id": question.question_id,
                    "answer_id": self._answer_id(model_name, question.question_id, turns),
                    "model_id": model_name,
                    "choices": [{"index": 0, "turns": turns}],
                    "tstamp": time.time(),
                }
            )

        self._write_jsonl(answers_file, answer_records)
        logger.info("Saved %d MT-Bench answers to %s", len(answer_records), answers_file)

    def _judge_answers(
        self,
        answers: list[dict[str, Any]],
        model_name: str,
    ) -> list[dict[str, Any]]:
        answers_by_question_id = {
            int(answer["question_id"]): answer for answer in answers
        }
        judge_prompt_items: list[tuple[dict[str, Any], JudgePrompt]] = []
        for question in self.questions:
            answer = answers_by_question_id[question.question_id]
            turns = answer["choices"][0]["turns"]
            for turn_index in (0, 1):
                template = self._select_judge_template(question, turn_index)
                prompt = self._format_judge_prompt(
                    template=template,
                    question=question,
                    answer_turns=turns,
                    turn_index=turn_index,
                )
                metadata = {
                    "question_id": question.question_id,
                    "category": question.category,
                    "turn": turn_index + 1,
                    "model_id": model_name,
                    "judge_prompt": template.name,
                }
                judge_prompt_items.append((metadata, prompt))

        logger.info("Judging %d MT-Bench answer turns", len(judge_prompt_items))
        prompts = [item[1] for item in judge_prompt_items]
        try:
            completions = self.judge.generate(prompts)
        finally:
            self.judge.unload()

        judgments = []
        for (metadata, prompt), completion in zip(
            judge_prompt_items, completions, strict=True
        ):
            score = self.parse_score(completion)
            judgments.append(
                {
                    **metadata,
                    "score": score,
                    "judge_output": completion,
                    "system_prompt": prompt.system_prompt,
                    "user_prompt": prompt.user_prompt,
                }
            )
        return judgments

    def _format_judge_prompt(
        self,
        template: JudgePromptTemplate,
        question: MTBenchQuestion,
        answer_turns: Sequence[str],
        turn_index: int,
    ) -> JudgePrompt:
        values = {
            "question": question.turns[turn_index],
            "answer": answer_turns[turn_index],
            "question_1": question.turns[0],
            "question_2": question.turns[1],
            "answer_1": answer_turns[0],
            "answer_2": answer_turns[1],
            "ref_answer_1": self._get_reference_answer(question.question_id, 0),
            "ref_answer_2": self._get_reference_answer(question.question_id, 1),
        }
        return JudgePrompt(
            system_prompt=template.system_prompt,
            user_prompt=template.prompt_template.format(**values),
        )

    def _select_judge_template(
        self,
        question: MTBenchQuestion,
        turn_index: int,
    ) -> JudgePromptTemplate:
        has_reference = question.question_id in self.reference_answers
        use_reference = question.category in _REFERENCE_CATEGORIES and has_reference
        if use_reference:
            key = "single-math-v1-multi-turn" if turn_index == 1 else "single-math-v1"
        else:
            key = "single-v1-multi-turn" if turn_index == 1 else "single-v1"
        return self.judge_prompts[key]

    def _build_leaderboard(
        self,
        judgments: list[dict[str, Any]],
        answers: list[dict[str, Any]],
        model_name: str,
    ) -> pd.DataFrame:
        judgments_df = pd.DataFrame(judgments)
        scores = pd.to_numeric(judgments_df["score"], errors="coerce")
        metrics: dict[str, float | int] = {
            "score": float(scores.mean()),
            "turn_1_score": float(
                pd.to_numeric(
                    judgments_df.loc[judgments_df["turn"] == 1, "score"],
                    errors="coerce",
                ).mean()
            ),
            "turn_2_score": float(
                pd.to_numeric(
                    judgments_df.loc[judgments_df["turn"] == 2, "score"],
                    errors="coerce",
                ).mean()
            ),
            "parse_error_rate": float(scores.isna().mean()),
            "n_questions": len(answers),
            "n_judgments": len(judgments),
            "avg_output_chars": self._avg_output_chars(answers),
        }
        for category, group in judgments_df.groupby("category"):
            category_scores = pd.to_numeric(group["score"], errors="coerce")
            metrics[f"{category}_score"] = float(category_scores.mean())

        return pd.DataFrame.from_dict({model_name: metrics}, orient="index")

    def _log_metrics(
        self,
        leaderboard: pd.DataFrame,
        model: BaseModel,
        epoch: Optional[int],
    ) -> None:
        if self.wandb_run is None or not self.wandb_run.enabled:
            return

        generator_config = "unknown"
        if getattr(model, "generator", None) is not None:
            generator_config = model.generator.get_short_name()
        metric_prefix = f"eval/mt_bench/{generator_config}"
        metrics = leaderboard.iloc[0].to_dict()
        log_dict = {f"{metric_prefix}/{key}": value for key, value in metrics.items()}
        if epoch is not None:
            log_dict["eval/epoch"] = epoch
        self.wandb_run.log(log_dict)

    def _log_generation_metrics(
        self,
        metrics: dict[str, Any],
        model: BaseModel,
        epoch: Optional[int],
    ) -> None:
        if not metrics or self.wandb_run is None or not self.wandb_run.enabled:
            return
        generator_config = model.generator.get_short_name()
        metric_prefix = f"eval/mt_bench/{generator_config}"
        log_dict = {f"{metric_prefix}/{key}": value for key, value in metrics.items()}
        if epoch is not None:
            log_dict["eval/epoch"] = epoch
        self.wandb_run.log(log_dict)

    def _load_questions(self) -> list[MTBenchQuestion]:
        records = self._load_jsonl(self.questions_file)
        questions = [
            MTBenchQuestion(
                question_id=int(record["question_id"]),
                category=str(record.get("category", "unknown")),
                turns=list(record["turns"]),
            )
            for record in records
        ]
        for question in questions:
            if len(question.turns) != 2:
                raise ValueError(
                    "MT-Bench evaluator expects exactly two turns per question; "
                    f"question {question.question_id} has {len(question.turns)}"
                )
        if self.num_samples is not None and self.num_samples > 0:
            questions = questions[: self.num_samples]
        logger.info("Loaded %d MT-Bench questions from %s", len(questions), self.questions_file)
        return questions

    def _load_judge_prompts(self) -> dict[str, JudgePromptTemplate]:
        if self.judge_prompts_file is None:
            return self._default_judge_prompts()

        prompts = {}
        for record in self._load_jsonl(self.judge_prompts_file):
            if record.get("type") != "single":
                continue
            prompts[str(record["name"])] = JudgePromptTemplate(
                name=str(record["name"]),
                system_prompt=str(record["system_prompt"]),
                prompt_template=str(record["prompt_template"]),
            )
        defaults = self._default_judge_prompts()
        defaults.update(prompts)
        return defaults

    def _load_reference_answers(self) -> dict[int, list[str]]:
        if self.reference_answers_file is None:
            return {}
        answers = {}
        for record in self._load_jsonl(self.reference_answers_file):
            answers[int(record["question_id"])] = list(record["choices"][0]["turns"])
        return answers

    def _get_reference_answer(self, question_id: int, turn_index: int) -> str:
        if question_id not in self.reference_answers:
            return ""
        reference_turns = self.reference_answers[question_id]
        if turn_index >= len(reference_turns):
            return ""
        return reference_turns[turn_index]

    @staticmethod
    def parse_score(completion: str) -> Optional[float]:
        match = _SCORE_PATTERN.search(completion)
        if match is None:
            return None
        return float(match.group(1))

    @staticmethod
    def _outputs_by_prompt(outputs: Sequence[dict[str, Any]]) -> dict[str, str]:
        return {
            MTBenchEvaluator._prompt_key(output["prompt"]): str(output["output"])
            for output in outputs
        }

    @staticmethod
    def _prompt_key(prompt: Any) -> str:
        return json.dumps(prompt, sort_keys=True, ensure_ascii=False)

    @staticmethod
    def _answer_id(model_name: str, question_id: int, turns: Sequence[str]) -> str:
        value = json.dumps(
            {"model": model_name, "question_id": question_id, "turns": list(turns)},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _avg_output_chars(answers: list[dict[str, Any]]) -> float:
        lengths = []
        for answer in answers:
            for turn in answer["choices"][0]["turns"]:
                lengths.append(len(str(turn)))
        if not lengths:
            return 0.0
        return float(sum(lengths) / len(lengths))

    @staticmethod
    def _init_judge(judge: BaseJudge | DictConfig | Mapping[str, Any]) -> BaseJudge:
        if isinstance(judge, BaseJudge):
            return judge
        if isinstance(judge, DictConfig):
            return instantiate(judge)
        if "_target_" in judge:
            return instantiate(OmegaConf.create(dict(judge)))
        raise ValueError("judge must be a BaseJudge or Hydra config with _target_")

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    @staticmethod
    def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _default_judge_prompts() -> dict[str, JudgePromptTemplate]:
        return {
            "single-v1": JudgePromptTemplate(
                name="single-v1",
                system_prompt="You are a helpful assistant.",
                prompt_template=(
                    "[Instruction]\n"
                    "Please act as an impartial judge and evaluate the quality "
                    "of the response provided by an AI assistant to the user "
                    "question displayed below. Your evaluation should consider "
                    "helpfulness, relevance, accuracy, depth, creativity, and "
                    "level of detail. Begin with a short explanation. Be as "
                    "objective as possible. After the explanation, rate the "
                    "response from 1 to 10 by replacing rating with a number "
                    "using exactly this format: \"[[rating]]\", for example: "
                    "\"Rating: [[5]]\".\n\n"
                    "[Question]\n{question}\n\n"
                    "[The Start of Assistant's Answer]\n{answer}\n"
                    "[The End of Assistant's Answer]"
                ),
            ),
            "single-v1-multi-turn": JudgePromptTemplate(
                name="single-v1-multi-turn",
                system_prompt=(
                    "Please act as an impartial judge and evaluate the quality "
                    "of the response provided by an AI assistant. Focus on the "
                    "assistant's answer to the second user question. After the "
                    "explanation, rate the response from 1 to 10 by replacing "
                    "rating with a number using exactly this format: "
                    "\"[[rating]]\", for example: \"Rating: [[5]]\"."
                ),
                prompt_template=(
                    "<|The Start of Assistant A's Conversation with User|>\n\n"
                    "### User:\n{question_1}\n\n"
                    "### Assistant A:\n{answer_1}\n\n"
                    "### User:\n{question_2}\n\n"
                    "### Assistant A:\n{answer_2}\n\n"
                    "<|The End of Assistant A's Conversation with User|>"
                ),
            ),
            "single-math-v1": JudgePromptTemplate(
                name="single-math-v1",
                system_prompt="You are a helpful assistant.",
                prompt_template=(
                    "[Instruction]\n"
                    "Please act as an impartial judge and evaluate correctness "
                    "and helpfulness. Compare the assistant's answer with the "
                    "reference answer. After the explanation, rate the response "
                    "from 1 to 10 by replacing rating with a number using "
                    "exactly this format: \"[[rating]]\", for example: "
                    "\"Rating: [[5]]\".\n\n"
                    "[Question]\n{question}\n\n"
                    "[The Start of Reference Answer]\n{ref_answer_1}\n"
                    "[The End of Reference Answer]\n\n"
                    "[The Start of Assistant's Answer]\n{answer}\n"
                    "[The End of Assistant's Answer]"
                ),
            ),
            "single-math-v1-multi-turn": JudgePromptTemplate(
                name="single-math-v1-multi-turn",
                system_prompt=(
                    "Please act as an impartial judge and evaluate correctness "
                    "and helpfulness. Focus on the assistant's answer to the "
                    "second question. After the explanation, rate the response "
                    "from 1 to 10 by replacing rating with a number using "
                    "exactly this format: \"[[rating]]\", for example: "
                    "\"Rating: [[5]]\"."
                ),
                prompt_template=(
                    "<|The Start of Reference Answer|>\n\n"
                    "### User:\n{question_1}\n\n"
                    "### Reference answer:\n{ref_answer_1}\n\n"
                    "### User:\n{question_2}\n\n"
                    "### Reference answer:\n{ref_answer_2}\n\n"
                    "<|The End of Reference Answer|>\n\n\n"
                    "<|The Start of Assistant A's Conversation with User|>\n\n"
                    "### User:\n{question_1}\n\n"
                    "### Assistant A:\n{answer_1}\n\n"
                    "### User:\n{question_2}\n\n"
                    "### Assistant A:\n{answer_2}\n\n"
                    "<|The End of Assistant A's Conversation with User|>"
                ),
            ),
        }
