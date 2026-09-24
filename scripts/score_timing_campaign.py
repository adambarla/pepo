"""Score one full AlpacaEval response file against a shared reference set."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import alpaca_eval
import pandas as pd
import yaml
from alpaca_eval.constants import EVALUATORS_CONFIG_DIR

ANNOTATOR_CONFIG = {
    "llama3_70b_judge": {
        "prompt_template": str(
            next((EVALUATORS_CONFIG_DIR / "alpaca_eval_llama3_70b_fn").glob("*.txt"))
        ),
        "completions_kwargs": {
            "batch_size": 1,
            "max_new_tokens": 512,
            "model_name": "meta-llama/Meta-Llama-3-70B-Instruct",
            "model_kwargs": {"dtype": "float16", "device_map": "auto"},
        },
        "fn_completions": "huggingface_local_completions",
        "fn_completion_parser": "ranking_parser",
    }
}


def _count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return len(json.load(handle))


def _existing_leaderboard(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob("*_leaderboard.csv"))
    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-outputs", type=Path, required=True)
    parser.add_argument("--reference-outputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--expected-count", type=int, default=805)
    args = parser.parse_args()

    if not args.model_outputs.exists():
        raise FileNotFoundError(args.model_outputs)
    if not args.reference_outputs.exists():
        raise FileNotFoundError(args.reference_outputs)

    model_count = _count(args.model_outputs)
    reference_count = _count(args.reference_outputs)
    if model_count != args.expected_count or reference_count != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} responses, got "
            f"model={model_count}, reference={reference_count}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = _existing_leaderboard(args.output_dir)
    if existing is not None:
        leaderboard = pd.read_csv(existing)
        print(f"Reusing existing leaderboard: {existing}")
        print(leaderboard.to_string(index=False))
        return

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as config_file:
        yaml.safe_dump(ANNOTATOR_CONFIG, config_file)
        config_file.flush()
        leaderboard, _ = alpaca_eval.evaluate(  # ty: ignore[unresolved-attribute]
            model_outputs=str(args.model_outputs),
            reference_outputs=str(args.reference_outputs),
            annotators_config=config_file.name,
            name=args.name,
            output_path=str(args.output_dir),
            precomputed_leaderboard=None,
            is_overwrite_leaderboard=False,
            is_return_instead_of_print=True,
        )

    print(f"{args.name}:")
    print(leaderboard.loc[args.name].to_string())


if __name__ == "__main__":
    main()
