"""Score existing AlpacaEval2 responses against one shared reference set."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import alpaca_eval
import yaml
from alpaca_eval.constants import EVALUATORS_CONFIG_DIR

BASE = "Llama-3.1-Tulu-3-8B-SFT-a0.1-b0.1-L4"
REFERENCE = "Llama-3.1-Tulu-3-8B-SFT-a0.0-b0.1-L1-e0_ns128_mt1024_responses.json"

RESPONSE_PATTERNS = {
    "token": BASE + "-e{epoch}_ns128_mt1024_responses.json",
    "mean_std": (
        BASE
        + "-e{epoch}_ns128_bon-mean_std-eta0.1-n128-mt1024"
        + "-top-p-t1.0-p0.9_responses.json"
    ),
    "min": (
        BASE + "-e{epoch}_ns128_bon-min-n128-mt1024-top-p-t1.0-p0.9_responses.json"
    ),
}

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


def score_one(
    model_outputs: Path,
    reference_outputs: Path,
    output_dir: Path,
    name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as config_file:
        yaml.safe_dump(ANNOTATOR_CONFIG, config_file)
        config_file.flush()
        leaderboard, _ = alpaca_eval.evaluate(  # ty: ignore[unresolved-attribute]
            model_outputs=str(model_outputs),
            reference_outputs=str(reference_outputs),
            annotators_config=config_file.name,
            name=name,
            output_path=str(output_dir),
            precomputed_leaderboard=None,
            is_overwrite_leaderboard=True,
            is_return_instead_of_print=True,
        )
    print(f"{name}:\n{leaderboard.loc[name].to_string()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampler", choices=sorted(RESPONSE_PATTERNS), required=True)
    parser.add_argument("--epochs", nargs="+", type=int, default=range(1, 7))
    parser.add_argument(
        "--response-dir", type=Path, default=Path("outputs/alpaca_eval/responses")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/alpaca_eval/common_reference")
    )
    args = parser.parse_args()

    reference = args.response_dir / REFERENCE
    if not reference.exists():
        raise FileNotFoundError(reference)

    for epoch in args.epochs:
        response_file = args.response_dir / RESPONSE_PATTERNS[args.sampler].format(
            epoch=epoch
        )
        if not response_file.exists():
            raise FileNotFoundError(response_file)
        name = f"{args.sampler}-e{epoch}-common-reference"
        score_one(
            model_outputs=response_file,
            reference_outputs=reference,
            output_dir=args.output_dir / args.sampler / f"e{epoch}",
            name=name,
        )


if __name__ == "__main__":
    main()
