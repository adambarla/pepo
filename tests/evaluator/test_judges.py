import torch

from pepo.evaluator.judges import JudgePrompt, LocalHFJudge


def test_local_hf_judge_normalizes_dtype_strings() -> None:
    kwargs = LocalHFJudge._normalize_model_kwargs(
        {"device_map": "auto", "torch_dtype": "float16", "dtype": "bfloat16"}
    )

    assert kwargs["device_map"] == "auto"
    assert kwargs["torch_dtype"] is torch.float16
    assert kwargs["dtype"] is torch.bfloat16


def test_local_hf_judge_fallback_prompt_includes_system_and_user() -> None:
    prompt = JudgePrompt(
        system_prompt="You are an impartial judge.",
        user_prompt="Rate this answer.",
    )

    formatted = LocalHFJudge._fallback_prompt(prompt)

    assert "You are an impartial judge." in formatted
    assert "User: Rate this answer." in formatted
    assert formatted.endswith("Assistant:")
