import torch

from pepo.evaluator.judges import JudgePrompt, LocalHFJudge, ManagedVLLMJudge


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


def test_managed_vllm_judge_builds_serve_command() -> None:
    judge = ManagedVLLMJudge(
        model_name="meta-llama/Meta-Llama-3-70B-Instruct",
        served_model_name="mtbench-judge",
        host="0.0.0.0",
        port=8123,
        api_key="EMPTY",
        tensor_parallel_size=8,
        dtype="float16",
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        extra_args=["--disable-log-requests"],
    )

    command = judge._build_command()

    assert command[:3] == [
        "vllm",
        "serve",
        "meta-llama/Meta-Llama-3-70B-Instruct",
    ]
    assert "--served-model-name" in command
    assert "mtbench-judge" in command
    assert "--tensor-parallel-size" in command
    assert "8" in command
    assert "--disable-log-requests" in command


def test_managed_vllm_judge_builds_chat_payload() -> None:
    judge = ManagedVLLMJudge(
        model_name="meta-llama/Meta-Llama-3-70B-Instruct",
        served_model_name="mtbench-judge",
        temperature=0.0,
        top_p=0.9,
        max_tokens=64,
    )

    payload = judge._build_chat_payload(
        JudgePrompt(
            system_prompt="You are an impartial judge.",
            user_prompt="Rate this answer.",
        )
    )

    assert payload["model"] == "mtbench-judge"
    assert payload["temperature"] == 0.0
    assert payload["top_p"] == 0.9
    assert payload["max_tokens"] == 64
    assert payload["messages"] == [
        {"role": "system", "content": "You are an impartial judge."},
        {"role": "user", "content": "Rate this answer."},
    ]
