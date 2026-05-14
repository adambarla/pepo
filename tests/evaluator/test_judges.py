import pytest

from pepo.evaluator.judges import JudgePrompt, ManagedVLLMJudge


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
        extra_args=["--disable-log-requests", "--disable-custom-all-reduce"],
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
    assert "--disable-custom-all-reduce" in command


def test_managed_vllm_judge_uses_configured_executable() -> None:
    judge = ManagedVLLMJudge(
        model_name="judge",
        vllm_executable="/opt/vllm/bin/vllm",
    )

    command = judge._build_command()

    assert command[:3] == ["/opt/vllm/bin/vllm", "serve", "judge"]


def test_managed_vllm_judge_defaults_tensor_parallel_to_device_manager(
    monkeypatch,
) -> None:
    class FakeDeviceManager:
        num_available_gpus = 3

    monkeypatch.setattr(
        "pepo.evaluator.judges.managed_vllm.get_device_manager",
        lambda: FakeDeviceManager(),
    )
    judge = ManagedVLLMJudge(model_name="judge")

    command = judge._build_command()

    assert command[command.index("--tensor-parallel-size") + 1] == "3"


def test_managed_vllm_judge_resolves_automatic_port_and_log_path(tmp_path) -> None:
    judge = ManagedVLLMJudge(model_name="judge", port=None, log_dir=str(tmp_path))

    judge._resolve_runtime_paths()

    assert isinstance(judge.port, int)
    assert judge.port > 0
    assert judge.log_path is not None
    assert judge.log_path.parent == tmp_path
    assert f"_p{judge.port}.log" in judge.log_path.name


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


def test_managed_vllm_judge_validates_missing_executable(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda executable: None)
    judge = ManagedVLLMJudge(model_name="judge")

    with pytest.raises(RuntimeError, match="Could not find the vLLM executable"):
        judge._validate_runtime(["vllm", "serve", "judge"])


def test_managed_vllm_judge_validates_visible_gpu_count(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda executable: "/usr/bin/vllm")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    judge = ManagedVLLMJudge(model_name="judge", tensor_parallel_size=8)

    with pytest.raises(RuntimeError, match="tensor_parallel_size=8"):
        judge._validate_runtime(["vllm", "serve", "judge"])


def test_managed_vllm_judge_reads_log_tail(tmp_path) -> None:
    log_path = tmp_path / "vllm.log"
    log_path.write_text(
        "\n".join(f"line {idx}" for idx in range(100)), encoding="utf-8"
    )
    judge = ManagedVLLMJudge(model_name="judge", log_path=str(log_path))

    tail = judge._read_log_tail(max_lines=3)

    assert tail == "line 97\nline 98\nline 99"
