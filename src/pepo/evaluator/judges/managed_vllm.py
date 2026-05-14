from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Optional, Sequence
from urllib import error, request

from tqdm.auto import tqdm

from ...utils import get_device_manager
from .base import BaseJudge, JudgePrompt

logger = logging.getLogger(__name__)


class ManagedVLLMJudge(BaseJudge):
    """Lifecycle-managed vLLM OpenAI-compatible judge backend."""

    def __init__(
        self,
        model_name: str,
        served_model_name: Optional[str] = None,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        api_key: str = "EMPTY",
        vllm_executable: str = "vllm",
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        tensor_parallel_size: Optional[int] = None,
        dtype: str = "auto",
        gpu_memory_utilization: Optional[float] = None,
        max_model_len: Optional[int] = None,
        trust_remote_code: bool = False,
        startup_timeout: float = 1800.0,
        request_timeout: float = 600.0,
        shutdown_timeout: float = 60.0,
        poll_interval: float = 5.0,
        extra_args: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
        log_path: Optional[str] = None,
        log_dir: str = "outputs/mt_bench",
        command: Optional[list[str]] = None,
    ) -> None:
        self.model_name = model_name
        self.served_model_name = served_model_name or model_name
        self.host = host
        self.port = port
        self.api_key = api_key
        self.vllm_executable = vllm_executable
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.shutdown_timeout = shutdown_timeout
        self.poll_interval = poll_interval
        self.extra_args = extra_args or []
        self.env = env or {}
        self.log_path = Path(log_path) if log_path else None
        self.log_dir = Path(log_dir)
        self.command = command
        self._process: Optional[subprocess.Popen[Any]] = None
        self._log_handle: Any = None

    def generate(self, prompts: Sequence[JudgePrompt]) -> list[str]:
        if not prompts:
            return []
        self._ensure_server()
        return [
            self._chat_completion(prompt)
            for prompt in tqdm(
                prompts,
                desc="Judging MT-Bench",
                unit="prompt",
                leave=False,
            )
        ]

    def unload(self) -> None:
        if self._process is None:
            self._close_log_handle()
            return

        logger.info("Stopping vLLM judge server on port %d", self._resolve_port())
        self._process.terminate()
        try:
            self._process.wait(timeout=self.shutdown_timeout)
        except subprocess.TimeoutExpired:
            logger.warning("vLLM judge server did not stop; killing process")
            self._process.kill()
            self._process.wait(timeout=self.shutdown_timeout)
        finally:
            self._process = None
            self._close_log_handle()

    def _ensure_server(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return

        self._resolve_runtime_paths()
        command = self._build_command()
        self._validate_runtime(command)
        logger.info("Starting vLLM judge server: %s", " ".join(command))
        stdout = subprocess.DEVNULL
        stderr = subprocess.DEVNULL
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = open(self.log_path, "a", encoding="utf-8")
            stdout = self._log_handle
            stderr = subprocess.STDOUT

        env = os.environ.copy()
        env.update(self.env)
        self._process = subprocess.Popen(  # noqa: S603
            command,
            stdout=stdout,
            stderr=stderr,
            env=env,
        )
        self._wait_until_ready()

    def _wait_until_ready(self) -> None:
        start_time = time.monotonic()
        while time.monotonic() - start_time < self.startup_timeout:
            if self._process is not None and self._process.poll() is not None:
                log_tail = self._read_log_tail()
                raise RuntimeError(
                    "vLLM judge server exited during startup with code "
                    f"{self._process.returncode}"
                    + (f"\nRecent vLLM log lines:\n{log_tail}" if log_tail else "")
                )
            if self._is_ready():
                logger.info("vLLM judge server is ready")
                return
            time.sleep(self.poll_interval)
        raise TimeoutError(
            f"vLLM judge server did not become ready within {self.startup_timeout}s"
        )

    def _is_ready(self) -> bool:
        try:
            self._request_json("GET", "/v1/models")
            return True
        except (OSError, TimeoutError, error.URLError, error.HTTPError):
            return False

    def _chat_completion(self, prompt: JudgePrompt) -> str:
        payload = self._build_chat_payload(prompt)
        response_payload = self._request_json("POST", "/v1/chat/completions", payload)
        choices = response_payload.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return str(message.get("content", "")).strip()

    def _build_chat_payload(self, prompt: JudgePrompt) -> dict[str, Any]:
        messages = []
        if prompt.system_prompt:
            messages.append({"role": "system", "content": prompt.system_prompt})
        messages.append({"role": "user", "content": prompt.user_prompt})
        return {
            "model": self.served_model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        data = None
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")

        req = request.Request(
            f"http://{self.host}:{self._resolve_port()}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        with request.urlopen(req, timeout=self.request_timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _build_command(self) -> list[str]:
        if self.command is not None:
            return [
                part.format(
                    model_name=self.model_name,
                    served_model_name=self.served_model_name,
                    host=self.host,
                    port=self._resolve_port(),
                    api_key=self.api_key,
                )
                for part in self.command
            ]

        command = [
            self.vllm_executable,
            "serve",
            self.model_name,
            "--host",
            self.host,
            "--port",
            str(self._resolve_port()),
            "--served-model-name",
            self.served_model_name,
            "--api-key",
            self.api_key,
            "--tensor-parallel-size",
            str(self._resolve_tensor_parallel_size()),
            "--dtype",
            self.dtype,
        ]
        if self.gpu_memory_utilization is not None:
            command.extend(
                ["--gpu-memory-utilization", str(self.gpu_memory_utilization)]
            )
        if self.max_model_len is not None:
            command.extend(["--max-model-len", str(self.max_model_len)])
        if self.trust_remote_code:
            command.append("--trust-remote-code")
        command.extend(self.extra_args)
        return command

    def _resolve_runtime_paths(self) -> None:
        self._resolve_port()
        if self.log_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.log_path = self.log_dir / (
                f"vllm_judge_{timestamp}_{os.getpid()}_p{self.port}.log"
            )

    def _resolve_port(self) -> int:
        if self.port is not None and self.port > 0:
            return self.port

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((self.host, 0))
            self.port = sock.getsockname()[1]
        return self.port

    def _validate_runtime(self, command: Sequence[str]) -> None:
        if not command:
            raise ValueError("vLLM judge command is empty")

        executable = command[0]
        if os.path.sep not in executable and shutil.which(executable) is None:
            raise RuntimeError(
                "Could not find the vLLM executable "
                f"{executable!r} on PATH. Install vLLM in this environment, load the "
                "cluster module that provides it, or set "
                "evaluator.judge.vllm_executable to the vLLM executable path."
            )

        visible_gpu_count = self._visible_gpu_count()
        tensor_parallel_size = self._resolve_tensor_parallel_size()
        if visible_gpu_count is not None and tensor_parallel_size > visible_gpu_count:
            raise RuntimeError(
                "vLLM judge tensor_parallel_size="
                f"{tensor_parallel_size} but only {visible_gpu_count} GPU(s) "
                "are visible through CUDA_VISIBLE_DEVICES. Expose enough GPUs for "
                "the judge model or override evaluator.judge.tensor_parallel_size "
                "and evaluator.judge.model_name."
            )

    def _resolve_tensor_parallel_size(self) -> int:
        if self.tensor_parallel_size is not None:
            return self.tensor_parallel_size
        try:
            return get_device_manager().num_available_gpus
        except RuntimeError:
            visible_gpu_count = self._visible_gpu_count()
            if visible_gpu_count is not None:
                return max(1, visible_gpu_count)
            return 1

    @staticmethod
    def _visible_gpu_count() -> Optional[int]:
        value = os.environ.get("CUDA_VISIBLE_DEVICES")
        if value is None or value.strip() == "":
            return None
        if value.strip() == "-1":
            return 0
        return len([item for item in value.split(",") if item.strip()])

    def _close_log_handle(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def _read_log_tail(self, max_lines: int = 80) -> str:
        if self.log_path is None or not self.log_path.exists():
            return ""
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as handle:
                lines = handle.readlines()
        except OSError:
            return ""
        return "".join(lines[-max_lines:]).strip()
