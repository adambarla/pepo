from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Optional, Sequence
from urllib import error, request

from .base import BaseJudge, JudgePrompt

logger = logging.getLogger(__name__)


class ManagedVLLMJudge(BaseJudge):
    """Lifecycle-managed vLLM OpenAI-compatible judge backend."""

    def __init__(
        self,
        model_name: str,
        served_model_name: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 8000,
        api_key: str = "EMPTY",
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        tensor_parallel_size: int = 1,
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
        command: Optional[list[str]] = None,
    ) -> None:
        self.model_name = model_name
        self.served_model_name = served_model_name or model_name
        self.host = host
        self.port = port
        self.api_key = api_key
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
        self.command = command
        self._process: Optional[subprocess.Popen[Any]] = None
        self._log_handle: Any = None

    def generate(self, prompts: Sequence[JudgePrompt]) -> list[str]:
        if not prompts:
            return []
        self._ensure_server()
        return [self._chat_completion(prompt) for prompt in prompts]

    def unload(self) -> None:
        if self._process is None:
            self._close_log_handle()
            return

        logger.info("Stopping vLLM judge server on port %d", self.port)
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

        command = self._build_command()
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
                raise RuntimeError(
                    "vLLM judge server exited during startup with code "
                    f"{self._process.returncode}"
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
            f"http://{self.host}:{self.port}{path}",
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
                    port=self.port,
                    api_key=self.api_key,
                )
                for part in self.command
            ]

        command = [
            "vllm",
            "serve",
            self.model_name,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--served-model-name",
            self.served_model_name,
            "--api-key",
            self.api_key,
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
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

    def _close_log_handle(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
