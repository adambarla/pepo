import contextlib
import logging
import queue
import threading
from typing import Iterator, List, Optional, Union

import torch

logger = logging.getLogger(__name__)

# Singleton instance
_instance: Optional["DeviceManager"] = None


def move_to_device(
    model: torch.nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    """Move model and optimizer state to device.

    Args:
        model: The model to move.
        device: Target device.
        optimizer: Optional optimizer whose state should also be moved.
    """
    model.to(device)
    if optimizer is not None:
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)


def init_device_manager(
    gpu_ids: Optional[List[int]] = None,
    dtype: Union[str, torch.dtype] = torch.bfloat16,
) -> "DeviceManager":
    """Initialize the global DeviceManager singleton.

    Args:
        gpu_ids: Optional list of GPU IDs. If None, uses all available GPUs.
        dtype: Data type for models (e.g., torch.bfloat16 or "bfloat16").

    Returns:
        The initialized DeviceManager instance.
    """
    global _instance
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    # Cast to ensure type is torch.dtype after potential string conversion
    dtype_resolved: torch.dtype = dtype
    _instance = DeviceManager(gpu_ids=gpu_ids, dtype=dtype_resolved)
    return _instance


def get_device_manager() -> "DeviceManager":
    """Get the global DeviceManager singleton.

    Raises:
        RuntimeError: If init_device_manager was not called first.
    """
    if _instance is None:
        raise RuntimeError(
            "DeviceManager not initialized. Call init_device_manager() first."
        )
    return _instance


class DeviceManager:
    """
    Device manager for distributing models across GPUs.
    Supports dynamic GPU acquisition via request_gpu() context manager.
    """

    def __init__(
        self,
        gpu_ids: Optional[List[int]] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize the device manager.

        Args:
            gpu_ids: Optional list of GPU IDs to use. If None, uses all available GPUs.
        """
        self.gpu_ids = gpu_ids
        self.dtype = dtype

        self._available_gpus = self._get_available_gpus()

        # Semaphore limits concurrent GPU access to number of GPUs
        self._gpu_semaphore = threading.Semaphore(value=len(self._available_gpus))
        # Queue tracks which GPUs are free (FIFO for fairness)
        self._gpu_queue: queue.Queue[int] = queue.Queue()
        for gpu_id in self._available_gpus:
            self._gpu_queue.put(gpu_id)

        self._log_environment_info()

    def _get_available_gpus(self) -> List[int]:
        """Get list of available GPU IDs."""
        if not torch.cuda.is_available():
            logger.error("CUDA is not available!")
            raise RuntimeError("CUDA is not available")

        num_gpus = torch.cuda.device_count()
        available_gpus = [i for i in range(num_gpus)]

        if self.gpu_ids is not None:
            for gpu_id in self.gpu_ids:
                if gpu_id not in available_gpus:
                    raise ValueError(
                        f"Invalid GPU ID: {gpu_id}. Available GPUs: {available_gpus}"
                    )
            gpu_ids = list(set(self.gpu_ids))
            gpu_ids.sort()
            return gpu_ids

        return available_gpus

    def get_device_for_model(self, model_idx: int) -> str:
        """
        Get device string for a specific model, round-robin spread across
        available GPUs.
        """
        gpu_id = self._get_gpu_id_for_model(model_idx)
        gpu_device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        return gpu_device

    def _get_gpu_id_for_model(self, model_idx: int) -> int:
        return self._available_gpus[model_idx % len(self._available_gpus)]

    @contextlib.contextmanager
    def request_gpu(self) -> Iterator[torch.device]:
        self._gpu_semaphore.acquire()
        gpu_id = self._gpu_queue.get()

        try:
            yield torch.device(f"cuda:{gpu_id}")
        finally:
            try:
                with torch.cuda.device(gpu_id):
                    torch.cuda.synchronize()
            except Exception:
                pass
            finally:
                self._gpu_queue.put(gpu_id)
                self._gpu_semaphore.release()

    @property
    def num_available_gpus(self) -> int:
        """Get the number of available GPUs."""
        return len(self._available_gpus)

    def clear_cache(self, model_idx: Optional[int] = None) -> None:
        """
        Clear CUDA cache on all managed GPUs.
        Useful after unloading models to free up GPU memory.
        """
        if not torch.cuda.is_available():
            return
        if model_idx is not None:
            gpu_id = self._get_gpu_id_for_model(model_idx)
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
            return
        for gpu_id in self._available_gpus:
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
        torch.cuda.empty_cache()

    def _log_environment_info(self) -> None:
        """Log device assignment information."""
        logger.info("Device Manager initialized:")
        logger.info(f"Available GPUs: {self._available_gpus}")
        logger.info(f"Dtype: {self.dtype}")
        for gpu_id in self._available_gpus:
            props = torch.cuda.get_device_properties(gpu_id)
            memory_gb = props.total_memory / (1024**3)
            logger.info(f"GPU {gpu_id}: {props.name}, {memory_gb:.1f} GB total memory")
