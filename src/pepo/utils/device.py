import logging
from typing import List, Optional, Union

import torch

logger = logging.getLogger(__name__)

# Singleton instance
_instance: Optional["DeviceManager"] = None


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
    Supports one GPU per model (simple approach).
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
