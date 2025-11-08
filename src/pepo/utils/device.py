from typing import List, Optional

import torch

from .logger import Logger


class DeviceManager:
    """
    Device manager for distributing models across GPUs.
    Supports one GPU per model (simple approach).
    """

    def __init__(
        self,
        gpu_ids: Optional[List[int]] = None,
        logger: Optional[Logger] = None,
    ):
        """
        Initialize the device manager.

        Args:
            gpu_ids: Optional list of GPU IDs to use. If None, uses all available GPUs.
            logger: Optional logger instance for device info messages.
        """
        self.gpu_ids = gpu_ids
        self.logger = logger

        self._available_gpus = self._get_available_gpus()

        if self.logger:
            self._log_environment_info()

    def _get_available_gpus(self) -> List[int]:
        """Get list of available GPU IDs."""
        if not torch.cuda.is_available():
            if self.logger:
                self.logger.error("CUDA is not available!")
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
        Get device string for a specific model, round-robin spread across available GPUs.
        """
        gpu_id = self._available_gpus[model_idx % len(self._available_gpus)]
        gpu_device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        if self.logger:
            self.logger.info(f"Model {model_idx} assigned to GPU {gpu_device}")
        return gpu_device

    @property
    def dtype(self) -> torch.dtype:
        """Get the appropriate dtype for the current device."""
        if torch.cuda.is_available():
            return torch.bfloat16
        elif torch.backends.mps.is_available():
            return torch.float16
        else:
            return torch.float32

    @property
    def num_available_gpus(self) -> int:
        """Get the number of available GPUs."""
        return len(self._available_gpus)

    def _log_environment_info(self):
        """Log device assignment information."""
        self.logger.info("Device Manager initialized:")
        self.logger.info(f"  Available GPUs: {self._available_gpus}")
        self.logger.info(f"  Dtype: {self.dtype}")
        for gpu_id in self._available_gpus:
            props = torch.cuda.get_device_properties(gpu_id)
            memory_gb = props.total_memory / (1024**3)
            self.logger.info(
                f"  GPU {gpu_id}: {props.name}, {memory_gb:.1f} GB total memory"
            )
