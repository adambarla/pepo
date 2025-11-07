from typing import List, Optional

import torch
from omegaconf import DictConfig

from .logger import Logger


class DeviceManager:
    """
    Device manager for distributing models across GPUs.
    Supports one GPU per model (simple approach).
    """

    def __init__(
        self,
        device_config: DictConfig,
        num_models: int,
        logger: Optional[Logger] = None,
    ):
        """
        Initialize the device manager.

        Args:
            device_config: Device configuration from Hydra config (e.g., cfg.device).
            num_models: Number of models to distribute (e.g., ensemble size).
            logger: Optional logger instance for device info messages.
        """
        self.device_config = device_config
        self.num_models = num_models
        self.logger = logger

        self._available_gpus = self._get_available_gpus()
        self._model_devices = self._assign_devices()

        if self.logger:
            self._log_device_assignment()

    def _get_available_gpus(self) -> List[int]:
        """Get list of available GPU IDs."""
        if not torch.cuda.is_available():
            if self.logger:
                self.logger.error("CUDA is not available!")
            raise RuntimeError("CUDA is not available")

        num_gpus = torch.cuda.device_count()

        # Check if gpu_ids are specified in config
        if self.device_config.get("gpu_ids") is not None:
            gpu_ids = [int(x.strip()) for x in str(self.device_config.gpu_ids).split(",")]
            # Validate GPU IDs
            invalid = [gpu_id for gpu_id in gpu_ids if gpu_id >= num_gpus]
            if invalid:
                raise ValueError(
                    f"Invalid GPU IDs: {invalid}. Available GPUs: 0-{num_gpus-1}"
                )
            return gpu_ids

        # Otherwise use all available GPUs
        return list(range(num_gpus))

    def _assign_devices(self) -> List[str]:
        """
        Assign one GPU per model.

        Returns:
            List of device strings (e.g., ["cuda:0", "cuda:1", "cuda:2"]).

        Raises:
            RuntimeError: If not enough GPUs available.
        """
        if len(self._available_gpus) < self.num_models:
            error_msg = (
                f"Not enough GPUs available. "
                f"Need {self.num_models} GPUs, but only {len(self._available_gpus)} available. "
                f"Available GPUs: {self._available_gpus}"
            )
            if self.logger:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Assign one GPU per model (round-robin if more GPUs than models)
        devices = []
        for model_idx in range(self.num_models):
            gpu_id = self._available_gpus[model_idx % len(self._available_gpus)]
            devices.append(f"cuda:{gpu_id}")

        return devices

    def get_device_for_model(self, model_idx: int) -> str:
        """
        Get device string for a specific model.

        Args:
            model_idx: Index of the model (0 to num_models-1).

        Returns:
            Device string (e.g., "cuda:0").

        Raises:
            IndexError: If model_idx is out of range.
        """
        if model_idx < 0 or model_idx >= self.num_models:
            raise IndexError(
                f"Model index {model_idx} out of range. "
                f"Valid range: 0 to {self.num_models - 1}"
            )
        return self._model_devices[model_idx]

    def get_all_devices(self) -> List[str]:
        """Get list of all assigned device strings."""
        return self._model_devices.copy()

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

    def _log_device_assignment(self):
        """Log device assignment information."""
        self.logger.info("Device Manager initialized:")
        self.logger.info(f"  Number of models: {self.num_models}")
        self.logger.info(f"  Available GPUs: {self._available_gpus}")
        self.logger.info(f"  Dtype: {self.dtype}")

        self.logger.info("  Model-to-GPU assignment:")
        for model_idx, device in enumerate(self._model_devices):
            self.logger.info(f"    Model {model_idx} -> {device}")

        # Log GPU memory info
        for gpu_id in self._available_gpus:
            props = torch.cuda.get_device_properties(gpu_id)
            memory_gb = props.total_memory / (1024**3)
            self.logger.info(
                f"  GPU {gpu_id}: {props.name}, {memory_gb:.1f} GB total memory"
            )

    def log_device_info(self):
        """Log device configuration information (for backward compatibility)."""
        self._log_device_assignment()
