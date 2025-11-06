from typing import Optional

import torch

from .logger import Logger


class DeviceManager:
    """
    A general-purpose device management class that handles device detection,
    dtype selection, and GPU assignment strategies.
    """

    def __init__(
        self,
        policy_cuda_index: int = 0,
        ref_cuda_index: Optional[int] = None,
        parallel: bool = False,
        logger: Optional[Logger] = None,
    ):
        """
        Initialize the device manager.

        Args:
            policy_cuda_index: GPU index for the policy/trainable model.
            ref_cuda_index: GPU index for the reference model. If None, auto-assigned.
            parallel: Whether running in parallel training mode.
            logger: Optional logger instance for device info messages.
        """
        self.policy_cuda_index = policy_cuda_index
        self.ref_cuda_index = ref_cuda_index
        self.parallel = parallel
        self.logger = logger

        self._device = None
        self._ref_device = None
        self._dtype = None
        self._num_available_gpus = 0

        self._setup_devices()

    def _setup_devices(self):
        """Setup devices based on available hardware."""
        if torch.cuda.is_available():
            self._num_available_gpus = torch.cuda.device_count()
            self._setup_cuda_devices()
            self._dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            self._device = "mps"
            self._ref_device = "mps"
            self._dtype = torch.float16
            if self.logger:
                self.logger.warning(
                    "Using MPS backend. Note: BFloat16 is not supported on MPS, using Float16."
                )
        else:
            self._device = "cpu"
            self._ref_device = "cpu"
            self._dtype = torch.float32

    def _setup_cuda_devices(self):
        """Setup CUDA devices with intelligent GPU assignment."""
        if self.parallel:
            self._device = f"cuda:{self.policy_cuda_index}"
            self._ref_device = (
                f"cuda:{self.ref_cuda_index}"
                if self.ref_cuda_index is not None
                else self._device
            )
        else:
            policy_gpu = (
                self.policy_cuda_index
                if self.policy_cuda_index < self._num_available_gpus
                else 0
            )

            if self.ref_cuda_index is not None:
                ref_gpu = (
                    self.ref_cuda_index
                    if self.ref_cuda_index < self._num_available_gpus
                    else policy_gpu
                )
            elif self._num_available_gpus > 1:
                ref_gpu = (policy_gpu + 1) % self._num_available_gpus
            else:
                ref_gpu = policy_gpu

            self._device = f"cuda:{policy_gpu}"
            self._ref_device = f"cuda:{ref_gpu}"

            if self.logger:
                self.logger.info(f"Auto-detected {self._num_available_gpus} GPU(s)")
                self.logger.info(
                    f"Sequential training mode: Policy model on GPU {policy_gpu}, "
                    f"Reference model on GPU {ref_gpu}"
                )

    @property
    def device(self) -> str:
        """Get the device string for the policy model."""
        return self._device

    @property
    def ref_device(self) -> str:
        """Get the device string for the reference model."""
        return self._ref_device

    @property
    def dtype(self) -> torch.dtype:
        """Get the appropriate dtype for the current device."""
        return self._dtype

    @property
    def num_available_gpus(self) -> int:
        """Get the number of available GPUs."""
        return self._num_available_gpus

    def get_device_for_model(self, model_idx: int, available_gpus: list) -> str:
        """
        Get device string for a specific model in parallel training.

        Args:
            model_idx: Index of the model in the ensemble.
            available_gpus: List of available GPU IDs.

        Returns:
            Device string (e.g., "cuda:0").
        """
        if torch.cuda.is_available() and available_gpus:
            gpu_id = available_gpus[model_idx % len(available_gpus)]
            return f"cuda:{gpu_id}"
        return self._device

    def log_device_info(self):
        """Log device configuration information."""
        if self.logger:
            self.logger.info(
                f"Selected device for policy model: {self.device} with dtype: {self.dtype}"
            )
            self.logger.info(f"Selected device for reference model: {self.ref_device}")

    @classmethod
    def get_available_gpus(cls) -> list:
        """
        Get list of available GPU IDs.

        Returns:
            List of GPU indices (e.g., [0, 1, 2, 3]).
        """
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        return []
