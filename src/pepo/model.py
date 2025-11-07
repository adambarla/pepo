from omegaconf import DictConfig

from .utils import DeviceManager, Logger


class PEPOModel:
    def __init__(
        self,
        pepo_config: DictConfig,
        full_config: DictConfig,
        logger: Logger,
        device_manager: DeviceManager,
    ):
        """
        Initialize PEPO Model.

        Args:
            pepo_config: PEPO-specific configuration (cfg.pepo).
            full_config: Full configuration for accessing other sections.
            logger: Logger instance.
            device_manager: Device manager instance.
        """
        self.pepo_config = pepo_config
        self.config = full_config
        self.logger = logger
        self.device_manager = device_manager

        self.alpha = pepo_config.alpha
        self.beta = pepo_config.beta

        self.logger.info(
            f"PEPOModel initialized with alpha={self.alpha}, beta={self.beta}"
        )

    def train(self):
        """
        Train the PEPO ensemble models and save the models to the hub.
        """
        self.logger.info("Training PEPO ensemble models...")
