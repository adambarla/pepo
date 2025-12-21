import os
import random
import re

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Additional settings for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def sanitize_filename(name: str) -> str:
    """
    Sanitize a string to be used as a filename.
    Replaces invalid characters with underscores.

    Args:
        name: String to sanitize.

    Returns:
        Sanitized string safe for use as a filename.
    """
    name = re.sub(r'[<>:"/\\|?*]', "-", name)
    name = re.sub(r"\s+", "-", name)
    return name


def strip_hydra_targets(obj: object) -> object:
    """
    Recursively strip Hydra _target_ and _partial_ keys from a config dictionary.

    This is useful when passing resolved configs to functions that shouldn't
    trigger Hydra's recursive instantiation (e.g., when logging configs to WandB).

    Args:
        obj: The object to process (typically a dict from OmegaConf.to_container()).

    Returns:
        A new object with all _target_ and _partial_ keys removed.
    """
    if isinstance(obj, dict):
        return {
            k: strip_hydra_targets(v)
            for k, v in obj.items()
            if k != "_target_" and k != "_partial_"
        }
    elif isinstance(obj, list):
        return [strip_hydra_targets(item) for item in obj]
    return obj
