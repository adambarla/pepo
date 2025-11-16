import os
import random
import re

import numpy as np
import torch


def set_seed(seed: int):
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
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name
