from typing import Any, Dict, List

import torch

from .base import DataCollator


class RewardDataCollator(DataCollator):
    """Collator that also collects reward ensemble columns."""

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(features)
        for s in ["chosen", "rejected"]:
            ks = sorted(
                k for k in features[0] if k.startswith("rewards_") and k.endswith(s)
            )
            if ks:
                batch[f"rewards_{s}"] = torch.tensor(
                    [[f[k] for k in ks] for f in features], dtype=torch.float
                )
        return batch
