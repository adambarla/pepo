from typing import Optional

import numpy as np
from torch.utils.data import BatchSampler, SequentialSampler


class LengthBasedBatchSampler(BatchSampler):
    """
    Batch sampler that groups consecutive examples (sorted by length),
    then shuffles batches.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        base_sampler = SequentialSampler(range(dataset_size))
        super().__init__(base_sampler, batch_size, drop_last=False)
        self.shuffle = shuffle
        self.seed = seed
        self._batches = [
            list(batch)
            for batch in BatchSampler(base_sampler, batch_size, drop_last=False)
        ]

    def __iter__(self):
        batches = self._batches.copy()
        if self.shuffle:
            if self.seed is not None:
                np.random.seed(self.seed)
            np.random.shuffle(batches)
        return iter(batches)
