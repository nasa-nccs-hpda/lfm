"""Epoch-addressable shuffling for the paired, single-device experiment."""

import torch
from torch.utils.data import Sampler


class EpochShuffleSampler(Sampler[int]):
    """Derive membership order from seed and epoch, including after resume."""

    def __init__(self, dataset, seed: int, epoch: int = 0):
        self.dataset = dataset
        self.seed = seed
        self.epoch = epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def __len__(self):
        return len(self.dataset)
