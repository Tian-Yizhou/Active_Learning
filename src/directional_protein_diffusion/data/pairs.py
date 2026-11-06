"""Paired supervision over within-partition variants. Strings stay on the CPU."""

from __future__ import annotations

from itertools import combinations
from typing import Sequence, TypedDict

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .labels import directional_labels
from .variants import Variant, validate_variants


class PairExample(TypedDict):
    """Single source/target example; directions is an int64 tensor [j]."""

    source: str
    target: str
    directions: Tensor


class PairBatch(TypedDict):
    """Collated strings of length B and int64 directions [B, j]."""

    source: list[str]
    target: list[str]
    directions: Tensor


class PairedVariantDataset(Dataset[PairExample]):
    """All within-partition pairs, optionally reversed; retains every <nc> pair.

    Stores O(M²) index pairs for clarity in this low-N scaffold. Latents are encoded
    on demand by the training loop, never in DataLoader worker processes.
    """

    def __init__(
        self,
        variants: Sequence[Variant],
        tau: Sequence[float],
        bidirectional: bool = True,
    ) -> None:
        validate_variants(variants)
        if len(variants) < 2:
            raise ValueError("A paired partition needs at least two variants")
        directional_labels(variants[0].fitness, variants[0].fitness, tau)
        self.variants = list(variants)
        self.tau = tuple(tau)
        self.pairs = list(combinations(range(len(variants)), 2))
        if bidirectional:
            self.pairs += [(b, a) for a, b in self.pairs]

    def __len__(self) -> int:
        """Number of ordered examples in this partition."""
        return len(self.pairs)

    def __getitem__(self, index: int) -> PairExample:
        """Fetch source/target strings and directional token IDs [j]."""
        a, b = (self.variants[i] for i in self.pairs[index])
        return {
            "source": a.sequence,
            "target": b.sequence,
            "directions": directional_labels(a.fitness, b.fitness, self.tau),
        }


def collate_pairs(examples: list[PairExample]) -> PairBatch:
    """Stack directions to [B, j], retaining the B source/target strings."""
    return {
        "source": [e["source"] for e in examples],
        "target": [e["target"] for e in examples],
        "directions": torch.stack([e["directions"] for e in examples]),
    }


def make_dataloader(
    dataset: PairedVariantDataset,
    batch_size: int,
    *,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader[PairBatch]:
    """Build a seeded loader; tensors are moved to the accelerator by the caller."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pairs,
        generator=torch.Generator().manual_seed(seed),
    )
