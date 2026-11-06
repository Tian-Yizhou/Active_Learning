"""Reproducible two-objective mock landscapes for plumbing tests, not biology."""

from __future__ import annotations

import random
from typing import Final

from ..constants import AMINO_ACIDS
from .variants import Variant

SYNTHETIC_OBJECTIVE_NAMES: Final = ("fitness_1", "fitness_2")
"""Column names written by the synthetic CSV writer; match ``DataConfig`` defaults."""


def generate_synthetic_variants(
    num_variants: int = 40,
    length: int = 32,
    seed: int = 42,
) -> list[Variant]:
    """Create unique mutants of one seed with two competing noisy mock scores.

    Each variant has 1..min(6,L) substitutions. Objective 1 favors hydrophobic
    residues; objective 2 favors charged residues. Neither is a real fitness assay.
    """
    if length < 1 or not 4 <= num_variants <= 1 + 19 * length:
        raise ValueError("Require length >= 1 and 4 <= num_variants <= 1 + 19*length")
    rng = random.Random(seed)
    parent = "".join(rng.choices(AMINO_ACIDS, k=length))
    sequences = [parent]
    seen = {parent}
    for _ in range(100 * num_variants):
        if len(sequences) == num_variants:
            break
        residues = list(parent)
        for position in rng.sample(range(length), rng.randint(1, min(6, length))):
            residues[position] = rng.choice(AMINO_ACIDS.replace(residues[position], ""))
        sequence = "".join(residues)
        if sequence not in seen:
            sequences.append(sequence)
            seen.add(sequence)
    if len(sequences) != num_variants:
        raise RuntimeError("Could not create enough unique variants; reduce num_variants")
    return [
        Variant(
            f"synthetic_{i:04d}",
            seq,
            (
                sum(aa in "AILMVFWY" for aa in seq) / length + rng.gauss(0, 0.01),
                sum(aa in "DEKR" for aa in seq) / length + rng.gauss(0, 0.01),
            ),
        )
        for i, seq in enumerate(sequences)
    ]
