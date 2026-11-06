"""Measured variant records, CSV round-tripping, and leakage-resistant splitting."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..constants import AMINO_ACIDS


@dataclass(frozen=True)
class Variant:
    """One unique measured sequence with ordered objective values of length j."""

    identifier: str
    sequence: str
    fitness: tuple[float, ...]


def validate_variants(variants: Sequence[Variant]) -> None:
    """Require unique IDs/sequences, finite fitness, and a common aligned length.

    Aggregate experimental replicates before calling this function. Rejecting duplicate
    sequences prevents identical proteins with different IDs leaking across splits.
    """
    if not variants:
        raise ValueError("No variants supplied")
    if len({v.identifier for v in variants}) != len(variants):
        raise ValueError("Variant IDs must be unique")
    if len({v.sequence for v in variants}) != len(variants):
        raise ValueError("Duplicate sequences: aggregate replicates before splitting")
    length, objectives = len(variants[0].sequence), len(variants[0].fitness)
    if length == 0 or objectives == 0:
        raise ValueError("Sequences and fitness vectors must be nonempty")
    for variant in variants:
        if len(variant.sequence) != length or set(variant.sequence) - set(AMINO_ACIDS):
            raise ValueError("Use aligned equal-length sequences of canonical amino acids")
        if len(variant.fitness) != objectives or not all(map(math.isfinite, variant.fitness)):
            raise ValueError("Fitness vectors must have equal width and finite values")


def read_variants(path: str | Path, objective_names: Sequence[str]) -> list[Variant]:
    """Read CSV columns ``id,sequence,<objective_names in configured order>``."""
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "sequence", *objective_names}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain columns {sorted(required)}")
        variants = [
            Variant(
                row["id"],
                row["sequence"].strip().upper(),
                tuple(float(row[name]) for name in objective_names),
            )
            for row in reader
        ]
    validate_variants(variants)
    return variants


def write_variants(
    path: str | Path,
    variants: Sequence[Variant],
    objective_names: Sequence[str],
) -> None:
    """Write ``id,sequence,<objective_names>``, the exact schema ``read_variants`` expects."""
    validate_variants(variants)
    if any(len(v.fitness) != len(objective_names) for v in variants):
        raise ValueError("objective_names must match the width of every fitness vector")
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "sequence", *objective_names])
        writer.writerows((v.identifier, v.sequence, *v.fitness) for v in variants)


def split_variants(
    variants: Sequence[Variant],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[Variant], list[Variant]]:
    """Split M unique sequences BEFORE pairing; each partition has at least two.

    This is sequence-disjoint, not a homology/lineage-disjoint split. For stronger
    extrapolation claims, supply an independently designed cluster/round split.
    """
    validate_variants(variants)
    if len(variants) < 4 or not 0 < val_fraction < 1:
        raise ValueError("Need at least four variants and 0 < val_fraction < 1")
    shuffled = list(variants)
    random.Random(seed).shuffle(shuffled)
    n_val = min(len(shuffled) - 2, max(2, round(len(shuffled) * val_fraction)))
    return shuffled[n_val:], shuffled[:n_val]
