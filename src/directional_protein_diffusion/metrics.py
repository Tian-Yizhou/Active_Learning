"""Sequence-space diagnostics; Hamming distance assumes aligned equal-length proteins."""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

from .constants import AMINO_ACIDS


def hamming_distance(seed: str, candidate: str) -> int:
    """Count substitutions; reject unequal lengths rather than silently ignoring indels."""
    if len(seed) != len(candidate) or not seed:
        raise ValueError("Hamming distance requires nonempty equal-length sequences")
    return sum(a != b for a, b in zip(seed, candidate))


def amino_acid_recovery(reference: str, candidate: str) -> float:
    """Return matching-residue fraction in [0,1] against an explicit reference sequence."""
    return 1 - hamming_distance(reference, candidate) / len(reference)


def shannon_entropy(sequences: Sequence[str]) -> float:
    """Mean per-position Shannon entropy (bits) across aligned generated sequences.

    This measures ensemble diversity, not within-sequence amino-acid composition.
    One sequence or a fully identical ensemble has entropy zero.
    """
    if not sequences or not sequences[0]:
        raise ValueError("Supply at least one nonempty sequence")
    if any(len(s) != len(sequences[0]) or set(s) - set(AMINO_ACIDS) for s in sequences):
        raise ValueError("Entropy requires equal-length canonical amino-acid sequences")
    n = len(sequences)
    return sum(
        -sum((count / n) * math.log2(count / n) for count in Counter(column).values())
        for column in zip(*sequences)
    ) / len(sequences[0])
