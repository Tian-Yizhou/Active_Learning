"""Sequence-space diagnostics used to report candidates."""

import pytest

from directional_protein_diffusion.metrics import (
    amino_acid_recovery,
    hamming_distance,
    shannon_entropy,
)


def test_sequence_metrics() -> None:
    assert hamming_distance("ACDE", "ACDF") == 1
    assert amino_acid_recovery("ACDE", "ACDF") == 0.75
    assert shannon_entropy(["AA", "CC"]) == 1.0
    assert shannon_entropy(["AC", "AC"]) == 0.0
    with pytest.raises(ValueError):
        hamming_distance("AC", "ACD")
