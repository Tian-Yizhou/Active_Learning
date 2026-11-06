"""Scientific data invariants: disjoint variants, threshold equality, and reversal."""

from dataclasses import replace
from pathlib import Path

import pytest

from directional_protein_diffusion.data.labels import (
    directional_labels,
    signed_directions_to_ids,
)
from directional_protein_diffusion.data.pairs import PairedVariantDataset
from directional_protein_diffusion.data.synthetic import (
    SYNTHETIC_OBJECTIVE_NAMES,
    generate_synthetic_variants,
)
from directional_protein_diffusion.data.variants import (
    read_variants,
    split_variants,
    write_variants,
)


def test_split_precedes_pairing_and_rejects_replicate_leakage() -> None:
    variants = generate_synthetic_variants(10, 8)
    training, validation = split_variants(variants)
    assert {v.sequence for v in training}.isdisjoint(v.sequence for v in validation)
    assert split_variants(variants) == (training, validation)
    train_pairs = PairedVariantDataset(training, [0.05, 0.05])
    val_pairs = PairedVariantDataset(validation, [0.05, 0.05])
    assert len(train_pairs) == len(training) * (len(training) - 1)
    assert len(val_pairs) == len(validation) * (len(validation) - 1)
    with pytest.raises(ValueError, match="Duplicate sequences"):
        split_variants([*variants, replace(variants[0], identifier="replicate")])


def test_threshold_boundaries_and_reversal() -> None:
    source = [0.0] * 5
    target = [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert directional_labels(source, target, [1.0] * 5).tolist() == [0, 1, 1, 1, 2]
    assert directional_labels(target, source, [1.0] * 5).tolist() == [2, 1, 1, 1, 0]
    assert signed_directions_to_ids([1, -1], 2) == [2, 0]
    with pytest.raises(ValueError):
        directional_labels([0.0], [float("nan")], [0.1])
    with pytest.raises(ValueError, match="-1, 0, or \\+1"):
        signed_directions_to_ids([3, 0], 2)


def test_csv_round_trip(tmp_path: Path) -> None:
    variants = generate_synthetic_variants(6, 8)
    path = tmp_path / "variants.csv"
    write_variants(path, variants, SYNTHETIC_OBJECTIVE_NAMES)
    restored = read_variants(path, SYNTHETIC_OBJECTIVE_NAMES)
    assert [v.identifier for v in restored] == [v.identifier for v in variants]
    assert [v.sequence for v in restored] == [v.sequence for v in variants]
    for before, after in zip(variants, restored):
        assert after.fitness == pytest.approx(before.fitness)
    with pytest.raises(ValueError, match="objective_names"):
        write_variants(path, variants, ["only_one"])
