"""Domain vocabulary and directional label IDs.

This module must stay free of internal dependencies so that every layer can import
it without depending on data loading or on model code.
"""

from __future__ import annotations

from typing import Final

AMINO_ACIDS: Final = "ACDEFGHIKLMNPQRSTVWY"
"""The 20 canonical amino acids, in the fixed order used for decoding."""

DEC: Final = 0
"""Measured value decreased beyond the objective's noise threshold."""

NC: Final = 1
"""No change distinguishable from experimental noise."""

INC: Final = 2
"""Measured value increased beyond the objective's noise threshold."""

NULL: Final = 3
"""Dropped directional condition for classifier-free guidance; never a measurement."""

NUM_DIRECTION_IDS: Final = 4
"""Size of the per-objective direction vocabulary, including the null token."""

DIRECTION_TOKENS: Final = {DEC: "<dec>", NC: "<nc>", INC: "<inc>", NULL: "<null>"}
"""Readable token for each direction ID, for logging and documentation."""

SIGNED_TO_ID: Final = {-1: DEC, 0: NC, 1: INC}
"""CLI-facing signed directions mapped to internal IDs; NULL is not requestable."""
