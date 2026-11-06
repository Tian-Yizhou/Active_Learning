"""The directional label contract: measured differences and the CLI sign mapping.

Equality with a threshold belongs to ``<nc>`` because a directional claim must exceed
experimental noise. Reversing a pair swaps ``<dec>`` and ``<inc>`` and preserves ``<nc>``.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor

from ..constants import DEC, INC, NC, SIGNED_TO_ID


def directional_labels(
    source: Sequence[float],
    target: Sequence[float],
    tau: Sequence[float],
) -> Tensor:
    """Return int64 [j]: 0=decrease, 1=no change, 2=increase; equality is no change."""
    if not len(source) == len(target) == len(tau) or not tau:
        raise ValueError("source, target, and tau must have the same nonzero width")
    if any(not math.isfinite(x) for x in (*source, *target, *tau)) or any(x < 0 for x in tau):
        raise ValueError("Fitness and thresholds must be finite; thresholds nonnegative")
    return torch.tensor(
        [
            INC if b - a > threshold else DEC if b - a < -threshold else NC
            for a, b, threshold in zip(source, target, tau)
        ],
        dtype=torch.long,
    )


def signed_directions_to_ids(directions: Sequence[int], num_objectives: int) -> list[int]:
    """Map CLI signs [-1,0,+1] to token IDs [0,1,2] in configured objective order.

    The null token is deliberately unreachable here: it is a guidance mechanism, not a
    requestable experimental intent.
    """
    if len(directions) != num_objectives or any(d not in SIGNED_TO_ID for d in directions):
        raise ValueError("Provide one direction per objective, using only -1, 0, or +1")
    return [SIGNED_TO_ID[d] for d in directions]
