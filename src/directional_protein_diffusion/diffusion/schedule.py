"""Cumulative noise schedules indexed so that state zero is exactly clean."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def make_alpha_bar(timesteps: int, schedule: str = "cosine") -> Tensor:
    """Return float32 cumulative alphas [T+1]; index zero is exactly the clean state."""
    if timesteps < 2:
        raise ValueError("timesteps must be at least two")
    if schedule == "cosine":
        grid = torch.linspace(0, 1, timesteps + 1, dtype=torch.float64)
        cumulative = torch.cos((grid + 0.008) / 1.008 * math.pi / 2).square()
        cumulative /= cumulative[0].clone()
        betas = (1 - cumulative[1:] / cumulative[:-1]).clamp(1e-8, 0.999)
    elif schedule == "linear":
        # Fixed endpoints; unlike image-code rescaling, valid even for tiny smoke tests.
        betas = torch.linspace(1e-4, 0.02, timesteps, dtype=torch.float64)
    else:
        raise ValueError("schedule must be cosine or linear")
    return torch.cat((torch.ones(1, dtype=torch.float64), (1 - betas).cumprod(0))).float()
