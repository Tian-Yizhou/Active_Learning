"""Time, ordered directional tokens, and adaptive normalization; B=batch, D=width."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..constants import NULL, NUM_DIRECTION_IDS


class SinusoidalTimeEmbedding(nn.Module):
    """Map timesteps [B] to sinusoidal features and an MLP output [B, D]."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 4 or dim % 2:
            raise ValueError("Time embedding width must be even and at least 4")
        frequencies = torch.exp(-math.log(10000) * torch.arange(dim // 2) / (dim // 2 - 1))
        self.register_buffer("frequencies", frequencies)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim))

    def forward(self, timesteps: Tensor) -> Tensor:
        """Embed [B] integer timesteps into floating features [B, D]."""
        angles = timesteps.float()[:, None] * self.frequencies[None, :]
        return self.mlp(torch.cat((angles.sin(), angles.cos()), dim=-1))


class DirectionEmbedding(nn.Module):
    """Separate four-token vocabularies per objective, summed then mapped by an MLP."""

    def __init__(self, num_objectives: int, dim: int) -> None:
        super().__init__()
        if num_objectives < 1:
            raise ValueError("num_objectives must be positive")
        self.num_objectives = num_objectives
        self.embedding = nn.Embedding(NUM_DIRECTION_IDS * num_objectives, dim)
        self.register_buffer("offsets", NUM_DIRECTION_IDS * torch.arange(num_objectives))
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, directions: Tensor) -> Tensor:
        """Embed int64 [B, j] into [B, D], preserving which objective each ID belongs to."""
        if directions.ndim != 2 or directions.shape[1] != self.num_objectives:
            raise ValueError("directions must have shape [B, num_objectives]")
        if directions.dtype != torch.long or ((directions < 0) | (directions > NULL)).any():
            raise ValueError("Direction IDs must be int64 in [0, 3]")
        return self.mlp(self.embedding(directions + self.offsets).sum(dim=1))


def drop_directions(directions: Tensor, probability: float) -> Tensor:
    """Replace all j tokens in selected rows of [B,j] by NULL; never mutate the input."""
    if not 0 <= probability <= 1:
        raise ValueError("Drop probability must lie in [0, 1]")
    dropped = torch.rand(directions.shape[0], 1, device=directions.device) < probability
    return torch.where(dropped, NULL, directions)


class AdaptiveLayerNorm(nn.Module):
    """Apply gamma(c)*LN(h)+beta(c) to h [B,L,D] with c [B,D].

    Initialize gamma=1 and beta=0 to preserve a supplied pretrained normalization.
    The explicit gamma includes the identity offset used in residual formulations.
    """

    def __init__(self, dim: int, norm: nn.Module | None = None) -> None:
        super().__init__()
        self.norm = norm if norm is not None else nn.LayerNorm(dim)
        self.modulation = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
        with torch.no_grad():
            self.modulation.bias[:dim].fill_(1)

    def forward(self, hidden: Tensor, condition: Tensor) -> Tensor:
        """Return conditioned activations [B,L,D], broadcasting over residue positions."""
        gamma, beta = self.modulation(condition).chunk(2, dim=-1)
        return gamma[:, None, :] * self.norm(hidden) + beta[:, None, :]
