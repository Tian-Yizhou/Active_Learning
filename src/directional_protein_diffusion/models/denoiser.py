"""Seed-conditioned x0 prediction with copied pretrained ESM-2 attention and FFNs."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import Tensor, nn
from transformers.models.esm.modeling_esm import EsmLayer

from .conditioning import AdaptiveLayerNorm, DirectionEmbedding, SinusoidalTimeEmbedding
from .esm import FrozenESM2


class ConditionalESMBlock(nn.Module):
    """Copy an ESM-2 block and modulate both pretrained pre-norms with condition c.

    Attention, rotary positions, residual paths, and feed-forward weights retain the
    Hugging Face ESM-2 architecture. No parameter aliases the frozen encoder.
    """

    def __init__(self, pretrained: EsmLayer, dim: int) -> None:
        super().__init__()
        layer = deepcopy(pretrained)
        self.attention = layer.attention.self
        self.attention_output = layer.attention.output
        self.intermediate = layer.intermediate
        self.output = layer.output
        self.attention_norm = AdaptiveLayerNorm(dim, layer.attention.LayerNorm)
        self.ffn_norm = AdaptiveLayerNorm(dim, layer.LayerNorm)
        self.requires_grad_(True)

    def forward(self, hidden: Tensor, condition: Tensor, attention_mask: Tensor) -> Tensor:
        """Transform [B,L,D] using [B,D] conditioning and additive mask [B,1,1,L]."""
        attended = self.attention(
            self.attention_norm(hidden, condition),
            attention_mask=attention_mask,
        )[0]
        hidden = self.attention_output(attended, hidden)
        return self.output(self.intermediate(self.ffn_norm(hidden, condition)), hidden)


class ConditionalDenoiser(nn.Module):
    """Concatenate target/source features and predict a clean target latent [B,L,D]."""

    def __init__(
        self,
        esm: FrozenESM2,
        num_objectives: int,
        num_layers: int | None = None,
    ) -> None:
        super().__init__()
        dim = esm.hidden_size
        pretrained = esm.model.esm.encoder.layer
        depth = len(pretrained) if num_layers is None else num_layers
        if not 1 <= depth <= len(pretrained):
            raise ValueError("Denoiser depth must fit the selected pretrained ESM-2 model")
        self.hidden_size = dim
        self.num_objectives = num_objectives
        self.input_projection = nn.Linear(2 * dim, dim)
        self.time_embedding = SinusoidalTimeEmbedding(dim)
        self.direction_embedding = DirectionEmbedding(num_objectives, dim)
        self.blocks = nn.ModuleList(
            [ConditionalESMBlock(layer, dim) for layer in pretrained[:depth]]
        )
        self.final_norm = deepcopy(esm.model.esm.encoder.emb_layer_norm_after)
        self.final_norm.requires_grad_(True)
        self.output_projection = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(
        self,
        noisy_target: Tensor,
        timesteps: Tensor,
        source: Tensor,
        directions: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Predict x0 [B,L,D] from target/source [B,L,D], t [B], d [B,j], mask [B,L].

        ``mask=True`` means a valid residue. Direction dropout belongs to the training
        diffusion module so CFG inference can explicitly evaluate either branch.
        """
        if noisy_target.shape != source.shape or source.ndim != 3:
            raise ValueError("Target and source must share shape [B,L,D]")
        if source.shape[-1] != self.hidden_size or timesteps.shape != (source.shape[0],):
            raise ValueError("Incorrect latent width or timestep shape")
        if mask.shape != source.shape[:2] or mask.dtype != torch.bool or not mask.any(dim=1).all():
            raise ValueError("mask must be bool [B,L] with at least one valid residue per row")
        if directions.shape[0] != source.shape[0]:
            raise ValueError("Directions and source must share batch size")
        hidden = self.input_projection(torch.cat((noisy_target, source), dim=-1))
        hidden = hidden.masked_fill(~mask[:, :, None], 0)
        condition = self.time_embedding(timesteps) + self.direction_embedding(directions)
        attention_mask = (~mask[:, None, None, :]).to(hidden.dtype) * torch.finfo(hidden.dtype).min
        for block in self.blocks:
            hidden = block(hidden, condition, attention_mask)
        return self.output_projection(self.final_norm(hidden)).masked_fill(~mask[:, :, None], 0)
