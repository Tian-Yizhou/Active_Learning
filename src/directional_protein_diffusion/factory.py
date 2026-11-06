"""Shared model construction, so a checkpoint always rebuilds the same architecture."""

from __future__ import annotations

from .config import ProjectConfig
from .diffusion.gaussian import GaussianDiffusion
from .models.denoiser import ConditionalDenoiser
from .models.esm import FrozenESM2


def build_models(config: ProjectConfig) -> tuple[FrozenESM2, GaussianDiffusion]:
    """Load frozen ESM-2, copy its blocks, and construct the x0 diffusion module."""
    esm = FrozenESM2.from_pretrained(
        config.model.name,
        revision=config.model.revision,
        local_files_only=config.model.local_files_only,
    )
    denoiser = ConditionalDenoiser(esm, config.data.num_objectives, config.model.num_layers)
    diffusion = GaussianDiffusion(
        denoiser,
        config.diffusion.timesteps,
        config.diffusion.noise_schedule,
        config.diffusion.p_uncond,
    )
    return esm, diffusion
