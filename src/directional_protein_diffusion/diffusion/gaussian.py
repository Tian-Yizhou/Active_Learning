"""Target-only Gaussian diffusion with masked x0 loss and DDIM/SDEdit sampling."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn

from ..constants import NULL
from ..models.conditioning import drop_directions
from ..models.denoiser import ConditionalDenoiser
from .schedule import make_alpha_bar


class GaussianDiffusion(nn.Module):
    """Own the trainable denoiser and schedule; never own or update the frozen encoder."""

    def __init__(
        self,
        denoiser: ConditionalDenoiser,
        timesteps: int = 1000,
        noise_schedule: str = "cosine",
        p_uncond: float = 0.1,
    ) -> None:
        super().__init__()
        if not 0 <= p_uncond <= 1:
            raise ValueError("p_uncond must lie in [0, 1]")
        self.denoiser = denoiser
        self.timesteps = timesteps
        self.p_uncond = p_uncond
        self.register_buffer("alpha_bar", make_alpha_bar(timesteps, noise_schedule))

    def q_sample(self, clean: Tensor, timesteps: Tensor, noise: Tensor | None = None) -> Tensor:
        """Noise clean [B,L,D] at t [B] in [0,T]; t=0 returns the clean latent exactly."""
        if clean.ndim != 3 or timesteps.shape != (clean.shape[0],):
            raise ValueError("Expected clean [B,L,D] and timesteps [B]")
        if timesteps.dtype != torch.long or ((timesteps < 0) | (timesteps > self.timesteps)).any():
            raise ValueError("Timesteps must be int64 in [0,T]")
        noise = torch.randn_like(clean) if noise is None else noise
        if noise.shape != clean.shape:
            raise ValueError("Noise must have the same shape as clean latents")
        alpha = self.alpha_bar[timesteps][:, None, None]
        return alpha.sqrt() * clean + (1 - alpha).sqrt() * noise

    def forward(
        self,
        clean_target: Tensor,
        source: Tensor,
        directions: Tensor,
        mask: Tensor,
        *,
        reduction: Literal["mean", "none"] = "mean",
    ) -> Tensor:
        """Masked x0 MSE: inputs [B,L,D], [B,L,D], [B,j], [B,L]; scalar or [B] loss.

        Training drops directions with p_uncond. Evaluation retains directions but
        still samples t and noise, so validation is a stochastic reconstruction metric.
        Each sample is averaged over its valid residues and hidden features.
        """
        if ((directions < 0) | (directions > 2)).any():
            raise ValueError("Measured training labels must be in [0,2]")
        t = torch.randint(1, self.timesteps + 1, (source.shape[0],), device=source.device)
        noisy = self.q_sample(clean_target, t).masked_fill(~mask[:, :, None], 0)
        d = drop_directions(directions, self.p_uncond) if self.training else directions
        predicted = self.denoiser(noisy, t, source, d, mask)
        squared = (predicted.float() - clean_target.float()).square()
        per_sample = squared.masked_fill(~mask[:, :, None], 0).sum(dim=(1, 2))
        per_sample = per_sample / (mask.sum(dim=1) * source.shape[-1])
        if reduction == "none":
            return per_sample
        if reduction != "mean":
            raise ValueError("reduction must be mean or none")
        return per_sample.mean()

    def guided_prediction(
        self,
        state: Tensor,
        t: Tensor,
        source: Tensor,
        directions: Tensor,
        mask: Tensor,
        guidance_scale: float,
    ) -> Tensor:
        """Compute x0_null+w*(x0_cond-x0_null), keeping the same clean seed in both."""
        if guidance_scale == 1:
            return self.denoiser(state, t, source, directions, mask)
        null = torch.full_like(directions, NULL)
        unconditional = self.denoiser(state, t, source, null, mask)
        if guidance_scale == 0:
            return unconditional
        conditional = self.denoiser(state, t, source, directions, mask)
        return unconditional + guidance_scale * (conditional - unconditional)

    @torch.no_grad()
    def sample(
        self,
        source: Tensor,
        directions: Tensor,
        mask: Tensor,
        *,
        steps: int = 50,
        guidance_scale: float = 2.0,
        eta: float = 0.0,
        t_start: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Return predicted clean targets [B,L,D] via DDIM or seed-initialized SDEdit.

        ``t_start=None`` draws pure noise at T; otherwise 0<t_start<T noises a COPY
        of the seed. The clean source condition is constant throughout the trajectory.
        eta=0 is deterministic given initial noise. No image-style latent clipping is
        applied. This is soft edit control, not a guaranteed mutation budget.
        """
        start = self.timesteps if t_start is None else t_start
        if t_start is not None and not 0 < t_start < self.timesteps:
            raise ValueError("SDEdit requires 0 < t_start < T")
        if not 1 <= steps <= start or not 0 <= eta <= 1:
            raise ValueError("Require 1 <= steps <= start and 0 <= eta <= 1")
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and nonnegative")
        if ((directions < 0) | (directions > 2)).any():
            raise ValueError("Requested directions must be in [0,2]")
        noise = torch.randn(
            source.shape, device=source.device, dtype=source.dtype, generator=generator
        )
        t = torch.full((source.shape[0],), start, device=source.device, dtype=torch.long)
        state = noise if t_start is None else self.q_sample(source, t, noise)
        state = state.masked_fill(~mask[:, :, None], 0)
        trajectory = torch.linspace(start, 0, steps + 1).round().long().tolist()
        was_training = self.training
        self.eval()
        try:
            for current, following in zip(trajectory[:-1], trajectory[1:]):
                t.fill_(current)
                predicted = self.guided_prediction(
                    state,
                    t,
                    source,
                    directions,
                    mask,
                    guidance_scale,
                )
                if following == 0:
                    state = predicted
                    break
                a, a_next = self.alpha_bar[current], self.alpha_bar[following]
                epsilon = (state - a.sqrt() * predicted) / (1 - a).sqrt()
                sigma = eta * ((1 - a_next) / (1 - a) * (1 - a / a_next)).clamp_min(0).sqrt()
                state = (
                    a_next.sqrt() * predicted
                    + (1 - a_next - sigma.square()).clamp_min(0).sqrt() * epsilon
                )
                if eta > 0:
                    state = state + sigma * torch.randn(
                        source.shape,
                        device=source.device,
                        dtype=source.dtype,
                        generator=generator,
                    )
                state = state.masked_fill(~mask[:, :, None], 0)
        finally:
            self.train(was_training)
        return state.masked_fill(~mask[:, :, None], 0)
