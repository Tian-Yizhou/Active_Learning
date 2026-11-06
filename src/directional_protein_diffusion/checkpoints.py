"""The inference checkpoint payload format, owned in one place.

Training writes this payload and generation reads it, so the format version and the
field names live here rather than being repeated at both ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from .config import ProjectConfig, config_from_dict

CHECKPOINT_FORMAT_VERSION = 1
"""Bump when the payload layout changes incompatibly; readers reject other values."""

INFERENCE_FILENAME = "inference.pt"
"""Name of the portable payload inside a checkpoint directory."""


@dataclass(frozen=True)
class InferenceCheckpoint:
    """A trained denoiser's weights, its configuration, and its provenance."""

    path: Path
    config: ProjectConfig
    state: dict[str, Tensor]
    epoch: int
    metrics: dict[str, float]


def save_inference_checkpoint(
    path: str | Path,
    config: ProjectConfig,
    state: Mapping[str, Tensor],
    epoch: int,
    metrics: Mapping[str, float],
) -> None:
    """Write configuration and denoiser weights on CPU; the frozen ESM is not included."""
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "epoch": epoch,
            "config": config.to_dict(),
            "diffusion": {key: value.cpu() for key, value in state.items()},
            "metrics": dict(metrics),
        },
        path,
    )


def load_inference_checkpoint(
    path: str | Path,
    *,
    sampling_overrides: Mapping[str, Any] | None = None,
) -> InferenceCheckpoint:
    """Load and validate a payload, applying non-None sampling overrides.

    Overrides are validated against the configuration schema here, so every caller
    gets the same rejection of impossible sampling requests.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported checkpoint format")
    config = config_from_dict(payload["config"])
    for name, value in (sampling_overrides or {}).items():
        if not hasattr(config.sampling, name):
            raise ValueError(f"Unknown sampling setting: {name}")
        if value is not None:
            setattr(config.sampling, name, value)
    config.validate()
    return InferenceCheckpoint(
        path=Path(path),
        config=config,
        state=payload["diffusion"],
        epoch=int(payload.get("epoch", 0)),
        metrics=dict(payload.get("metrics", {})),
    )
