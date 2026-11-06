"""Typed configuration; defaults mirror ``configs/default.yaml``."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Frozen Hugging Face ESM-2 checkpoint and copied denoiser depth."""

    name: str = "facebook/esm2_t6_8M_UR50D"
    revision: str = "main"
    local_files_only: bool = False
    num_layers: int | None = None


@dataclass
class DataConfig:
    """Ordered objectives and thresholds; split unique sequences before pairing."""

    csv_path: str | None = None
    objective_names: list[str] = field(default_factory=lambda: ["fitness_1", "fitness_2"])
    num_objectives: int = 2
    tau: list[float] = field(default_factory=lambda: [0.05, 0.05])
    val_fraction: float = 0.2
    bidirectional: bool = True
    synthetic_size: int = 40
    sequence_length: int = 32


@dataclass
class DiffusionConfig:
    """Gaussian schedule and whole-vector directional dropout."""

    timesteps: int = 1000
    noise_schedule: str = "cosine"
    p_uncond: float = 0.1


@dataclass
class TrainingConfig:
    """AdamW, per-epoch cosine decay, and Accelerate runtime settings."""

    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    num_workers: int = 0
    mixed_precision: str = "no"
    output_dir: str = "runs/default"
    checkpoint_every: int = 1


@dataclass
class SamplingConfig:
    """DDIM settings; ``t_start=None`` means full-noise initialization."""

    steps: int = 50
    guidance_scale: float = 2.0
    eta: float = 0.0
    t_start: int | None = None


@dataclass
class ProjectConfig:
    """Serializable experiment configuration, including fixed objective ordering."""

    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    def validate(self) -> None:
        """Reject inconsistent dimensions, invalid thresholds, and unsafe schedules."""
        d, t, s = self.data, self.training, self.sampling
        if d.num_objectives < 1 or len(d.tau) != d.num_objectives:
            raise ValueError("tau must have num_objectives entries")
        if len(d.objective_names) != d.num_objectives:
            raise ValueError("objective_names must have num_objectives entries")
        if len(set(d.objective_names)) != d.num_objectives:
            raise ValueError("objective_names must be unique")
        if any(not math.isfinite(x) or x < 0 for x in d.tau):
            raise ValueError("tau must be finite and nonnegative")
        if not 0 < d.val_fraction < 1:
            raise ValueError("val_fraction must lie in (0, 1)")
        if self.diffusion.timesteps < 2 or self.diffusion.noise_schedule not in {
            "cosine",
            "linear",
        }:
            raise ValueError("Use at least two timesteps and a cosine or linear schedule")
        if not 0 <= self.diffusion.p_uncond <= 1:
            raise ValueError("p_uncond must lie in [0, 1]")
        if min(t.batch_size, t.epochs, t.checkpoint_every) < 1 or t.num_workers < 0:
            raise ValueError("Invalid training counts")
        if t.learning_rate <= 0 or t.weight_decay < 0 or t.gradient_clip <= 0:
            raise ValueError("Invalid optimizer settings")
        if t.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be no, fp16, or bf16")
        if self.model.num_layers is not None and self.model.num_layers < 1:
            raise ValueError("num_layers must be positive or null")
        start = self.diffusion.timesteps if s.t_start is None else s.t_start
        if s.t_start is not None and not 0 < s.t_start < self.diffusion.timesteps:
            raise ValueError("SDEdit requires 0 < t_start < T")
        if not 1 <= s.steps <= start or not 0 <= s.eta <= 1:
            raise ValueError("Sampling requires 1 <= steps <= start and 0 <= eta <= 1")
        if not math.isfinite(s.guidance_scale) or s.guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and nonnegative")

    def to_dict(self) -> dict[str, Any]:
        """Return plain values suitable for YAML and checkpoint serialization."""
        return asdict(self)


def config_from_dict(raw: dict[str, Any]) -> ProjectConfig:
    """Construct nested dataclasses, rejecting misspelled configuration keys."""
    values = dict(raw)
    for key, cls in (
        ("model", ModelConfig),
        ("data", DataConfig),
        ("diffusion", DiffusionConfig),
        ("training", TrainingConfig),
        ("sampling", SamplingConfig),
    ):
        values[key] = cls(**values.get(key, {}))
    result = ProjectConfig(**values)
    result.validate()
    return result


def load_config(path: str | Path) -> ProjectConfig:
    """Read a YAML mapping and validate the resulting experiment configuration."""
    with Path(path).open() as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return config_from_dict(raw)
