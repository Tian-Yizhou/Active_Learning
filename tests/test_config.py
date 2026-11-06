"""Configuration validation, the gate protecting the scientific invariants."""

from pathlib import Path
from typing import Callable

import pytest
import yaml

from directional_protein_diffusion.config import ProjectConfig, config_from_dict, load_config

INVALID: dict[str, Callable[[ProjectConfig], None]] = {
    "threshold width": lambda c: setattr(c.data, "tau", [0.05]),
    "objective name count": lambda c: setattr(c.data, "objective_names", ["only_one"]),
    "duplicate objective names": lambda c: setattr(c.data, "objective_names", ["same", "same"]),
    "negative threshold": lambda c: setattr(c.data, "tau", [0.05, -0.1]),
    "validation fraction": lambda c: setattr(c.data, "val_fraction", 1.0),
    "timestep count": lambda c: setattr(c.diffusion, "timesteps", 1),
    "schedule name": lambda c: setattr(c.diffusion, "noise_schedule", "quadratic"),
    "dropout probability": lambda c: setattr(c.diffusion, "p_uncond", 1.5),
    "batch size": lambda c: setattr(c.training, "batch_size", 0),
    "learning rate": lambda c: setattr(c.training, "learning_rate", 0.0),
    "mixed precision": lambda c: setattr(c.training, "mixed_precision", "fp8"),
    "denoiser depth": lambda c: setattr(c.model, "num_layers", 0),
    "sdedit start": lambda c: setattr(c.sampling, "t_start", 0),
    "steps exceed start": lambda c: setattr(c.sampling, "t_start", 20),
    "guidance scale": lambda c: setattr(c.sampling, "guidance_scale", -1.0),
    "eta range": lambda c: setattr(c.sampling, "eta", 1.5),
}


@pytest.mark.parametrize("case", sorted(INVALID))
def test_validate_rejects_inconsistent_settings(case: str) -> None:
    config = ProjectConfig()
    INVALID[case](config)
    with pytest.raises(ValueError):
        config.validate()


def test_defaults_validate_and_round_trip(tmp_path: Path) -> None:
    config = ProjectConfig()
    config.validate()
    assert config_from_dict(config.to_dict()).to_dict() == config.to_dict()
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config.to_dict()))
    assert load_config(path).to_dict() == config.to_dict()


def test_misspelled_and_malformed_configurations_are_rejected(tmp_path: Path) -> None:
    raw = ProjectConfig().to_dict()
    raw["data"]["objektive_names"] = ["a", "b"]
    with pytest.raises(TypeError):
        config_from_dict(raw)
    path = tmp_path / "bad.yaml"
    path.write_text("- not\n- a mapping\n")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(path)
