"""Train the conditional latent denoiser from a YAML experiment configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

HELP = "Train the conditional latent diffusion denoiser"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register training arguments on the given parser."""
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Experiment configuration YAML (default: configs/default.yaml)",
    )


def run(args: argparse.Namespace) -> None:
    """Load the configuration and run training to completion."""
    from ..config import load_config
    from ..training import run_training

    checkpoint = run_training(load_config(args.config))
    print(f"Last checkpoint: {checkpoint}")
