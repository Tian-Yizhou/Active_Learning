"""Candidate generation from a trained checkpoint, callable without the CLI.

Records are plain dictionaries carrying their own provenance, so a caller can write
them as JSONL, collect them in a DataFrame, or feed them to candidate selection.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from accelerate.utils import set_seed

from .checkpoints import InferenceCheckpoint
from .factory import build_models
from .metrics import amino_acid_recovery, hamming_distance


def generate(
    checkpoint: InferenceCheckpoint,
    sequences: Sequence[str],
    direction_ids: Sequence[int],
    *,
    num_samples: int = 4,
    device: str | torch.device = "cpu",
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate ``num_samples`` candidates per seed sequence, one record each.

    ``direction_ids`` are internal IDs in [0,2] in the checkpoint's objective order;
    signed CLI values are converted by ``data.labels.signed_directions_to_ids``.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if not sequences:
        raise ValueError("Supply at least one seed sequence")
    config = checkpoint.config
    if len(direction_ids) != config.data.num_objectives:
        raise ValueError("Provide one direction per configured objective")
    set_seed(seed)
    esm, diffusion = build_models(config)
    diffusion.load_state_dict(checkpoint.state, strict=True)
    device = torch.device(device)
    esm.to(device)
    diffusion.to(device).eval()
    seeds = [sequence.upper() for sequence in sequences for _ in range(num_samples)]
    encoded = esm.encode(seeds)
    directions = torch.tensor(list(direction_ids), device=device).expand(len(seeds), -1)
    settings = config.to_dict()["sampling"]
    latents = diffusion.sample(
        encoded.latents,
        directions,
        encoded.mask,
        **settings,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    candidates = esm.decode(latents, encoded.mask)
    return [
        {
            "seed_sequence": source,
            "sequence": candidate,
            "objective_names": config.data.objective_names,
            "direction_ids": list(direction_ids),
            "hamming_distance": hamming_distance(source, candidate),
            "seed_recovery": amino_acid_recovery(source, candidate),
            "sampling": settings,
            "random_seed": seed,
            "checkpoint": str(checkpoint.path),
        }
        for source, candidate in zip(seeds, candidates)
    ]
