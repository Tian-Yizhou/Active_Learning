"""Generate candidate mutants from a checkpoint using signed objective directions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HELP = "Generate candidate mutants from a trained checkpoint"


def _positive_int(value: str) -> int:
    """Parse a positive integer, so argparse reports the error rather than the model."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register generation arguments on the given parser."""
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to inference.pt")
    parser.add_argument("--sequences", nargs="+", required=True, help="Canonical seed sequences")
    parser.add_argument(
        "--directions",
        nargs="+",
        type=int,
        required=True,
        help="Signed directions, e.g. --directions 1 -1 (not token IDs)",
    )
    parser.add_argument("--num-samples", type=_positive_int, default=4, help="Candidates per seed")
    parser.add_argument("--steps", type=int, help="Override the checkpoint's DDIM step count")
    parser.add_argument("--guidance-scale", type=float, help="Override the guidance scale")
    parser.add_argument("--eta", type=float, help="Override DDIM stochasticity")
    parser.add_argument("--t-start", type=int, help="SDEdit start timestep; must lie in (0,T)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the sampling noise")
    parser.add_argument("--device", default=None, help="Defaults to cuda when available")
    parser.add_argument("--output", type=Path, default=Path("runs/samples.jsonl"))


def run(args: argparse.Namespace) -> None:
    """Generate candidates and write one JSON record per line."""
    import torch

    from ..checkpoints import load_inference_checkpoint
    from ..data.labels import signed_directions_to_ids
    from ..generation import generate

    checkpoint = load_inference_checkpoint(
        args.checkpoint,
        sampling_overrides={
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "eta": args.eta,
            "t_start": args.t_start,
        },
    )
    ids = signed_directions_to_ids(args.directions, checkpoint.config.data.num_objectives)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    records = generate(
        checkpoint,
        args.sequences,
        ids,
        num_samples=args.num_samples,
        device=device,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"Wrote {len(records)} candidates to {args.output}")
