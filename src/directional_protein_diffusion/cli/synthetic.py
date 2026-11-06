"""Write the built-in synthetic two-objective library as a CSV for smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

HELP = "Write synthetic two-objective variants as a CSV"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register synthetic data arguments on the given parser."""
    parser.add_argument("--output", type=Path, default=Path("data/synthetic.csv"))
    parser.add_argument("--num-variants", type=int, default=40)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)


def run(args: argparse.Namespace) -> None:
    """Generate the mock library and write it in the schema the loader expects."""
    from ..data.synthetic import SYNTHETIC_OBJECTIVE_NAMES, generate_synthetic_variants
    from ..data.variants import write_variants

    variants = generate_synthetic_variants(args.num_variants, args.length, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_variants(args.output, variants, SYNTHETIC_OBJECTIVE_NAMES)
    print(f"Wrote {len(variants)} variants to {args.output}")
