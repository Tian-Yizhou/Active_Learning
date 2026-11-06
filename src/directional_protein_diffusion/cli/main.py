"""Command-line entry point dispatching to one module per subcommand."""

from __future__ import annotations

import argparse
from typing import Sequence

from .. import __version__
from . import sample, synthetic, train

COMMANDS = {"train": train, "sample": sample, "synthetic": synthetic}
"""Subcommand name to the module providing ``HELP``, ``add_arguments`` and ``run``."""


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser, delegating arguments to each command module."""
    parser = argparse.ArgumentParser(
        prog="protdiff",
        description=(
            "Multi-objective protein sequence optimization by conditional latent "
            "diffusion in ESM-2 embedding space."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, module in COMMANDS.items():
        subparser = subparsers.add_parser(name, help=module.HELP, description=module.__doc__)
        module.add_arguments(subparser)
        subparser.set_defaults(handler=module.run)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Parse arguments and dispatch; ``argv=None`` reads ``sys.argv``."""
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
