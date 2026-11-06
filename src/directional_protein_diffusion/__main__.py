"""Support ``python -m directional_protein_diffusion`` and ``accelerate launch -m``."""

from __future__ import annotations

from .cli.main import main

if __name__ == "__main__":
    main()
