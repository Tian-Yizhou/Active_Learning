"""Multi-objective protein optimization by conditional latent diffusion in ESM-2 space.

This package intentionally keeps its top level light so that importing it, and running
``protdiff --help``, does not pay for PyTorch and Transformers. Import the entry points
from their own modules:

- ``config``: ``ProjectConfig``, ``load_config``
- ``factory``: ``build_models``
- ``training``: ``run_training``
- ``generation``: ``generate``
- ``checkpoints``: ``load_inference_checkpoint``
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("directional-protein-diffusion")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
