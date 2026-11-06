"""Parse the import graph so the documented dependency layers cannot rot quietly."""

import ast
from pathlib import Path

import pytest

import directional_protein_diffusion

ROOT = Path(directional_protein_diffusion.__file__).parent
PACKAGE = directional_protein_diffusion.__name__
MODULES = sorted(path for path in ROOT.rglob("*.py"))

# Sibling subsystems that must not import one another in either direction.
FORBIDDEN = [("models", "data"), ("diffusion", "data"), ("data", "models"), ("data", "diffusion")]


def package_parts(path: Path) -> list[str]:
    """Return the dotted parts of the package a module's relative imports resolve against.

    Dropping the final component is correct for both cases: ``models/denoiser.py`` resolves
    against ``models``, and ``models/__init__.py`` — being the package itself — also does.
    """
    return list(path.relative_to(ROOT).with_suffix("").parts)[:-1]


def internal_imports(path: Path) -> set[str]:
    """Return intra-package modules imported by one file, with relative imports resolved."""
    names: set[str] = set()
    base = package_parts(path)
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                prefix = base[: len(base) - (node.level - 1)]
                target = [*prefix, *(node.module.split(".") if node.module else [])]
            elif node.module and node.module.split(".")[0] == PACKAGE:
                target = node.module.split(".")[1:]
            else:
                continue
            if target:
                names.add(".".join(target))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == PACKAGE:
                    names.add(".".join(alias.name.split(".")[1:]))
    return {name for name in names if name}


def test_module_discovery_sees_the_whole_package() -> None:
    assert len(MODULES) > 15
    assert ROOT / "constants.py" in MODULES


@pytest.mark.parametrize("name", ["constants", "config"])
def test_bottom_layer_has_no_internal_dependencies(name: str) -> None:
    assert internal_imports(ROOT / f"{name}.py") == set()


@pytest.mark.parametrize(("subsystem", "forbidden"), FORBIDDEN)
def test_sibling_subsystems_stay_independent(subsystem: str, forbidden: str) -> None:
    for path in sorted((ROOT / subsystem).rglob("*.py")):
        offending = {name for name in internal_imports(path) if name.split(".")[0] == forbidden}
        assert not offending, f"{path.name} imports {sorted(offending)}"


def test_nothing_outside_the_cli_imports_the_cli() -> None:
    for path in MODULES:
        if path.parent.name == "cli" or path.name == "__main__.py":
            continue
        offending = {name for name in internal_imports(path) if name.split(".")[0] == "cli"}
        assert not offending, f"{path.name} imports {sorted(offending)}"
