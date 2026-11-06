"""Exercise the installed CLI end to end, without downloads and outside the repository."""

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from directional_protein_diffusion.config import ProjectConfig
from directional_protein_diffusion.data.synthetic import SYNTHETIC_OBJECTIVE_NAMES
from directional_protein_diffusion.data.variants import read_variants

MODULE = "directional_protein_diffusion"
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
}


def run_cli(*arguments: str, cwd: Path, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    """Invoke the package as a module from ``cwd``, which is never the repository root.

    Running outside the source tree is the point: an installed package must not depend
    on the working directory.
    """
    return subprocess.run(
        [sys.executable, "-m", MODULE, *arguments],
        cwd=cwd,
        env={**os.environ, **OFFLINE_ENV},
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_offline_train_and_sample(local_esm: Path, tmp_path: Path) -> None:
    config = ProjectConfig()
    config.model.name = str(local_esm)
    config.model.local_files_only = True
    config.data.synthetic_size = 6
    config.data.sequence_length = 6
    config.diffusion.timesteps = 8
    config.sampling.steps = 2
    config.training.batch_size = 6
    config.training.epochs = 1
    config.training.output_dir = str(tmp_path / "run")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config.to_dict()))
    run_cli("train", "--config", str(path), cwd=tmp_path)
    checkpoint = tmp_path / "run/epoch_0001/inference.pt"
    assert checkpoint.exists()
    split = json.loads((tmp_path / "run/split.json").read_text())
    assert {v["sequence"] for v in split["train"]}.isdisjoint(
        v["sequence"] for v in split["validation"]
    )
    for mode in ("ddim", "sdedit"):
        output = tmp_path / f"{mode}.jsonl"
        arguments = [
            "sample",
            "--checkpoint",
            str(checkpoint),
            "--sequences",
            "ACDEFG",
            "--directions",
            "1",
            "-1",
            "--num-samples",
            "2",
            "--device",
            "cpu",
            "--output",
            str(output),
        ]
        if mode == "sdedit":
            arguments += ["--t-start", "4"]
        run_cli(*arguments, cwd=tmp_path)
        records = [json.loads(line) for line in output.read_text().splitlines()]
        assert len(records) == 2
        assert all(len(row["sequence"]) == 6 and row["direction_ids"] == [2, 0] for row in records)


def test_synthetic_command_writes_a_loadable_csv(tmp_path: Path) -> None:
    output = tmp_path / "synthetic.csv"
    run_cli(
        "synthetic",
        "--output",
        str(output),
        "--num-variants",
        "6",
        "--length",
        "8",
        cwd=tmp_path,
        timeout=120,
    )
    variants = read_variants(output, SYNTHETIC_OBJECTIVE_NAMES)
    assert len(variants) == 6
    assert all(len(v.sequence) == 8 for v in variants)


def test_help_does_not_import_torch(tmp_path: Path) -> None:
    """Keep the subcommand modules free of import-time heavy dependencies."""
    probe = (
        "import sys;"
        f"from {MODULE}.cli.main import build_parser;"
        "build_parser();"
        "print('torch' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env={**os.environ, **OFFLINE_ENV},
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.stdout.strip() == "False"
    run_cli("--help", cwd=tmp_path, timeout=120)
