"""Training loop for conditional ESM-2 latent diffusion, importable without the CLI."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .checkpoints import INFERENCE_FILENAME, save_inference_checkpoint
from .config import ProjectConfig
from .data.pairs import PairBatch, PairedVariantDataset, make_dataloader
from .data.synthetic import generate_synthetic_variants
from .data.variants import read_variants, split_variants
from .factory import build_models
from .models.esm import FrozenESM2


def encode_pair_batch(esm: FrozenESM2, batch: PairBatch) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return target/source [B,L,D], directions [B,j], and bool valid mask [B,L]."""
    source, target = esm.encode(batch["source"]), esm.encode(batch["target"])
    if not torch.equal(source.mask, target.mask):
        raise ValueError("Source and target must be aligned with equal lengths")
    return (
        target.latents,
        source.latents,
        batch["directions"].to(source.latents.device),
        source.mask,
    )


def run_training(config: ProjectConfig) -> Path:
    """Run the research training scaffold and return the last checkpoint directory.

    Accelerate supports single-device execution and data-parallel launch. Frozen ESM
    encoding is performed without gradients on each process's device. Checkpoints
    include an inference payload and Accelerate optimizer/scheduler/RNG state.
    """
    config.validate()
    accelerator = Accelerator(mixed_precision=config.training.mixed_precision)
    set_seed(config.seed)
    if config.data.csv_path:
        variants = read_variants(config.data.csv_path, config.data.objective_names)
    else:
        if config.data.num_objectives != 2:
            raise ValueError("Built-in synthetic data has exactly two objectives; supply a CSV")
        variants = generate_synthetic_variants(
            config.data.synthetic_size,
            config.data.sequence_length,
            config.seed,
        )
    train_variants, val_variants = split_variants(variants, config.data.val_fraction, config.seed)
    train_data = PairedVariantDataset(train_variants, config.data.tau, config.data.bidirectional)
    val_data = PairedVariantDataset(val_variants, config.data.tau, config.data.bidirectional)
    train_loader = make_dataloader(
        train_data,
        config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        seed=config.seed,
    )
    val_loader = make_dataloader(
        val_data,
        config.training.batch_size,
        num_workers=config.training.num_workers,
        seed=config.seed,
    )
    esm, diffusion = build_models(config)
    esm.to(accelerator.device)
    # Persist the resolved Hub revision when available, rather than a mutable branch.
    revision = getattr(esm.model.config, "_commit_hash", None)
    if revision:
        config.model.revision = revision
    optimizer = AdamW(
        diffusion.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.training.epochs)
    diffusion, optimizer, train_loader, val_loader = accelerator.prepare(
        diffusion,
        optimizer,
        train_loader,
        val_loader,
    )
    accelerator.register_for_checkpointing(scheduler)
    output = Path(config.training.output_dir)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
        (output / "split.json").write_text(
            json.dumps(
                {
                    "train": [asdict(v) for v in train_variants],
                    "validation": [asdict(v) for v in val_variants],
                },
                indent=2,
            )
        )
    accelerator.wait_for_everyone()
    last_checkpoint = output
    for epoch in range(1, config.training.epochs + 1):
        diffusion.train()
        train_totals = torch.zeros(2, device=accelerator.device)
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            args = encode_pair_batch(esm, batch)
            with accelerator.autocast():
                loss = diffusion(*args)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(diffusion.parameters(), config.training.gradient_clip)
            optimizer.step()
            size = len(batch["source"])
            train_totals += torch.stack((loss.detach().float() * size, loss.new_tensor(size)))
        scheduler.step()
        train_totals = accelerator.reduce(train_totals, reduction="sum")
        diffusion.eval()
        validation: list[Tensor] = []
        with torch.no_grad():
            for batch in val_loader:
                with accelerator.autocast():
                    losses = diffusion(*encode_pair_batch(esm, batch), reduction="none")
                validation.append(accelerator.gather_for_metrics(losses))
        metrics = {
            "epoch": epoch,
            "train_mse": (train_totals[0] / train_totals[1]).item(),
            "val_mse": torch.cat(validation).mean().item(),
            "learning_rate": scheduler.get_last_lr()[0],
        }
        accelerator.print(json.dumps(metrics))
        if accelerator.is_main_process:
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(metrics) + "\n")
        if epoch % config.training.checkpoint_every == 0 or epoch == config.training.epochs:
            last_checkpoint = output / f"epoch_{epoch:04d}"
            accelerator.save_state(str(last_checkpoint))
            state = accelerator.get_state_dict(diffusion)
            if accelerator.is_main_process:
                save_inference_checkpoint(
                    last_checkpoint / INFERENCE_FILENAME,
                    config,
                    state,
                    epoch,
                    metrics,
                )
            accelerator.wait_for_everyone()
    accelerator.end_training()
    return last_checkpoint
