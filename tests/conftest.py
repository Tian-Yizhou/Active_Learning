"""Tiny local ESM-2 fixtures: no network, pretrained downloads, or GPU required."""

from pathlib import Path

import pytest
import torch
from transformers import EsmConfig, EsmForMaskedLM, EsmTokenizer

from directional_protein_diffusion.constants import AMINO_ACIDS
from directional_protein_diffusion.models.esm import FrozenESM2

torch.set_num_threads(1)


@pytest.fixture
def local_esm(tmp_path: Path) -> Path:
    """Persist a random two-layer ESM-2 plus tokenizer for full CLI integration tests."""
    path = tmp_path / "tiny_esm"
    path.mkdir()
    vocab = ["<cls>", "<pad>", "<eos>", "<unk>", *AMINO_ACIDS, "<mask>"]
    (path / "vocab.txt").write_text("\n".join(vocab) + "\n")
    tokenizer = EsmTokenizer(vocab_file=str(path / "vocab.txt"))
    tokenizer.save_pretrained(path)
    config = EsmConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        pad_token_id=1,
        mask_token_id=len(vocab) - 1,
        position_embedding_type="rotary",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        token_dropout=True,
    )
    EsmForMaskedLM(config).save_pretrained(path)
    return path


@pytest.fixture
def esm(local_esm: Path) -> FrozenESM2:
    """Load the local random checkpoint through the production wrapper."""
    return FrozenESM2.from_pretrained(str(local_esm), local_files_only=True)
