"""Frozen Hugging Face ESM-2 encoder and LM head; residue tensors exclude BOS/EOS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from transformers import EsmForMaskedLM, EsmTokenizer

from ..constants import AMINO_ACIDS


@dataclass
class EncodedBatch:
    """Residue latents [B,L,D] and bool mask [B,L], where True denotes a real residue."""

    latents: Tensor
    mask: Tensor


class FrozenESM2(nn.Module):
    """Keep pretrained ESM-2 and its tied language-model head frozen and in eval mode."""

    def __init__(self, model: EsmForMaskedLM, tokenizer: EsmTokenizer) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_size = model.config.hidden_size
        ids = [tokenizer.convert_tokens_to_ids(aa) for aa in AMINO_ACIDS]
        if len(set(ids)) != 20 or tokenizer.unk_token_id in ids:
            raise ValueError("Tokenizer must contain all 20 canonical amino acids")
        self.register_buffer("amino_acid_ids", torch.tensor(ids, dtype=torch.long))
        self.requires_grad_(False)
        self.train(False)

    @classmethod
    def from_pretrained(
        cls,
        name: str = "facebook/esm2_t6_8M_UR50D",
        *,
        revision: str = "main",
        local_files_only: bool = False,
    ) -> FrozenESM2:
        """Load a Hub ID or local directory; never download at module import time."""
        tokenizer = EsmTokenizer.from_pretrained(
            name,
            revision=revision,
            local_files_only=local_files_only,
        )
        model = EsmForMaskedLM.from_pretrained(
            name,
            revision=revision,
            local_files_only=local_files_only,
        )
        return cls(model, tokenizer)

    def train(self, mode: bool = True) -> FrozenESM2:
        """Always remain in eval mode, including when a parent calls train()."""
        super().train(False)
        return self

    @torch.no_grad()
    def encode(self, sequences: Sequence[str]) -> EncodedBatch:
        """Encode B nonempty canonical sequences to final normalized [B,L,D] latents.

        Right padding is removed from the representation contract via a bool mask.
        BOS/EOS are seen by the frozen encoder but never diffused or scored in MSE.
        """
        if not sequences or any(not s or set(s) - set(AMINO_ACIDS) for s in sequences):
            raise ValueError("Expected nonempty canonical amino-acid sequences")
        device = next(self.model.parameters()).device
        tokens = self.tokenizer(list(sequences), padding=True, return_tensors="pt")
        tokens = {key: value.to(device) for key, value in tokens.items()}
        hidden = self.model.esm(**tokens, return_dict=True).last_hidden_state
        lengths = torch.tensor([len(s) for s in sequences], device=device)
        length = max(map(len, sequences))
        mask = torch.arange(length, device=device)[None, :] < lengths[:, None]
        latents = hidden[:, 1 : length + 1, :].masked_fill(~mask[:, :, None], 0)
        return EncodedBatch(latents, mask)

    @torch.no_grad()
    def decode_logits(self, latents: Tensor) -> Tensor:
        """Apply the frozen LM head to [B,L,D], returning vocabulary logits [B,L,V]."""
        if latents.ndim != 3 or latents.shape[-1] != self.hidden_size:
            raise ValueError("latents must have shape [B,L,hidden_size]")
        return self.model.lm_head(latents.to(next(self.model.parameters()).dtype))

    @torch.no_grad()
    def decode(self, latents: Tensor, mask: Tensor) -> list[str]:
        """Argmax over only the 20 canonical amino acids, returning B residue strings."""
        if mask.shape != latents.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be bool [B,L]")
        logits = self.decode_logits(latents).index_select(-1, self.amino_acid_ids)
        indices = logits.argmax(dim=-1).cpu()
        return [
            "".join(AMINO_ACIDS[i] for i in row[valid].tolist())
            for row, valid in zip(indices, mask.cpu())
        ]
