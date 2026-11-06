"""Verify pretrained block reuse, frozen weights, masking, and canonical decoding."""

import torch

from directional_protein_diffusion.models.denoiser import ConditionalESMBlock
from directional_protein_diffusion.models.esm import FrozenESM2


def test_frozen_encoder_and_block_identity(esm: FrozenESM2) -> None:
    esm.train()
    assert not esm.training and not esm.model.training
    assert all(not p.requires_grad for p in esm.parameters())
    pretrained = esm.model.esm.encoder.layer[0]
    copied = ConditionalESMBlock(pretrained, esm.hidden_size).eval()
    h, c = torch.randn(2, 5, 16), torch.randn(2, 16)
    mask = torch.zeros(2, 1, 1, 5)
    with torch.no_grad():
        expected = pretrained(h, attention_mask=mask)
        if isinstance(expected, tuple):  # Older Transformers versions return tuples.
            expected = expected[0]
    torch.testing.assert_close(copied(h, c, mask), expected)
    assert {p.data_ptr() for p in copied.parameters()}.isdisjoint(
        p.data_ptr() for p in esm.parameters()
    )


def test_wrapper_padding_and_canonical_decode(esm: FrozenESM2) -> None:
    encoded = esm.encode(["ACDE", "AC"])
    alone = esm.encode(["AC"])
    assert encoded.latents.shape == (2, 4, 16)
    assert encoded.mask.tolist() == [[True] * 4, [True, True, False, False]]
    torch.testing.assert_close(encoded.latents[1, :2], alone.latents[0])
    assert torch.count_nonzero(encoded.latents[1, 2:]) == 0
    assert list(map(len, esm.decode(encoded.latents, encoded.mask))) == [4, 2]
