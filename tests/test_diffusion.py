"""Schedule indexing, masked x0 loss, sampler algebra, and CFG endpoints."""

import pytest
import torch
from torch import Tensor, nn

from directional_protein_diffusion.diffusion.gaussian import GaussianDiffusion
from directional_protein_diffusion.diffusion.schedule import make_alpha_bar
from directional_protein_diffusion.models.conditioning import drop_directions
from directional_protein_diffusion.models.denoiser import ConditionalDenoiser
from directional_protein_diffusion.models.esm import FrozenESM2


@pytest.mark.parametrize("schedule", ["cosine", "linear"])
def test_schedule_is_clean_at_zero_and_decreasing(schedule: str) -> None:
    alpha_bar = make_alpha_bar(16, schedule)
    assert alpha_bar.shape == (17,)
    assert alpha_bar[0] == 1
    assert torch.all(alpha_bar[1:] < alpha_bar[:-1])
    assert torch.all(alpha_bar > 0)
    with pytest.raises(ValueError):
        make_alpha_bar(16, "quadratic")
    with pytest.raises(ValueError):
        make_alpha_bar(1)


def test_loss_backward_and_padding_invariance(esm: FrozenESM2) -> None:
    diffusion = GaussianDiffusion(ConditionalDenoiser(esm, 2), timesteps=8, p_uncond=0)
    source = esm.encode(["ACDE", "AC"])
    target = esm.encode(["ACDF", "AD"])
    directions = torch.tensor([[2, 0], [1, 2]])
    torch.manual_seed(4)
    loss = diffusion(target.latents, source.latents, directions, source.mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert diffusion.denoiser.input_projection.weight.grad is not None
    assert all(p.grad is None for p in esm.parameters())
    altered = target.latents.clone()
    altered[~target.mask] = 10000
    torch.manual_seed(4)
    same = diffusion(altered, source.latents, directions, source.mask)
    torch.testing.assert_close(loss, same)
    zero = torch.zeros(2, dtype=torch.long)
    torch.testing.assert_close(diffusion.q_sample(target.latents, zero), target.latents)


def test_loss_rejects_null_as_a_measured_label(esm: FrozenESM2) -> None:
    diffusion = GaussianDiffusion(ConditionalDenoiser(esm, 2), timesteps=8)
    encoded = esm.encode(["ACDE"])
    with pytest.raises(ValueError, match=r"\[0,2\]"):
        diffusion(
            encoded.latents,
            encoded.latents,
            torch.tensor([[3, 3]]),
            encoded.mask,
        )


class OracleDenoiser(nn.Module):
    """Known clean target for testing sampler algebra independently of model quality."""

    def forward(self, state: Tensor, t: Tensor, source: Tensor, d: Tensor, mask: Tensor) -> Tensor:
        """Return source for conditional predictions and zero for null predictions."""
        return source * (d[:, :1] != 3).to(source.dtype)[:, :, None]


@pytest.mark.parametrize("t_start", [None, 4])
@pytest.mark.parametrize("eta", [0.0, 0.5])
def test_sampling_recovers_oracle_and_preserves_source(t_start: int | None, eta: float) -> None:
    diffusion = GaussianDiffusion(OracleDenoiser(), timesteps=8)
    source = torch.randn(2, 5, 4)
    original = source.clone()
    mask = torch.ones(2, 5, dtype=torch.bool)
    directions = torch.tensor([[2, 0], [0, 2]])
    result = diffusion.sample(
        source, directions, mask, steps=4, t_start=t_start, eta=eta, guidance_scale=1
    )
    torch.testing.assert_close(result, source)
    torch.testing.assert_close(source, original)
    assert diffusion.training  # sampler restores caller mode
    t = torch.ones(2, dtype=torch.long)
    for scale in [0.0, 1.0, 2.0]:
        predicted = diffusion.guided_prediction(source, t, source, directions, mask, scale)
        torch.testing.assert_close(predicted, source * scale)


def test_cfg_drops_whole_vector_without_mutation() -> None:
    directions = torch.tensor([[0, 2], [1, 0]])
    torch.testing.assert_close(drop_directions(directions, 0), directions)
    assert (drop_directions(directions, 1) == 3).all()
    assert directions.tolist() == [[0, 2], [1, 0]]
