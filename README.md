# Directional Protein Diffusion

**Multi-objective protein sequence optimization by conditional latent diffusion in ESM-2 embedding space.**

Given a seed protein sequence and a requested direction for each measured objective — increase, decrease, or no change — this project generates candidate mutant sequences. It is built for the low-data regime of experimental directed evolution, where a round of measurements yields too few variants to fit a reliable fitness predictor.

## Method

Training data is built from **pairwise comparisons** between measured variants. For an ordered pair and each objective, the measured change is reduced to a direction — decrease, no change, or increase — according to whether it exceeds that objective's experimental noise threshold. A pair is therefore labeled by a vector of discrete directions, one per objective. From *M* measured variants this yields up to *M*(*M*−1) ordered training examples, and it requires no numerical fitness model.

Generation is a conditional latent diffusion process. A frozen ESM-2 encoder maps both sequences of a pair to residue embeddings. Only the target embedding is noised; the seed embedding stays clean and conditions every denoising step. A denoiser assembled from copies of pretrained ESM-2 attention blocks predicts the clean target embedding directly, with the diffusion timestep and the requested directions injected through adaptive layer norm inside each block. Training minimizes mean squared error over valid residue positions.

At inference the reverse process runs either from pure noise (DDIM) or from a partially noised seed (SDEdit), the latter keeping candidates closer to a known-good sequence. Classifier-free guidance strengthens the directional signal. The predicted embedding is decoded by the frozen ESM-2 language-model head, restricted to the 20 canonical amino acids.

A requested direction is an intention, not a prediction: whether a candidate actually improves a property is established by experiment. [docs/method.md](docs/method.md) gives the full specification.

## Installation

Requires Python 3.11 or newer.

```bash
conda env create --prefix ./.conda-env -f environment.yml
conda activate "${PWD}/.conda-env"
```

This installs the project and provides the `protdiff` command. Select a PyTorch build that matches the target machine before GPU training.

## Quick Start

Train with the default configuration, which uses built-in synthetic data and exercises the pipeline end to end:

```bash
protdiff train --config configs/default.yaml
```

Generate candidates from the resulting checkpoint, here requesting an increase in the first objective and a decrease in the second:

```bash
protdiff sample --checkpoint runs/default/epoch_0010/inference.pt \
  --sequences MKTAYIAKQRQISFVKSHFSRQ --directions 1 -1 --num-samples 8
```

Each run records its resolved configuration, variant split, per-epoch metrics and checkpoints under `training.output_dir`. Generated candidates are written as JSONL, one record per candidate, carrying the seed, the requested directions, the edit distance and the sampling settings used.

To train on measured data, set `data.csv_path` to a CSV of variants and objective values, then configure the objective names and their noise thresholds. [docs/data.md](docs/data.md) describes the input schema and how to choose thresholds; [docs/usage.md](docs/usage.md) is the configuration and command reference.

## References

This work builds on three published sources:

1. **Wang et al.**, *Active learning-guided optimization of cell-free biosensors for lead testing in drinking water*, Nature Communications — pairwise experimental supervision, multi-objective directional labels, and the iterative experimental cycle. Its ProtT5 generator is the baseline replaced here. [Code](https://github.com/ShuklaGroup/multiobjective_controlled_extrapolation)
2. **Chen et al.**, *AMP-Diffusion: Integrating Latent Diffusion with Protein Language Models for Antimicrobial Peptide Generation*, NeurIPS 2023 Generative AI and Biology Workshop — frozen ESM-2 latents, Gaussian latent diffusion, reuse of pretrained attention blocks, and `lm_head` decoding. [Preprint](https://www.biorxiv.org/content/10.1101/2024.03.03.583201v1) · [Code](https://github.com/programmablebio/amp-diffusion)
3. **ESM-2** (Lin et al., Meta AI) — the frozen protein language model providing the latent space and the decoding head. [Code](https://github.com/facebookresearch/esm)

This project departs from the first two deliberately: `<nc>` is retained as an explicit third direction instead of filtering comparisons that fall within experimental noise, the denoiser predicts the clean latent under MSE rather than noise under an L1 objective, and conditioning enters through joint time-and-direction adaptive layer norm inside each attention block rather than a time-only transform applied before them.

## License

MIT — see [LICENSE](LICENSE).
