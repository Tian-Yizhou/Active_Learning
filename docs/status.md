# Implementation status

This document exists to keep two claims apart: *the pipeline runs end to end* and *the pipeline designs proteins*. Only the first is supported today.

Last reviewed: 2026-11-05.

## Summary

| | State |
| --- | --- |
| Pipeline implemented | Yes — paired data, frozen encoder, conditional denoiser, training loop, sampling, checkpointing |
| Offline tests passing | Yes — 44 tests, against a tiny randomly initialized ESM-2 |
| Trained with pretrained ESM-2 weights | **No** |
| Validated on GPU or multi-process | **No** |
| Run on experimental data | **No** |
| Biological or functional validation | **No** |

## What is implemented

- **Paired supervision.** CSV loading with validation, unique-sequence splitting before pairing, directional labels from per-objective thresholds with `<nc>` retained, optional pair reversal, and a synthetic two-objective generator for smoke tests.
- **Frozen encoder and decoder.** A Hugging Face ESM-2 wrapper that cannot be un-frozen or put into training mode, strips BOS/EOS, masks padding, and decodes by argmax over the 20 canonical amino acids.
- **Conditional denoiser.** Feature-axis seed concatenation, sinusoidal time embedding, per-objective direction embeddings, and adaptive layer norm on both pre-norms of attention blocks deep-copied from pretrained ESM-2, initialized to the identity.
- **Training.** Target-only forward noising, $x_0$-prediction under masked MSE, classifier-free-guidance dropout of the whole direction vector, AdamW with per-epoch cosine decay, gradient clipping, and Accelerate integration including mixed-precision options.
- **Sampling.** Full DDIM from noise and SDEdit from a partially noised seed, deterministic at `eta = 0`, with classifier-free guidance and short-circuits at $w \in \{0, 1\}$.
- **Reproducibility plumbing.** Resolved configuration, split manifest, per-epoch metrics, inference checkpoints, and generation records carrying seeds, directions, sampling settings, and checkpoint provenance.
- **Packaging and tooling.** A single installed package under `src/`, the `protdiff` command with `train`, `sample`, and `synthetic` subcommands, importable `run_training` and `generate` for programmatic use, exported type information, and CI running lint and tests on CPU.

## What the tests cover

All tests are offline: they build a two-layer, 16-dimensional ESM-2 with random weights and a local tokenizer, so no download, network access, or GPU is needed.

| Test | What it establishes |
| --- | --- |
| Split precedes pairing | Train and validation sequences are disjoint, the split is deterministic, pair counts match $m(m-1)$, and duplicate sequences are rejected |
| Threshold boundaries and reversal | Labels at and inside the threshold are `<nc>`, reversal swaps `<dec>`/`<inc>`, non-finite values are rejected, and signed CLI directions map correctly |
| Sequence metrics | Hamming distance, seed recovery, and ensemble entropy behave on known inputs |
| CSV round-trip | Variants written and read back preserve IDs, sequences, and objective values, and a mismatched objective count is rejected |
| Configuration validation | Sixteen inconsistent configurations are each rejected, defaults round-trip through YAML and dicts unchanged, and misspelled or non-mapping configurations raise |
| Schedule indexing | Cumulative alphas have `T+1` entries, start at exactly one, decrease strictly, stay positive, and reject unknown schedules |
| Architectural layering | The import graph is parsed: the bottom layer has no internal dependencies, the sibling subsystems stay independent, and nothing outside `cli/` imports it |
| Frozen encoder and block identity | The encoder stays in eval mode with no trainable parameters, a copied conditional block reproduces its pretrained source numerically at identity modulation, and the two share no parameter storage |
| Padding and canonical decode | Latent shapes and masks are correct for ragged batches, a padded sequence's valid positions match its unpadded encoding, padding is zero-filled, and decoded lengths match the mask |
| Loss, gradients, and padding invariance | The loss is finite and differentiable, gradients reach the denoiser and never the encoder, garbage written into padded positions does not change the loss, and `q_sample` at `t = 0` is the identity |
| Sampler algebra | With an oracle denoiser, both DDIM and SDEdit recover the known target for `eta` of `0` and `0.5`; source latents are not mutated; the caller's train/eval mode is restored; guided prediction scales as $w$ |
| CFG dropout | Probability `0` and `1` behave as the endpoints and the input tensor is never mutated |
| End-to-end CLI | A full train → checkpoint → sample cycle succeeds for both DDIM and SDEdit, writing a valid split manifest and candidate records of the right length and direction. It runs from outside the repository, so the installed package cannot depend on the working directory. The synthetic subcommand writes a CSV the loader accepts, and building the argument parser does not import PyTorch |

**What they do not establish:** that generated sequences are plausible proteins, that requested directions correlate with measured outcomes, that the architecture learns anything useful from real data, or that the chosen defaults are reasonable. Random 16-dimensional weights cannot answer any of those questions.

## Not yet validated

- **Pretrained-weight training.** No run has been performed with real ESM-2 weights, so loss scale, convergence behavior, and sensible values for learning rate, epochs, `p_uncond`, guidance scale, and `t_start` are all unknown.
- **Scale.** Only the 8M-parameter `esm2_t6_8M_UR50D` path has been exercised, and only on CPU with tiny inputs. Memory and throughput for larger ESM-2 variants are unmeasured.
- **Distributed and mixed precision.** Accelerate is integrated but has only run single-process on CPU with `mixed_precision: "no"`. Multi-GPU launches, `fp16`, and `bf16` need separate validation.
- **Decoding fidelity.** Whether the frozen `lm_head` decodes predicted latents into sequences at useful accuracy — including how far a predicted latent can drift from the encoder's manifold before decoding degrades — has not been measured.
- **Defaults.** Everything in `configs/default.yaml` is a development placeholder, including the noise thresholds.

## Known limitations

These are design boundaries, not bugs:

- **Equal-length, aligned variants only.** No insertions or deletions. Hamming distance refuses unequal lengths rather than silently tolerating them.
- **Sequence-disjoint splits only.** Near-identical variants can land on opposite sides, which weakens any extrapolation claim. Homology- or round-based splits must be supplied deliberately.
- **No edit budget.** `t_start` controls exploration softly; nothing bounds the mutation count. Hard limits require filtering decoded candidates.
- **Validation loss is not fitness.** It is a stochastic latent reconstruction error, useful for diagnosing training, meaningless as evidence of improvement.
- **No resume CLI**, despite Accelerate state being saved.
- **Re-encoding every epoch**, with no latent cache.
- **$O(m^2)$ pair enumeration**, appropriate for low-$N$ libraries and not for large ones.

A fuller list of deferred capabilities, with pointers into the code, is in [architecture.md](architecture.md#extension-points).

## Suggested next steps

Roughly in dependency order:

1. **A real training run** with pretrained ESM-2 weights on synthetic data, to establish loss scale and sane hyperparameters.
2. **Decoding diagnostics** — reconstruct known targets from their own clean latents, then from predicted latents, and measure recovery. This isolates decoder fidelity from model quality.
3. **Conditioning sanity checks** — verify that flipping a requested direction changes the generated candidates, and that guidance scale moves candidates monotonically away from the null branch.
4. **Experimental data** for a real objective pair, with calibrated thresholds.
5. **Held-out evaluation** on a split that reflects the generalization claim being made, not merely sequence disjointness.
6. **Scale and distributed validation** once the method shows signal at small scale.

Keep this document current when any of these lands. It is the single place where the project's status is recorded, so no claim about what works needs to be reconciled against another file.
