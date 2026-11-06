# Architecture and code map

Where things live, the dependency rules between them, the conventions every tensor obeys, and where to extend. The mathematics behind these choices is in [method.md](method.md).

## Repository layout

```text
├── README.md         # Introduction, installation, usage
├── docs/             # Method, data, usage, architecture, status
├── configs/
│   └── default.yaml  # Development defaults, not tuned settings
├── src/              # The directional_protein_diffusion package
├── tests/            # Offline tests using a tiny random ESM-2
├── .github/
│   └── workflows/    # GitHub Actions: lint and tests on CPU
├── environment.yml   # Conda environment recipe
└── pyproject.toml    # Package metadata, dependencies, tool configuration
```

## Package layout

All importable code is one package under `src/`:

```text
src/directional_protein_diffusion/
├── constants.py      # Amino-acid alphabet and direction IDs; no internal dependencies
├── config.py         # Typed configuration dataclasses and validation
├── factory.py        # Shared model construction
├── checkpoints.py    # Inference payload format, written and read in one place
├── metrics.py        # Hamming distance, seed recovery, ensemble entropy
├── training.py       # Training loop, importable without the CLI
├── generation.py     # Candidate generation, importable without the CLI
├── data/
│   ├── variants.py   # Variant records, CSV round-trip, unique-sequence splitting
│   ├── labels.py     # Directional label contract and the signed CLI mapping
│   ├── pairs.py      # PairedVariantDataset, collation, DataLoader construction
│   └── synthetic.py  # Two-objective mock library
├── models/
│   ├── esm.py        # FrozenESM2: encoding and lm_head decoding
│   ├── conditioning.py  # Time and direction embeddings, adaptive layer norm
│   └── denoiser.py   # ConditionalESMBlock and ConditionalDenoiser
├── diffusion/
│   ├── schedule.py   # Cumulative noise schedules
│   └── gaussian.py   # GaussianDiffusion: q_sample, masked x0 loss, CFG, sampler
└── cli/
    ├── main.py       # protdiff dispatcher
    └── train.py · sample.py · synthetic.py
```

Two packaging decisions are deliberate:

- **One import namespace.** Everything lives under `directional_protein_diffusion`, so installing the project never publishes generic top-level names. Nothing else in the repository is importable.
- **`src/` layout.** Because the package is not at the repository root, it must be installed before anything can import it, and tests therefore exercise the *installed distribution* rather than the working directory. This makes the "works in the repo, breaks once installed" failure mode impossible instead of merely unlikely.

## Dependency layers

Imports point downward only. Within the package, imports are written relative (`from ..constants import NULL`), which keeps lines short given the package name.

```text
layer 4   cli/                           argument parsing only; heavy imports deferred
layer 3   training.py  generation.py     orchestration, importable without argparse
layer 2   factory.py  checkpoints.py     model assembly and serialization
layer 1   data/   models/   diffusion/   domain subsystems, mutually independent
layer 0   constants.py   config.py       no internal dependencies at all
```

Two rules matter most, and `tests/test_layering.py` enforces them by parsing the import graph:

- **Domain vocabulary lives at layer 0.** The amino-acid alphabet and the direction IDs are in `constants.py`, so `models/` and `diffusion/` never import from `data/` to learn what `<null>` is. Nothing in `data/` imports `models/` or `diffusion/` either; the three subsystems are siblings.
- **Nothing imports `cli/`.** The command line is a consumer of the library, never a dependency of it.

## Data flow

```text
CSV or synthetic
   │  read_variants / generate_synthetic_variants   → list[Variant]
   ▼
split_variants                                      → train / val variants (sequence-disjoint)
   │
   ▼
PairedVariantDataset → collate_pairs                → {source: str[B], target: str[B], directions: int64[B,j]}
   │
   ▼
FrozenESM2.encode (no_grad, in the training loop)   → latents float[B,L,D], mask bool[B,L]
   │
   ▼
GaussianDiffusion.forward                           → masked x0 MSE, scalar or per-sample
   │
   └── sampling path: GaussianDiffusion.sample      → predicted clean latents float[B,L,D]
                      FrozenESM2.decode             → candidate sequences str[B]
```

Strings stay on the CPU and are tokenized inside the frozen encoder; only latents, labels, and masks reach the device. Encoding deliberately happens in the training loop rather than in DataLoader workers, so worker processes never need a copy of the model.

## Conventions

These hold everywhere; code that violates them is a bug.

| Convention | Statement |
| --- | --- |
| Latent shape | `[B, L, D]` float. `L` counts residues only — BOS/EOS are stripped after encoding, never diffused |
| Mask | `bool[B, L]`, `True` means a valid residue. Every row must have at least one valid position |
| Padding | Zero-filled in latents, excluded from attention via an additive mask, and excluded from the loss |
| Time indexing | `t = 0` is the clean state; `t = 1..T` are noisy. `alpha_bar` has `T+1` entries with `alpha_bar[0] == 1`, so `q_sample(x, 0)` is exactly the identity. Training samples `t` from `1..T` |
| Direction IDs | `0 = <dec>`, `1 = <nc>`, `2 = <inc>`, `3 = <null>`, defined once in `constants.py`. Measured labels are always in `[0,2]`; `3` appears only through CFG dropout or guidance |
| CLI vs. internal | `protdiff sample` takes signed `-1/0/+1`; datasets and checkpoints store internal IDs |
| Prediction target | $x_0$, the clean latent. Never noise, never velocity |
| Latent range | Unconstrained. No clipping to a pixel-style range, and no rescaling |
| Frozen vs. trainable | The encoder and `lm_head` are frozen; everything in `GaussianDiffusion` is trainable |
| Decoding | Argmax restricted to the 20 canonical amino acids, so special tokens cannot be emitted |
| Checkpoint format | Owned by `checkpoints.py`, including the version constant. Writers and readers never re-spell the field names |

## Enforced invariants

Several requirements are structural rather than documented hopes:

- **The encoder cannot be trained.** `FrozenESM2` calls `requires_grad_(False)` and overrides `train()` so that a parent module's `train()` cannot flip it back into training mode — which matters because dropout inside a "frozen" encoder would silently perturb the latents.
- **No parameter aliasing.** Denoiser blocks are deep copies of the pretrained blocks. A test asserts the two parameter sets share no storage, so optimizing the denoiser cannot drift the representation space that decoding depends on.
- **AdaLN starts at the identity.** The modulation projection is zero-initialized with the scale bias set to one, so $\gamma=1$ and $\beta=0$ at construction. A test asserts a freshly built conditional block reproduces its pretrained source block numerically.
- **Padding cannot influence training.** Loss reduction masks invalid positions and normalizes per sample by valid residues times hidden width. A test writes large values into padded positions and asserts the loss is unchanged.
- **Measured labels are range-checked.** Both the loss and the sampler reject direction IDs outside `[0,2]`, so `<null>` can never enter as if it were an experimental observation.
- **Dropout never mutates inputs.** `drop_directions` returns a new tensor, so a batch's labels remain intact for logging and for the conditional branch.
- **The sampler is side-effect free.** It runs under `no_grad`, restores the caller's train/eval mode when it finishes, and leaves the source latents untouched.
- **Splits precede pairing.** Unique sequences are partitioned before any pair exists, and duplicate sequences are rejected at load so one protein cannot occupy both partitions.
- **The layering holds.** The import graph is parsed in tests, so the dependency rules above cannot rot quietly.
- **`protdiff --help` does not import PyTorch.** Command modules declare arguments at import time and defer library imports into `run()`. A test asserts `torch` is absent from `sys.modules` after the parser is built.

## Denoiser internals

`ConditionalDenoiser` composes:

1. `input_projection`: `Linear(2D → D)` applied to the feature-axis concatenation of noisy target and clean seed, then re-masked.
2. `time_embedding` + `direction_embedding`, summed into one condition vector `[B, D]`.
3. A stack of `ConditionalESMBlock`s. Each keeps the pretrained self-attention, attention output, intermediate, and output submodules — including rotary positions and residual paths — and replaces only the two pre-norms with `AdaptiveLayerNorm` wrapping the original `LayerNorm`.
4. `final_norm`: a copy of ESM-2's `emb_layer_norm_after`, kept so the output stays in the normalized space the frozen head consumes.
5. `output_projection`: `Linear → GELU → Linear`, followed by re-masking.

Depth defaults to every available pretrained block and is capped by the loaded checkpoint, so `model.num_layers` cannot request more blocks than exist.

## Sampler internals

`GaussianDiffusion.sample` builds a step trajectory by rounding `linspace(start, 0, steps + 1)`, then at each transition:

1. Compute the guided clean-latent prediction.
2. If the next timestep is `0`, take the prediction as the result.
3. Otherwise recover the implied noise from the current state and prediction, form the DDIM update toward the next timestep, add the $\eta$ term when `eta > 0`, and re-mask.

`guided_prediction` short-circuits: one network call at `w = 1`, one at `w = 0`, two otherwise. `start` is `T` for full DDIM or `t_start` for SDEdit, where the initial state is a noised **copy** of the seed while the clean seed remains the condition.

## Entry points

The top-level `__init__.py` stays light on purpose — importing the package does not pull in PyTorch — so the API is imported from its own modules:

| Entry point | Module |
| --- | --- |
| `load_config`, `ProjectConfig` | `config` |
| `build_models` | `factory` |
| `run_training` | `training` |
| `generate` | `generation` |
| `load_inference_checkpoint`, `save_inference_checkpoint` | `checkpoints` |
| `signed_directions_to_ids`, `directional_labels` | `data.labels` |

A worked example is in [usage.md](usage.md#programmatic-use).

## Extension points

Known-missing capabilities, with the reason each is deferred rather than hidden:

| Extension | Current state |
| --- | --- |
| Candidate selection | No module yet. `generate` returns plain dictionaries so filtering, deduplication, and ranking can consume them directly |
| Latent caching | Sequences are re-encoded every epoch. Caching is an obvious speedup for a fixed library |
| Insertions and deletions | Source and target must be equal length and aligned. Supporting indels is a design change — the per-position seed concatenation assumes alignment |
| Homology- or round-disjoint splits | Only sequence-disjoint splitting is implemented. See [data.md](data.md#splitting-before-pairing) |
| Hard edit-distance filtering | `t_start` is a soft control; decoded candidates are reported with Hamming distance but never rejected |
| Residue sampling policies | Decoding is argmax only; temperature and top-$k$ are not implemented |
| Pair subsampling | All within-partition pairs are enumerated, which is $O(m^2)$ |
| Training resume | Accelerate state is saved but no resume subcommand exists |
| Denoiser construction | `ConditionalDenoiser` takes a `FrozenESM2` to read its width and blocks. Accepting those primitives directly would decouple it further |

A new subcommand is one module in `cli/` exposing `HELP`, `add_arguments`, and `run`, registered in `cli/main.COMMANDS`.

When adding one of these, update [status.md](status.md), this file, and — if the behavior changes a scientific contract — [method.md](method.md) in the same change.
