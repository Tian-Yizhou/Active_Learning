# Usage

Environment setup, the full configuration reference, and what training and sampling read and write. Run every command from the repository root.

## Environment

The project uses a dedicated, project-local Miniconda environment at `.conda-env`:

```bash
conda env create --prefix ./.conda-env -f environment.yml
conda activate "${PWD}/.conda-env"
python -m pytest -q
```

`environment.yml` pins Python 3.11 and installs this package in editable mode with development extras, which pulls PyTorch, Transformers, Accelerate, PyYAML, and NumPy from `pyproject.toml`.

To control the PyTorch build explicitly — for example a CPU-only environment for scaffold validation:

```bash
conda create --prefix ./.conda-env python=3.11 pip
conda activate "${PWD}/.conda-env"
python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
```

Choose a CUDA-matched build before GPU training. The CPU validation environment exercises neither CUDA, mixed precision, nor multi-process execution.

Installing the project puts the `protdiff` command on the path; `python -m directional_protein_diffusion` is an equivalent invocation, and both work from any directory. Because the package lives under `src/`, **installation is required before anything can import it** — a bare clone has no importable module on the path. That is deliberate: it keeps tests honest by exercising the installed distribution instead of the working directory.

### Offline and local checkpoints

By default the frozen encoder is downloaded from the Hugging Face Hub on first use. To run without network access, save a checkpoint locally and point the configuration at the directory:

```yaml
model:
  name: /path/to/esm2_t6_8M_UR50D
  local_files_only: true
```

Hidden width and the number of available attention blocks are read from the checkpoint, so larger ESM-2 variants work without code changes. Setting `HF_HUB_OFFLINE=1` additionally guarantees no network call is attempted — the test suite does exactly this.

## Configuration reference

A run is defined entirely by one YAML file; [`configs/default.yaml`](../configs/default.yaml) is the annotated starting point. Values are validated on load, and misspelled keys are rejected rather than ignored.

### Top level

| Key | Default | Meaning |
| --- | --- | --- |
| `seed` | `42` | Seeds the split, data order, dropout, and noise. Recorded with the run. |

### `model`

| Key | Default | Meaning |
| --- | --- | --- |
| `name` | `facebook/esm2_t6_8M_UR50D` | Hub ID or local directory for the frozen encoder and `lm_head` |
| `revision` | `main` | Hub revision. Pin a commit for reproducible research runs; the resolved commit hash is written back into the saved config when available |
| `local_files_only` | `false` | Forbid downloads |
| `num_layers` | `null` | How many pretrained blocks to copy into the denoiser. `null` copies all of them; a smaller value gives a shallower, cheaper denoiser |

### `data`

| Key | Default | Meaning |
| --- | --- | --- |
| `csv_path` | `null` | Measurement CSV. `null` generates synthetic two-objective data |
| `objective_names` | `[fitness_1, fitness_2]` | CSV column names **in fixed objective order** |
| `num_objectives` | `2` | Must equal the length of `objective_names` and of `tau` |
| `tau` | `[0.05, 0.05]` | Per-objective noise thresholds. Placeholders — calibrate for real data |
| `val_fraction` | `0.2` | Fraction of unique sequences held out, in `(0,1)`; each partition keeps at least two variants |
| `bidirectional` | `true` | Add the reverse orientation of every pair |
| `synthetic_size` | `40` | Variant count for synthetic data |
| `sequence_length` | `32` | Residue length for synthetic data |

See [data.md](data.md) for the CSV contract and threshold guidance.

### `diffusion`

| Key | Default | Meaning |
| --- | --- | --- |
| `timesteps` | `1000` | $T$. Clean state is $t=0$; noisy states are $1..T$ |
| `noise_schedule` | `cosine` | `cosine` or `linear` |
| `p_uncond` | `0.1` | Probability of replacing a sample's whole direction vector with `<null>` during training, enabling classifier-free guidance. The seed is kept in both branches |

### `training`

| Key | Default | Meaning |
| --- | --- | --- |
| `batch_size` | `8` | Pairs per step |
| `epochs` | `10` | Also the period of the cosine learning-rate decay |
| `learning_rate` | `0.0001` | AdamW learning rate |
| `weight_decay` | `0.01` | AdamW weight decay |
| `gradient_clip` | `1.0` | Max gradient norm |
| `num_workers` | `0` | DataLoader workers. Workers only move strings and labels; encoding happens in the main process |
| `mixed_precision` | `"no"` | `no`, `fp16`, or `bf16`, passed to Accelerate |
| `output_dir` | `runs/default` | Run directory |
| `checkpoint_every` | `1` | Checkpoint period in epochs; the final epoch always checkpoints |

### `sampling`

| Key | Default | Meaning |
| --- | --- | --- |
| `steps` | `50` | DDIM steps. Must satisfy `1 <= steps <= t_start or T` |
| `guidance_scale` | `2.0` | $w$. `1` is the plain conditional prediction and costs one network call per step; `0` is the null-direction prediction; `>1` amplifies the directional signal using two calls per step |
| `eta` | `0.0` | `0` is deterministic DDIM given the initial noise; up to `1` adds stochasticity |
| `t_start` | `null` | `null` starts from pure noise. An integer in `(0,T)` enables SDEdit from a partially noised seed |

These are **development defaults, not tuned settings.** Nothing here has been optimized against an experimental objective.

## Training

```bash
protdiff train --config configs/default.yaml
```

Under an Accelerate-configured runtime, including multi-GPU:

```bash
accelerate launch -m directional_protein_diffusion train --config configs/default.yaml
```

Each step encodes the batch's seed and target sequences with the frozen encoder under `no_grad`, noises only the target at a random timestep, predicts the clean target, and takes a masked-MSE step with AdamW and gradient clipping. The learning rate follows a per-epoch cosine decay. Validation runs with direction dropout disabled but with random timesteps and noise.

### Run outputs

```text
runs/default/
├── config.yaml          # Fully resolved configuration, including the resolved model revision
├── split.json           # Train and validation variant records, with sequences and fitness values
├── metrics.jsonl        # One JSON object per epoch
└── epoch_0010/
    ├── inference.pt     # Config + denoiser/schedule weights, for protdiff sample
    └── ...              # Accelerate optimizer, scheduler, and RNG state
```

`metrics.jsonl` records `epoch`, `train_mse`, `val_mse`, and `learning_rate`.

**Read `val_mse` carefully.** It is a stochastic latent reconstruction error: mean squared error between the predicted and true clean target latent, at randomly drawn noise levels, over valid residues. It is useful for detecting divergence or overfitting. It is **not** a measure of functional improvement, and it will fluctuate between epochs from noise sampling alone.

`inference.pt` is a small payload — configuration, denoiser and schedule state, the epoch, and its metrics. The frozen encoder is *not* stored in it; it is reloaded from `model.name`, so that checkpoint must remain reachable.

Accelerate's optimizer, scheduler, and RNG state are saved alongside, which makes resuming possible in principle. **A resume CLI is not implemented** — that state is written for future use.

## Sampling

```bash
protdiff sample --checkpoint runs/default/epoch_0010/inference.pt \
  --sequences ACDEFGHIKLMNPQRSTVWY --directions 1 -1 \
  --num-samples 4 --output runs/ddim.jsonl
```

SDEdit, initializing from a partially noised seed for more conservative edits:

```bash
protdiff sample --checkpoint runs/default/epoch_0010/inference.pt \
  --sequences ACDEFGHIKLMNPQRSTVWY --directions 1 -1 \
  --t-start 200 --steps 50 --num-samples 4 --output runs/sdedit.jsonl
```

### Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--checkpoint` | required | Path to `inference.pt` |
| `--sequences` | required | One or more canonical seed sequences |
| `--directions` | required | **Signed** directions, one per objective, in configured order |
| `--num-samples` | `4` | Candidates generated per seed |
| `--steps`, `--guidance-scale`, `--eta`, `--t-start` | from checkpoint | Override the checkpoint's sampling configuration |
| `--seed` | `42` | Seeds the sampling noise |
| `--device` | `cuda` if available | Execution device |
| `--output` | `runs/samples.jsonl` | Output path |

### Signed directions

The CLI takes **signs, not token IDs**, because signs are what an experimentalist means:

| CLI value | Internal ID | Request |
| --- | --- | --- |
| `+1` | `2` | increase this objective |
| `0` | `1` | hold this objective unchanged |
| `-1` | `0` | decrease this objective |

So with objectives ordered `[lead_response, zinc_response]`, `--directions 1 -1` requests higher lead response and lower zinc response, and becomes `[2, 0]` internally. Exactly one value per objective is required. Token `3` (`<null>`) cannot be requested — it exists only for guidance.

### Output records

The output is JSONL, one record per candidate:

| Field | Contents |
| --- | --- |
| `seed_sequence` | The seed this candidate came from |
| `sequence` | The generated candidate |
| `objective_names`, `direction_ids` | The request, with objective order preserved |
| `hamming_distance` | Substitutions from the seed |
| `seed_recovery` | Fraction of positions identical to the seed |
| `sampling` | Resolved steps, guidance scale, eta, and `t_start` |
| `random_seed`, `checkpoint` | Provenance for reproducing the set |

`hamming_distance` and `seed_recovery` are the practical handles on exploration strength. If candidates come back identical to their seeds, raise `t_start` or the guidance scale; if they come back implausibly distant, lower them. Neither the sampler nor `t_start` enforces an edit budget — filter decoded candidates when you need a hard limit.

For ensemble diversity across a generated set, `metrics.shannon_entropy` gives mean per-position entropy in bits over aligned sequences. It measures spread across the set, not within a single sequence.

## Programmatic use

Training and generation are plain functions, so the active-learning cycle can be driven from a script or notebook without shelling out to the CLI:

```python
from directional_protein_diffusion.checkpoints import load_inference_checkpoint
from directional_protein_diffusion.config import load_config
from directional_protein_diffusion.data.labels import signed_directions_to_ids
from directional_protein_diffusion.generation import generate
from directional_protein_diffusion.training import run_training

last = run_training(load_config("configs/default.yaml"))

checkpoint = load_inference_checkpoint(
    last / "inference.pt",
    sampling_overrides={"t_start": 200, "guidance_scale": 2.0},
)
records = generate(
    checkpoint,
    ["ACDEFGHIKLMNPQRSTVWY"],
    signed_directions_to_ids([1, -1], checkpoint.config.data.num_objectives),
    num_samples=8,
    device="cpu",
)
```

`generate` returns the same dictionaries the CLI writes as JSONL, so candidate filtering, deduplication, and ranking can consume them directly. `load_inference_checkpoint` validates overrides against the configuration schema, so an impossible sampling request fails before a model is built.

## Tests and linting

```bash
python -m pytest -q
ruff check src tests
```

Tests construct a tiny random two-layer ESM-2 locally and never touch the network or a GPU. What they do and do not establish is spelled out in [status.md](status.md#what-the-tests-cover).
