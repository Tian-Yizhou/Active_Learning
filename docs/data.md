# Data preparation

How to bring experimental measurements into the pipeline, how directional labels are derived, and how pairs are constructed without leaking information across splits. The label definition itself is specified in [method.md](method.md#2-paired-supervision-and-the-label-contract).

## CSV schema

Set `data.csv_path` and provide a CSV whose columns are `id`, `sequence`, and one column per objective, named exactly as in `data.objective_names` and in that order:

```csv
id,sequence,lead_response,zinc_response
var_001,MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ,1.42,0.87
var_002,MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVH,2.10,0.91
var_003,MKTAYIAKQRQIAFVKSHFSRQLEERLGLIEVQ,0.95,0.84
```

Extra columns are ignored, so annotations may be carried alongside. The matching configuration is:

```yaml
data:
  csv_path: data/measurements.csv
  objective_names: [lead_response, zinc_response]
  num_objectives: 2
  tau: [0.05, 0.05]
```

Configuration validation rejects a mismatch between `num_objectives`, the length of `objective_names`, and the length of `tau`, as well as duplicate objective names.

Note that `data/` is listed in `.gitignore`, so experimental measurements placed there are not committed by accident.

## Sequence requirements

| Requirement | Why | Failure mode |
| --- | --- | --- |
| Uppercase canonical amino acids only (`ACDEFGHIKLMNPQRSTVWY`) | The encoder and the decoding vocabulary are restricted to the 20 canonical residues | Rejected at load and at encoding |
| All sequences the same length, position-aligned | Seed and target are concatenated per residue position; the current contract has no indel handling | Rejected at load |
| No duplicate sequence strings | Two IDs for the same protein would place that protein on both sides of the split | Rejected at load, with an explicit message |
| Replicates aggregated upstream | The pipeline has no replicate model; it treats each row as one measurement | Silent loss of information if not done |
| Finite objective values | Labels are derived from differences | Rejected at load |

Aggregating replicates is a scientific decision — mean, median, or a fitted estimate — and belongs in your upstream analysis, not here. The threshold $\tau$ is where replicate noise re-enters the pipeline.

## Choosing noise thresholds

`data.tau` holds one threshold per objective, in the configured order. These thresholds define the entire label set, so they deserve real attention:

- $\tau_k$ should reflect the **experimental noise of objective $k$** — for example a pooled standard deviation across replicates, or a confidence-interval half-width at the level of certainty you want behind each directional claim.
- Thresholds are in the **same units as the objective column**. There is no normalization step, so raw values and thresholds must be consistent across all rows.
- Larger $\tau_k$ yields more `<nc>` labels and fewer, more trustworthy directional claims. Smaller $\tau_k$ yields more directional pairs, some of which encode noise as signal.
- Different objectives routinely warrant very different thresholds; that is why `tau` is a vector.

The defaults `tau: [0.05, 0.05]` exist only to make the synthetic smoke test run. **They are placeholders and are meaningless for real data.**

## Objective ordering

The order of `objective_names` is a permanent part of the model's interface. It fixes:

- which CSV column becomes objective $k$,
- which embedding table sees which label,
- what `--directions` means on the `protdiff sample` command line.

Changing the order after training invalidates a checkpoint silently — the tensors still have the right shape. The order is saved into each run's `config.yaml` and into `inference.pt`; keep it with your results.

## Retaining `<nc>`

Pairs whose objectives all fall within noise are **kept** and labeled `<nc>`, not discarded. This is a deliberate departure from prior directional-label work, which typically filters comparisons that fall within experimental noise.

The rationale: "this edit changed nothing measurable" is a real and useful supervision signal, and it is also a direction a user may legitimately request — hold one property constant while moving another. A model trained only on significant changes has never been shown what *not* changing looks like.

## Splitting before pairing

Splits are computed over **unique sequences**, and pairs are then built within each partition independently:

```text
M unique variants
      │
      ├── shuffle with the run seed
      │
      ├──► train partition  ──► all within-partition pairs (+ reversals)
      └──► val partition    ──► all within-partition pairs (+ reversals)
```

If pairs were built first and then split, a held-out variant would appear inside training pairs and validation would report on sequences the model had already seen. Splitting first makes the partitions sequence-disjoint, which a test asserts directly.

Each partition is guaranteed at least two variants, since a partition with fewer cannot form a pair. The split is deterministic given `seed` and `val_fraction`, and is written to `split.json` in the run directory with full variant records.

**This is a sequence-disjoint split, not a homology- or lineage-disjoint one.** Two variants differing by a single substitution can land on opposite sides. For extrapolation claims — generalizing to a genuinely novel region of sequence space, or to a later experimental round — supply a cluster-based or round-based split instead. That is an extension point, not current behavior.

## Pair construction

Within a partition of $m$ variants:

- $\binom{m}{2}$ unordered pairs are enumerated.
- With `bidirectional: true` (the default), each pair is added in both orientations, giving $m(m-1)$ ordered examples with directions swapped accordingly.
- Labels are computed from the two fitness vectors and the thresholds at access time, so changing `tau` changes labels without touching the data file.
- Pair *indices* are stored, not latents. Encoding happens on demand in the training loop, never in DataLoader workers.

Memory grows as $O(m^2)$ in pair indices, which is appropriate for the low-$N$ regime this targets — a few hundred variants at most. Pair subsampling is an extension point for larger libraries.

Reversal is an orientation of the same experimental comparison. It doubles training examples but adds no independent information, and it should not be described as doubling the dataset in any scientific sense.

## Worked labeling example

Objectives ordered `[lead_response, zinc_response]` with `tau: [0.05, 0.05]`, using the CSV above:

| Source → Target | $\Delta$ lead | $\Delta$ zinc | Label | Reading |
| --- | --- | --- | --- | --- |
| `var_001` → `var_002` | +0.68 | +0.04 | `[2, 1]` | lead increased; zinc change within noise |
| `var_002` → `var_001` | −0.68 | −0.04 | `[0, 1]` | the same comparison, reversed |
| `var_001` → `var_003` | −0.47 | −0.03 | `[0, 1]` | lead decreased; zinc unchanged |
| `var_003` → `var_002` | +1.15 | +0.07 | `[2, 2]` | both increased beyond noise |

Note `var_001` → `var_002`: a raw difference of +0.04 in zinc response is *not* evidence of an increase at a threshold of 0.05, and becomes `<nc>`. Equality with the threshold would also be `<nc>`.

## Synthetic data

With `data.csv_path: null`, training generates a mock two-objective library: unique mutants of one random parent, scored by hydrophobic-residue fraction and charged-residue fraction with small Gaussian noise added.

```bash
protdiff synthetic --output data/synthetic.csv --num-variants 40 --length 32
```

This exists to exercise the pipeline end to end. The two objectives are not biology, they are deliberately competing counts of residue classes, and the built-in generator is hard-wired to exactly two objectives — more objectives require a real CSV. Never report results from synthetic data as a model capability.
