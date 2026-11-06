# Method and mathematical specification

This document is the authoritative statement of what the model is and what the training and sampling procedures compute. Implementation details and file-level navigation live in [architecture.md](architecture.md); data preparation lives in [data.md](data.md).

Notation: $B$ batch size, $L$ residue length, $D$ ESM-2 hidden width, $j$ number of objectives, $T$ diffusion steps.

## 1. Problem setting

A measured library contains $M$ protein variants, each with values for $j$ experimental objectives. We want a generator that, conditioned on a seed sequence and a requested direction per objective, proposes mutant sequences likely to realize those directions.

Formally, we learn a conditional distribution over target sequences

$$
p_\theta\left(X_2 \mid X_1, d\right), \qquad d \in \{0,1,2\}^j,
$$

where $X_1$ is the seed, $X_2$ the proposed mutant, and $d$ an ordered vector of discrete directional intents. The model operates not on tokens but on the continuous residue embeddings of a frozen protein language model.

### Why directional labels

This targets the low-data regime of experimental directed evolution, where a round of measurements yields perhaps a few dozen variants — usually too few to fit a trustworthy numerical fitness predictor, which is what optimizing a surrogate would require. Supervision instead comes from pairwise comparisons between measured variants, labeled only by the *sign* of each objective's change. Three properties make this practical:

- **Data amplification.** $M$ measured variants yield up to $M(M-1)$ ordered training pairs. These are not additional independent measurements, but they are many more training examples.
- **Multi-objective by construction.** The condition is an ordered vector, one token per objective, so a selectivity request such as "raise lead response while lowering zinc response" is expressible directly as $d=[2,0]$.
- **No fitness regression.** The model learns what kind of edit moves a property in a direction, rather than a numerical landscape.

No numerical fitness predictor is required anywhere in this pipeline.

### Position in the experimental cycle

The generator is one stage of an iterative loop, not a standalone optimizer:

```text
   measured variants ──► labeled pairs ──► train denoiser ──► conditional generation
          ▲                                                           │
          └────────── experimental measurement ◄── candidate selection ◄┘
```

1. Measure experimental variants on all objectives and estimate each objective's noise threshold.
2. Construct source–target pairs with directional labels, then train the conditional denoiser.
3. Choose seed sequences and requested directions; generate mutants with DDIM or SDEdit and optional guidance.
4. Apply candidate constraints, deduplicate, measure the selected sequences, and append the results for the next round.

## 2. Paired supervision and the label contract

Let $f_k(X)$ be the measured value of objective $k$, and $\tau_k \geq 0$ its experimental noise threshold. For an ordered pair $(X_1, X_2)$, define $\Delta f_k = f_k(X_2) - f_k(X_1)$ and

$$
d_k=\begin{cases}
0\;(\texttt{<dec>}) & \Delta f_k<-\tau_k,\\
1\;(\texttt{<nc>}) & |\Delta f_k|\leq\tau_k,\\
2\;(\texttt{<inc>}) & \Delta f_k>\tau_k.
\end{cases}
$$

The four token IDs are fixed across the project:

| ID | Token | Meaning |
| --- | --- | --- |
| 0 | `<dec>` | Measured value decreased beyond the noise threshold |
| 1 | `<nc>` | No change distinguishable from noise |
| 2 | `<inc>` | Measured value increased beyond the noise threshold |
| 3 | `<null>` | Directional condition dropped, for classifier-free guidance only — never an experimental label |

Rules that hold throughout the project:

- **Threshold equality belongs to `<nc>`.** A directional claim must strictly exceed experimental noise.
- **Token 3 = `<null>` is reserved** for classifier-free-guidance dropout. It is never produced by a measurement. Training code rejects labels outside $[0,2]$.
- **Reversing a pair swaps 0 and 2 and preserves 1.** Reversal follows from $\Delta f_k \mapsto -\Delta f_k$.
- **Objective order is fixed** across preprocessing, training, checkpoints, and inference. Order and thresholds are saved with every run.
- **Directions describe measured values, not desirability.** With objectives ordered `[lead_response, zinc_response]`, wanting higher lead response and lower zinc response is $d=[2,0]$.

From $M$ variants there are $\binom{M}{2}$ unordered pairs; orienting each as source and target and optionally adding the reverse gives up to $M(M-1)$ ordered examples. This amplifies training data, not independent measurement.

## 3. Frozen ESM-2 latents

For aligned sequences of equal residue length $L$, a frozen encoder $E$ gives

$$
z^{(1)}=E(X_1),\qquad z_0^{(2)}=E(X_2),\qquad z^{(1)},z_0^{(2)}\in\mathbb{R}^{L\times D}.
$$

Batched tensors are $B\times L\times D$. The representation used is the **final normalized ESM-2 hidden state**, the representation its language-model head was trained to consume, so predicted latents can be decoded without a learned decoder.

- $L$ counts residues only: BOS/EOS are seen by the encoder but stripped from every diffused tensor, and padding is excluded from attention and from the loss through an explicit boolean mask.
- The encoder stays in evaluation mode with all parameters frozen.
- **No latent rescaling or renormalization is applied.** If any is introduced later it must be recorded and inverted before decoding.
- The contract assumes aligned, equal-length source and target. Insertions and deletions require an explicit design extension, not just a code change.

## 4. Forward process: target only

With $\alpha_t = 1-\beta_t$, $\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$, and $\bar\alpha_0=1$:

$$
z_t^{(2)}=\sqrt{\bar\alpha_t}\,z_0^{(2)}+\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad \epsilon\sim\mathcal N(0,\mathbf I).
$$

Only the **target** is noised. The seed latent $z^{(1)}$ remains clean at every step and at every noise level, acting purely as a condition. Time indexing is explicit: $t=0$ is the clean state and $t=1,\dots,T$ are the noisy states, so $q(z_0 \mid z_0)$ is the identity.

Both a cosine and a linear $\beta$ schedule are available; cosine is the default.

## 5. Conditional denoiser

The network predicts the clean target directly:

$$
\hat z_0^{(2)}=\hat z_\theta\left(z_t^{(2)},t,z^{(1)},d\right).
$$

One training step, with the frozen encoder $E$ and the trainable denoiser $\hat z_\theta$:

```text
  seed X1 ──► E ──► z1 ───────────────────────────────┐  (clean condition)
                                                      │
target X2 ──► E ──► z2_0 ──► add noise at step t ──► z2_t
                                                      │
                                   concat on features ▼
                          ┌───────────────────────────────────────────┐
  t, directions d ──► c ─►│ copied ESM-2 blocks, AdaLN-conditioned on c│──► predicted z2_0
                          └───────────────────────────────────────────┘
                                                                        │
                                                 frozen lm_head ◄───────┘ ──► candidate X2
```

### 5.1 Sequence conditioning

Concatenate the noisy target and the clean seed along the **feature axis**, then project back to width $D$:

$$
h_0=P\left(\left[z_t^{(2)};z^{(1)}\right]_{\text{features}}\right),
\qquad \mathbb{R}^{L\times 2D}\xrightarrow{\;P\;}\mathbb{R}^{L\times D}.
$$

Concatenating on features rather than sequence length keeps residue positions aligned between seed and target, which is what makes the equal-length assumption useful: position $i$ of the target is conditioned on position $i$ of the seed.

### 5.2 Time and direction conditioning

A sinusoidal timestep embedding passed through an MLP gives $e_t\in\mathbb{R}^D$. The ordered direction vector is embedded so that **objective identity is preserved** — separate per-objective embedding tables, summed — giving $e_d\in\mathbb{R}^D$. The two combine additively:

$$
c=e_t+e_d\in\mathbb{R}^{D}.
$$

Preserving objective identity matters: $[2,0]$ and $[0,2]$ must not collide, which a single shared three-token table summed over objectives would permit.

### 5.3 Adaptive layer norm injection

The condition enters pretrained ESM-2 attention blocks by modulating their pre-norms:

$$
\mathrm{AdaLN}(h,c)=\gamma(c)\odot\mathrm{LayerNorm}(h)+\beta(c),
$$

with $\gamma$ and $\beta$ broadcast across residue positions. Here $\beta(c)$ is the conditioning shift — distinct from the diffusion schedule $\beta_t$.

- Initialization is $\gamma=1$, $\beta=0$, so a freshly constructed block reproduces the pretrained block exactly and training departs from pretrained behavior gradually.
- Both the attention pre-norm and the feed-forward pre-norm of each block are modulated, and each wraps the *pretrained* `LayerNorm` module so its learned parameters are retained.

### 5.4 Parameter ownership

Attention blocks used for denoising are **independent copies** of the pretrained ESM-2 blocks. The frozen encoder and decoding head are never shared with trainable modules, so optimization cannot drift the representation space that decoding depends on. Copied blocks, projections, and all conditioning parameters are trainable.

## 6. Training objective

Use **$x_0$-prediction with mean squared error**:

$$
\mathcal L=\mathbb E_{(X_1,X_2,d),\,t,\,\epsilon}
\left[\left\|\hat z_\theta\left(z_t^{(2)},t,z^{(1)},d\right)-z_0^{(2)}\right\|_2^2\right].
$$

In a padded batch the squared error is reduced over valid residue positions and hidden features only, then averaged per sample, so sequences of different lengths contribute comparably and padding cannot influence the gradient.

This is a deliberate choice, not an inherited default: predicting the clean latent keeps the training target in the same space the frozen `lm_head` decodes, and makes partial-noise sampling well behaved. Reference diffusion code that defaults to noise prediction or an L1 objective must be reconfigured explicitly.

## 7. Classifier-free guidance

**Training.** With configurable probability $p_{\text{uncond}}$, the *entire* direction vector of a sample is replaced by $d_\varnothing=(3,\dots,3)$. The seed latent is retained in both cases, so the null branch is unconditional with respect to directions while remaining seed-conditional. Dropout applies in training mode only.

**Sampling.** The matching combination is

$$
\hat z_{0,\mathrm{CFG}}=\hat z_{0,\varnothing}+w\left(\hat z_{0,d}-\hat z_{0,\varnothing}\right).
$$

- $w=1$ is the ordinary conditional prediction — a single network evaluation.
- $w=0$ is the null-direction prediction.
- $w>1$ amplifies the directional component at the cost of diversity, and requires two evaluations per step.

Any change to *which* conditions are dropped must be made consistently in both training and sampling, and recorded with the checkpoint. Guidance scale and dropout convention belong in the provenance of every generated set.

## 8. Sampling

Both initialization modes share one reverse sampler consistent with $x_0$-prediction. At each step the predicted clean latent is converted to an implied noise term, then re-noised to the next lower timestep in DDIM fashion; $\eta=0$ gives a deterministic trajectory for fixed initial noise, and $\eta>0$ injects the corresponding stochastic component.

**Full DDIM.** Initialize $z_T\sim\mathcal N(0,\mathbf I)$ and reverse to $t=0$, conditioned throughout on the clean $z^{(1)}$ and on $d$. This explores broadly and does not anchor on the seed except through the condition.

**SDEdit / partial noising.** Choose $0<t_0<T$ and initialize the evolving state from a noised copy of the seed,

$$
z_{t_0}^{(1)}=\sqrt{\bar\alpha_{t_0}}\,z^{(1)}+\sqrt{1-\bar\alpha_{t_0}}\,\epsilon,
$$

then reverse from $t_0$. A separate, clean $z^{(1)}$ is retained as the condition — the noised copy is the state being denoised, not the condition. Lower $t_0$ keeps candidates closer to the seed.

Partial noising is a **soft** control on exploration. It does not bound the number of mutations. If a hard edit-distance limit is required, enforce it by filtering decoded candidates.

## 9. Decoding

The predicted clean latent is decoded with the frozen head:

$$
\ell=\mathrm{lm\_head}\left(\hat z_0^{(2)}\right)\in\mathbb{R}^{L\times|\mathcal V|},
\qquad \hat X_2=\mathrm{Decode}(\ell).
$$

The decoding policy is **argmax restricted to the 20 canonical amino acids**: special tokens are excluded from the logit selection rather than filtered after the fact, so no special token can ever appear as a residue. Temperature or top-$k$ sampling over residues is a documented extension point, not current behavior.

## 10. Reproducibility requirements

Every run must record: the objective order, the thresholds $\tau$, the train/validation split, the frozen model identifier and resolved revision, the diffusion schedule and step count, the CFG dropout probability, the sampling settings including guidance scale and any $t_0$, and all random seeds. Latent preprocessing, if ever introduced, is part of this list.

For claims about generalization to unseen variants, split variants **before** building pairs — otherwise a held-out sequence reappears inside training pairs and the evaluation is meaningless. The implemented split is sequence-disjoint; homology- or round-disjoint splits are stronger and must be supplied deliberately. See [data.md](data.md#splitting-before-pairing).

## 11. What the model does not establish

A generated candidate carries a *requested* direction, not a demonstrated one. Validation loss is a latent reconstruction error under random noise levels; it is not a measure of functional improvement. Only experimental measurement closes that gap, which is why the method is framed as one stage inside an active-learning cycle rather than as a standalone optimizer.
