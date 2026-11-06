# Documentation

Project documentation for [Directional Protein Diffusion](../README.md). Start with the top-level [README](../README.md) for an introduction and quickstart; these documents carry the detail.

| Document | Read it when you want to |
| --- | --- |
| [method.md](method.md) | Understand the mathematics: label contract, forward diffusion, conditional denoiser, training objective, classifier-free guidance, sampling, decoding |
| [data.md](data.md) | Prepare experimental data, choose noise thresholds, understand splitting and pair construction |
| [usage.md](usage.md) | Set up the environment, configure a run, train, read checkpoints and metrics, generate candidates |
| [architecture.md](architecture.md) | Navigate the code, learn the tensor conventions, or extend the implementation |
| [status.md](status.md) | Know exactly what is implemented, what the tests prove, and what remains unvalidated |

Keep these pages consistent with the code whenever the design changes: a change to the scientific contract belongs in [method.md](method.md), and a change in what is implemented or validated belongs in [status.md](status.md) and the top-level README's status note.
