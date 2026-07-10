# Murano notebooks

Hands-on Jupyter notebooks that walk through Murano workflows end-to-end. Every
one of them is executed before it is committed, so the outputs you see are the
ones the code produced. They also render on the
[documentation site](https://ukplab.github.io/murano/docs/notebooks/getting_started/).

New to Murano? Start with [`getting_started.ipynb`](getting_started.ipynb): it
covers the quick API, the `Pipeline` / `Step` / `Results` core, saving, and
plotting in one sitting. Everything else assumes it.

Each notebook states the extras it needs at the top, and shares the same shape: an
overview, the questions it answers, a numbered outline, and a "What next". To
install everything:

```bash
pip install -e ".[all]"
```

The datasets are not copied between notebooks. They come from
[`murano.tasks`](../src/murano/tasks.py): `ioi()` builds the
indirect-object-identification task, `sentiment()` returns contrastive sentences.
Neither is a component sweep hand-rolled: a study that runs the same chain once per
component uses the [`Sweep`](../src/murano/steps/sweep.py) step, and every pipeline
is built at cell level so a reader can see the steps.

## Getting started

- [`getting_started.ipynb`](getting_started.ipynb) — record activations, find a
  steering direction, generate with it, then rebuild the same experiment as an
  explicit pipeline and write a custom step. Extras: `[plot]`.

## Applications

One self-contained notebook per application. Most run on GPT-2 small and take about
a minute each, dominated by loading the model; the exceptions are called out below.

**Concepts and directions**

- [`applications/steering.ipynb`](applications/steering.ipynb) — derive a
  sentiment direction and steer generation with it, including the two ways to get
  it wrong. Extras: `[plot]`.
- [`applications/probing.ipynb`](applications/probing.ipynb) — train a linear
  probe per layer, then project the same activations to see the structure the
  probe exploits. Extras: `[probe,plot]`.
- [`applications/logit_lens.ipynb`](applications/logit_lens.ipynb) — read each
  layer's running next-token prediction. Extras: `[plot]`.

**Finding the components responsible for a behavior**

- [`applications/logit_attribution.ipynb`](applications/logit_attribution.ipynb)
  — attribute a logit difference to individual heads and MLPs, then rank them.
  Extras: `[plot]`.
- [`applications/attention.ipynb`](applications/attention.ipynb) — summarize
  every head with one number, inspect a single head's pattern, ablate it, and read
  what it writes through its OV circuit. Extras: `[plot]`.
- [`applications/ablation.ipynb`](applications/ablation.ipynb) — zero, mean and
  resample ablation on the same head, and why they disagree. No extras.
- [`applications/activation_patching.ipynb`](applications/activation_patching.ipynb)
  — patch clean activations into a corrupted run and measure how much behavior
  each head restores. Extras: `[plot]`.
- [`applications/circuit_discovery.ipynb`](applications/circuit_discovery.ipynb)
  — compose attribution, sweeping, selection and path patching into one experiment
  that recovers the indirect-object-identification circuit. Extras: `[plot]`.

**Measuring and composing**

- [`applications/metrics.ipynb`](applications/metrics.ipynb) — every metric step
  on one intervention, and why they disagree. No extras.
- [`applications/custom_pipeline.ipynb`](applications/custom_pipeline.ipynb) —
  mix probing and steering to answer a question no single step answers, by writing
  the step that is missing. Extras: `[probe,plot]`.
- [`applications/weight_ablation.ipynb`](applications/weight_ablation.ipynb) —
  project a direction out of the weights themselves, not just the activations.
  Needs a Llama-family model (Llama-3.2-1B) rather than GPT-2, and a few GB of GPU
  memory. No extras.

**Sparse autoencoders**

- [`applications/sae_features.ipynb`](applications/sae_features.ipynb) — encode
  prompts through a pre-trained SAE, confirm the code is sparse, and read what a
  feature means through its decoder. Extras: `[sae]`.
- [`applications/sae_steering.ipynb`](applications/sae_steering.ipynb) — find the
  feature for a concept, then steer with it, and find that no strength both invokes
  the concept and preserves the text. Extras: `[sae]`.
- [`applications/sae_enrichment.ipynb`](applications/sae_enrichment.ipynb) — rank
  thousands of features by how well they separate two classes, and see why
  selectivity is not meaning. Extras: `[sae,data,plot]`.

`sae_features` and `sae_steering` run **Gemma 2 2B**, which is gated on the Hub:
accept its licence once, then `hf auth login`. Expect a few GB of GPU memory.

## Reproductions

Published results, reproduced with Murano. These use larger models than the
application notebooks.

- [`reproductions/wang2023_ioi.ipynb`](reproductions/wang2023_ioi.ipynb) — the
  indirect-object-identification circuit in GPT-2 small.
- [`reproductions/marks2023_geometry_of_truth.ipynb`](reproductions/marks2023_geometry_of_truth.ipynb)
  — the linear structure of truth in Llama-2-13B.
- [`reproductions/todd2024_function_vectors.ipynb`](reproductions/todd2024_function_vectors.ipynb)
  — function vectors in GPT-J-6B.
