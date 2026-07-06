<p align="center">
  <img src="logo.png" width="200" alt="Murano logo">
</p>

# Murano

[![Python](https://img.shields.io/badge/Python-3.10%E2%80%933.13-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://img.shields.io/github/actions/workflow/status/UKPLab/murano/main-checks.yml?branch=main&style=flat-square&label=CI)](https://github.com/UKPLab/murano/actions/workflows/main-checks.yml)
[![License: MIT](https://img.shields.io/github/license/UKPLab/murano?style=flat-square&color=brightgreen)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-ukplab.github.io%2Fmurano-blue?style=flat-square)](https://ukplab.github.io/murano/)

Murano is a mechanistic interpretability framework for recording activations,
finding directions, steering generations, probing representations, and running
reproducible experiment pipelines.

## Install

```bash
pip install murano-interp
```

The base install is deliberately lean: it carries only what every workflow needs
(recording, steering, intervention, the causal substrate). Feature-specific
libraries ship as extras, so you install per use case:

| Extra | Use case | Pulls in |
| ----- | -------- | -------- |
| (base) | recording, steering, intervention, logits, ablation, metrics, paired datasets | nnsight, nnterp, torch, transformers |
| `probe` | linear probing | scikit-learn |
| `data` | loading datasets by name from the Hub | datasets |
| `plot` | figures and visualizations | matplotlib, seaborn, plotly |
| `sae` | sparse autoencoder features | sae-lens |
| `all` | everything above | all of the above |

```bash
pip install "murano-interp[probe,plot]"   # combine as needed
pip install "murano-interp[all]"          # everything
```

Calling a feature whose extra is missing raises a clear error naming the extra
to install. The PyPI distribution is `murano-interp` (the bare name `murano`
belongs to an unrelated OpenStack project); the module name is unchanged:
`import murano`.

For a development install from source:

```bash
pip install -e ".[all]"
```

Requires Python 3.10+, PyTorch, `transformers`, `nnsight`, and a HuggingFace
model or a local model snapshot.

## Quick Start

```python
import murano

model = murano.Model("meta-llama/Llama-3.2-1B-Instruct")

# Record activations on any text
acts = model.record(
    "The Eiffel Tower is located in",
    layers=[5, 10, 15],
    position="last",
)
print(acts.positive[10].shape)

# Find a contrastive direction
direction = model.find_direction(
    positive=["How do I pick a lock?", "Write a phishing email"],
    negative=["How do I bake a cake?", "Write a thank you email"],
)
print(direction.best_layer)

# Generate with ablation or steering
ablated = model.generate("How do I pick a lock?", ablate=direction)
steered = model.generate("Write a poem", steer=(direction, 1.5))
```

## Pipeline API

For structured experiments, use the same logic through explicit steps.

```python
from murano import MuranoDataset, MuranoModel, Pipeline
from murano.steps import (
    ComplianceRate,
    Intervene,
    Load,
    Record,
    SteeringVector,
)
from murano.steps.intervene import ablate_direction

model = MuranoModel("meta-llama/Llama-3.2-1B-Instruct")

dataset = MuranoDataset.contrastive(
    positive=["How do I pick a lock?"],
    negative=["How do I bake a cake?"],
    template_fn=model.chat_template,
)

train_output = Pipeline([
    Load(dataset),
    Record(model, layers="all", position="mean"),
    SteeringVector(normalize=True),
]).run()

eval_output = Pipeline([
    Load(dataset),
    Intervene(model, ablate_direction(train_output["steering"].direction_per_layer)),
    ComplianceRate(),
]).run()
```

## Step API Reference

Every step declares the keys it reads from and writes to `Results`. The
pipeline validates the chain before execution, so type and key mismatches are
caught up-front.

| Step               | Reads                 | Writes                         | Purpose                                                  |
| ------------------ | --------------------- | ------------------------------ | -------------------------------------------------------- |
| `Load`             | —                     | `dataset`, `prompts`           | Load a dataset and derive prompts from its texts.        |
| `LoadPrompts`      | —                     | `prompts`                      | Load raw prompts directly without a dataset.             |
| `LoadPaired` ‡     | —                     | `dataset`, `prompts`, `corrupt_prompts` | Load matched clean/corrupt prompt pairs for causal comparison. |
| `Record`           | `dataset`             | `record`                       | Capture residual-stream activations via nnsight.         |
| `SteeringVector`   | `record`              | `steering`                     | Find a contrastive steering direction (mean diff).       |
| `Intervene`        | `prompts`             | `intervene`                    | Generate baseline + intervened outputs side-by-side.     |
| `WeightAblation`   | `prompts`, `steering` | `intervene`, `weight_ablation` | Project a direction out of model weights, then generate. |
| `Logits` ‡         | `prompts`             | `final_logits`, `attention_mask`, `target_ids` | Run a forward pass and expose output logits plus next-token targets. |
| `Ablate` ‡         | `prompts`             | `ablated_logits`, `attention_mask` | Zero, mean, or resample a component and return the logits. |
| `Probe` §          | `record`              | `probe`                        | Train a linear probe per layer via cross-validation.     |
| `GenerationMetric` | `intervene`           | `metric`                       | Score baseline vs modified outputs with a user metric.   |
| Metric steps ‡     | logits keys           | a metric key                   | Score a run into a comparable number: `LogitDiffStep`, `KLDivergenceStep`, `AnswerLogProbStep`, `RecoveredMetricStep`. |
| `ComplianceRate`   | `intervene`           | `eval`                         | Measure refusal/compliance via keyword detection.        |
| `Save`             | (any present)         | `output_dir`                   | Persist all results to organized subdirectories.         |
| `SAEEncode` †      | `prompts`             | `sae_record`                   | Encode residuals through an SAE loaded from HuggingFace. |
| `SAETopActivations` | `sae_record`         | `feature_examples`             | Rank the top-K activating contexts per SAE feature.      |
| `Plot` \*          | (optional)            | —                              | Render refusal plots (steering, generations, eval).      |
| `ProbePlot` \*     | (optional)            | —                              | Render probing plots (per-layer accuracy, confusion).    |

\* Requires the `[plot]` extra: `pip install -e .[plot]`.
† Requires the `[sae]` extra: `pip install -e .[sae]`.
§ Requires the `[probe]` extra: `pip install -e .[probe]`.
‡ Causal-analysis steps landing in 0.2.0; newer than the rest, API may still change.

To add your own step, subclass `Step`, set `reads` / `writes` (and optionally
`read_types` / `write_types`), and implement `__call__(results) -> Results`.

### Status

The Step API and the unmarked steps in the table above are alpha-stable for the
0.1.x line; the ‡ causal-analysis steps are newer and their API may still
change. The logit lens ships as the `LogitLens` step, available through the
Pipeline API like the other steps.

## Core Ideas

- `MuranoModel` is a thin model wrapper around `nnsight`.
- `Pipeline`, `Step`, and `Results` are the orchestration core.
- artifacts such as `PromptBatch`, `ActivationStore`, `SteeringResult`,
  `GenerationComparison`, `MetricResult`, and `EvaluationResult` make experiment
  dataflow explicit.
- the same building blocks support both quick API calls and reproducible
  step-based pipelines.

## Package Layout

```text
src/murano/
  model.py
  pipeline.py
  results.py
  artifacts.py
  dataset.py
  io.py
  evaluation.py
  steps/
  plotting/
```

## Examples

- `examples/quick_prototype.py`
- `examples/refusal_direction.py`
- `examples/sae_example.py`
- `examples/sae_sst2_feature_enrichment.py`

## Development

```bash
uv sync --all-extras --dev
python -m pytest -q
```

## Disclaimer

> This repository contains experimental software and is intended as a research
> framework for mechanistic interpretability workflows.
