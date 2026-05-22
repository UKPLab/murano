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

The PyPI distribution is `murano-interp` (the bare name `murano` belongs to an
unrelated OpenStack project). The Python module name is unchanged: `import murano`.

For a development install from source:

```bash
pip install -e .
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
| `Record`           | `dataset`             | `record`                       | Capture residual-stream activations via nnsight.         |
| `SteeringVector`   | `record`              | `steering`                     | Find a contrastive steering direction (mean diff).       |
| `Intervene`        | `prompts`             | `intervene`                    | Generate baseline + intervened outputs side-by-side.     |
| `WeightAblation`   | `prompts`, `steering` | `intervene`, `weight_ablation` | Project a direction out of model weights, then generate. |
| `Probe`            | `record`              | `probe`                        | Train a linear probe per layer via cross-validation.     |
| `GenerationMetric` | `intervene`           | `metric`                       | Score baseline vs modified outputs with a user metric.   |
| `ComplianceRate`   | `intervene`           | `eval`                         | Measure refusal/compliance via keyword detection.        |
| `Save`             | (any present)         | `output_dir`                   | Persist all results to organized subdirectories.         |
| `Plot` \*          | (optional)            | —                              | Render refusal plots (steering, generations, eval).      |
| `ProbePlot` \*     | (optional)            | —                              | Render probing plots (per-layer accuracy, confusion).    |

\* Requires the `[plot]` extra: `pip install -e .[plot]`.

To add your own step, subclass `Step`, set `reads` / `writes` (and optionally
`read_types` / `write_types`), and implement `__call__(results) -> Results`.

### Status

The Step API and the steps in the table above are alpha-stable for the 0.1.x
line. The `murano.lenses` module (logit lens) ships in this release but is not
yet wired into the Pipeline API and should be considered experimental until the
work tracked in [#51](https://github.com/UKPLab/murano/issues/51) lands.

## Core Ideas

- `MuranoModel` is a thin model wrapper around `nnsight`.
- `Pipeline`, `Step`, and `Results` are the orchestration core.
- artifacts such as `PromptBatch`, `ActivationStore`, `SteeringResult`,
  `GenerationComparison`, and `MetricResult` make experiment dataflow explicit.
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

## Development

```bash
uv sync --all-extras --dev
python -m pytest -q
```

## Disclaimer

> This repository contains experimental software and is intended as a research
> framework for mechanistic interpretability workflows.
