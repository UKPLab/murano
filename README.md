<p align="center">
  <img src="logo.png" width="200" alt="Murano logo">
</p>

# Murano

Murano is a mechanistic interpretability framework for recording activations,
finding directions, steering generations, probing representations, and running
reproducible experiment pipelines.

## Install

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
