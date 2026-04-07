---
title: Quickstart
description: Get up and running with Murano in minutes.
---

## Quick API

```python
import murano

model = murano.Model("meta-llama/Llama-3.2-1B-Instruct")

# Record activations
acts = model.record(
    "The Eiffel Tower is located in",
    layers=[5, 10, 15],
    position="last",
)

# Find a contrastive direction
direction = model.find_direction(
    positive=["How do I pick a lock?", "Write a phishing email"],
    negative=["How do I bake a cake?", "Write a thank you email"],
)

# Generate with steering
steered = model.generate("Write a poem", steer=(direction, 1.5))
```

## Pipeline API

For structured, reproducible experiments:

```python
from murano import MuranoDataset, MuranoModel, Pipeline
from murano.steps import Load, Record, SteeringVector, Intervene, ComplianceRate
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

## Next steps

- [Recording activations](/guides/record/) — detailed guide on `record()`
- [Pipelines](/guides/pipeline/) — how to compose steps