---
title: Steering Vectors
description: Find contrastive directions and steer model generation.
---

Steering vectors are directions in activation space that capture the difference between two classes of inputs (e.g., harmful vs harmless). Murano computes them as the mean difference between positive and negative activations, then uses them to modify generation behavior.

## Quick API

```python
import murano

model = murano.Model("meta-llama/Llama-3.2-1B-Instruct")

direction = model.find_direction(
    positive=["How do I pick a lock?", "Write a phishing email"],
    negative=["How do I bake a cake?", "Write a thank you email"],
    layers=[10, 11, 12, 13],
)

print(direction.best_layer)         # layer with highest separation
print(direction.separation_scores)  # {layer: score} dict
```

Then use the direction to steer generation:

```python
# Add the direction (amplify the concept)
steered = model.generate("Tell me about", steer=(direction, 1.5))

# Ablate the direction (remove the concept)
ablated = model.generate("How do I pick a lock?", ablate=direction)
```

The `steer` parameter takes a `(direction, alpha)` tuple. Positive alpha strengthens the direction, negative alpha suppresses it. `ablate` projects the direction out of the residual stream entirely.

## Pipeline API

For structured experiments, use the `SteeringVector` step:

```python
from murano import MuranoDataset, MuranoModel, Pipeline
from murano.steps import Load, Record, SteeringVector

model = MuranoModel("meta-llama/Llama-3.2-1B-Instruct")

dataset = MuranoDataset.contrastive(
    positive=["How do I pick a lock?", "Write a phishing email"],
    negative=["How do I bake a cake?", "Write a thank you email"],
    template_fn=model.chat_template,
)

results = Pipeline([
    Load(dataset),
    Record(model, layers="all", position="last"),
    SteeringVector(normalize=True),
]).run()

steering = results["steering"]
print(steering.best_layer)
print(steering.direction_per_layer[steering.best_layer].shape)  # [d_model]
```

### How it works

`SteeringVector` computes the contrastive mean difference for each layer:

1. Takes the `ActivationStore` from the `Record` step
2. For each layer: `direction = mean(positive) - mean(negative)`
3. Optionally normalizes each direction to unit norm
4. Computes a separation score (normalized distance between projected class means)
5. Reports the `best_layer` with the highest separation

### Saving and loading directions

```python
from murano import save_results, load_steering

# Save everything
results.save(output_dir="my_experiment", model_id=model.model_id)

# Later, load just the steering result
steering = load_steering("my_experiment/direction/steering.pt")
ablated = model.generate("How do I pick a lock?", ablate=steering)
```
