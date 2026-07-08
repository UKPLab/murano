---
title: Activation Patching
description: Apply interventions during generation and compare outputs.
---

Activation patching modifies the model's hidden states during the forward pass. Murano supports two strategies: **activation-level** intervention (hooking the residual stream) and **weight-level** ablation (modifying the weight matrices directly).

## Activation-level intervention

The `Intervene` step generates both clean and modified outputs for the same prompts, using an intervention function that operates on the residual stream.

### Ablating a direction

```python
from murano import MuranoDataset, MuranoModel, Pipeline
from murano.steps import Load, Record, SteeringVector, Intervene
from murano.steps.intervene import ablate_direction

model = MuranoModel("meta-llama/Llama-3.2-1B-Instruct")

# Step 1: find the direction
train_ds = MuranoDataset.contrastive(
    positive=["What a wonderful, delightful day", "This is fantastic and uplifting"],
    negative=["What a miserable, dreadful day", "This is awful and depressing"],
    template_fn=model.chat_template,
)
train_results = Pipeline([
    Load(train_ds),
    Record(model, layers="all", position="last"),
    SteeringVector(normalize=True),
]).run()

# Step 2: intervene and compare
eval_ds = MuranoDataset.contrastive(
    positive=["What a wonderful, delightful day"],
    negative=[],
    template_fn=model.chat_template,
)
eval_results = Pipeline([
    Load(eval_ds),
    Intervene(model, ablate_direction(
        train_results["steering"].direction_per_layer,
    )),
]).run()

comparison = eval_results["intervene"]
print("Clean:", comparison.clean_generations[0])
print("Ablated:", comparison.modified_generations[0])
```

### Steering a direction

Use `steer_direction` to add a scaled direction instead of removing it:

```python
from murano.steps.intervene import steer_direction

fn = steer_direction(
    train_results["steering"].direction_per_layer,
    alpha=2.0,   # positive = strengthen, negative = suppress
)
results = Pipeline([
    Load(eval_ds),
    Intervene(model, fn),
]).run()
```

### Custom intervention functions

`Intervene` accepts any callable with signature `(activation: Tensor, layer_idx: int) -> Tensor`:

```python
import torch

def zero_out(activation: torch.Tensor, layer: int) -> torch.Tensor:
    """Zero all activations — a sanity check."""
    return torch.zeros_like(activation)

results = Pipeline([
    Load(eval_ds),
    Intervene(model, zero_out, layers=[10, 11, 12]),
]).run()
```

## Weight-level ablation

The `WeightAblation` step applies an orthogonal projection P = I - dd^T to the model's weight matrices directly. This removes the direction from **all** computations, not just the residual stream at hook points.

```python
from murano.steps import WeightAblation

results = Pipeline([
    Load(train_ds),
    Record(model, layers="all", position="last"),
    SteeringVector(normalize=True),
    Load(eval_ds),         # reload eval prompts
    WeightAblation(model),
]).run()

comparison = results["weight_ablation"]
print(f"Modified {comparison.n_modified} weight matrices")
print("Clean:", comparison.clean_generations[0])
print("Ablated:", comparison.modified_generations[0])
```

`WeightAblation` automatically saves and restores the original weights, so the model is unchanged after the step completes.

### Saving the ablated model

Pass `save_dir` to export the ablated model in HuggingFace format:

```python
WeightAblation(model, save_dir="ablated_model/")
```

This saves the weights and tokenizer so you can reload the ablated model with `transformers` or `MuranoModel`.

## Activation-level vs weight-level

| | Activation-level (`Intervene`) | Weight-level (`WeightAblation`) |
|---|---|---|
| What it modifies | Residual stream at hook points | All weight matrices |
| Scope | Only affects hooked layers | Global across all layers |
| Reversible | Yes (no weight changes) | Auto-restores after step |
| Speed | Faster (fewer computations) | Slower (modifies all weights) |
| When to use | Quick experiments, layer-specific | Strongest ablation, model export |
