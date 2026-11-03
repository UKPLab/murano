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

`Intervene` accepts any callable with signature `(activation: Tensor, node: Node) -> Tensor`. The second argument is a `murano.Node` naming the hooked site (`node.layer`, `node.module`, ...), not a bare layer index, so a function that targets specific sites branches on it:

```python
import torch

def zero_out(activation: torch.Tensor, node) -> torch.Tensor:
    """Zero all activations — a sanity check. `node` names the hooked site."""
    return torch.zeros_like(activation)

results = Pipeline([
    Load(eval_ds),
    Intervene(model, zero_out, layers=[10, 11, 12]),
]).run()
```

## Weight-level ablation

The `WeightAblation` step applies an orthogonal projection P = I - dd^T to the model's weight matrices directly (write matrices as `P @ W`, read matrices as `W @ P`), so the ablation is baked into the weights rather than applied at hook points during the forward pass. It is a stronger, more global edit than activation-level ablation; note that the read-matrix projection interacts with the intervening layer norm, so it does not remove the direction perfectly.

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

## Choosing what to patch: information flow routes

Before patching, you need candidate components. `InformationFlow` (Ferrando & Voita, 2024) finds the heads and MLPs a prediction goes through from two forward passes, with no corrupt prompts or gradients. It splits each residual vector into the parts that wrote it, scores each part by its L1 proximity to the whole, and walks down from the prediction keeping every edge above `threshold`. The model must be loaded with `enable_attention_probs=True`.

```python
from murano import MuranoModel, Pipeline, keys
from murano.steps import InformationFlow, LoadPaired
from murano.tasks import ioi

model = MuranoModel("gpt2", enable_attention_probs=True)
results = Pipeline([LoadPaired(ioi()), InformationFlow(model, threshold=0.03)]).run()

flow = results[keys.INFORMATION_FLOW]
```

`flow.contributions` maps each head and MLP to the share of prompts whose routes pass through it. That share depends on the threshold and the batch, and it says nothing about causal effect. On IOI many layer 0 heads sit on every route, so compare heads within the later layers. `flow.edges[i]` lists the edges of prompt `i`'s route. An edge's `source_pos` and `dest_pos` index `flow.tokens[i]`, and its `importance` is its share of the destination vector. To check whether the frequent heads matter, pass them to `Patch(targets=...)`, as the information flow notebook does.
