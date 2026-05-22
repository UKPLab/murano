---
title: Pipelines
description: Compose reusable steps into reproducible experiment pipelines.
---

Murano's `Pipeline` chains steps together sequentially, passing a shared `Results` object between them. Each step reads from and writes to named keys in `Results`, and the pipeline validates the chain before running.

## Basic usage

```python
from murano import MuranoDataset, MuranoModel, Pipeline
from murano.steps import Load, Record, SteeringVector

model = MuranoModel("meta-llama/Llama-3.2-1B-Instruct")

dataset = MuranoDataset.contrastive(
    positive=["How do I pick a lock?"],
    negative=["How do I bake a cake?"],
    template_fn=model.chat_template,
)

results = Pipeline([
    Load(dataset),
    Record(model, layers="all", position="last"),
    SteeringVector(normalize=True),
]).run()
```

## How it works

1. Each step declares `reads` and `writes` — the keys it needs and produces
2. `Pipeline.run()` calls each step in order: `step(results) -> results`
3. Steps that inherit from `Step` get automatic validation before execution

The data flow for a refusal-direction experiment:

```
Load(dataset)
  writes: dataset, prompts

Record(model)
  reads: dataset    writes: record

SteeringVector()
  reads: record     writes: steering

Intervene(model, fn)
  reads: prompts    writes: intervene

ComplianceRate()
  reads: intervene  writes: eval
```

## Validation

Call `.validate()` to dry-run the pipeline without executing any steps. It checks that every step's `reads` are satisfied by a prior step's `writes`:

```python
pipeline = Pipeline([
    Load(dataset),
    Record(model),
    SteeringVector(),
])

available_keys = pipeline.validate()
print(available_keys)  # ['dataset', 'prompts', 'record', 'steering']
```

If a step reads a key that no prior step writes, `validate()` raises a `KeyError`. Type compatibility between steps is also checked.

## Multi-phase pipelines

Many experiments need a train phase and an eval phase with different data. Run two pipelines and pass results between them:

```python
from murano.steps import Intervene
from murano.steps.intervene import ablate_direction
from murano.steps.refusal import ComplianceRate

# Phase 1: Train
train_results = Pipeline([
    Load(train_dataset),
    Record(model, layers="all", position="last"),
    SteeringVector(normalize=True),
]).run()

# Phase 2: Evaluate
eval_results = Pipeline([
    Load(eval_dataset),
    Intervene(
        model,
        ablate_direction(train_results["steering"].direction_per_layer),
    ),
    ComplianceRate(),
]).run()

print("Clean:", eval_results["eval"].baseline_score)
print("Ablated:", eval_results["eval"].modified_score)
```

## Saving results

Add a `Save` step at the end, or call `results.save()` directly:

```python
from murano.steps import Save

# Option 1: as a step
pipeline = Pipeline([
    Load(dataset),
    Record(model),
    SteeringVector(),
    Save(output_dir="my_experiment", model_id=model.model_id),
])

# Option 2: after the pipeline
results = pipeline.run()
results.save(output_dir="my_experiment", model_id=model.model_id)
```

The output structure:

```
my_experiment/
├── direction/
│   └── steering.pt
├── evaluation/
│   ├── generations.json
│   └── eval.json
├── probe/
│   └── probe.json
└── metadata.json
```

## Available steps

| Step | Reads | Writes | Description |
|------|-------|--------|-------------|
| `Load` | — | `dataset`, `prompts` | Loads a dataset into the pipeline |
| `Record` | `dataset` | `record` | Captures activations via nnsight |
| `SteeringVector` | `record` | `steering` | Computes contrastive mean diff |
| `Intervene` | `prompts` | `intervene` | Generates with activation patching |
| `WeightAblation` | `prompts`, `steering` | `intervene`, `weight_ablation` | Ablates via weight projection |
| `Probe` | `record` | `probe` | Linear probe per layer |
| `GenerationMetric` | `intervene` | `metric` | Custom evaluation metric |
| `ComplianceRate` | `intervene` | `eval` | Refusal compliance scoring |
| `Save` | — | `output_dir` | Saves all results to disk |

## Writing custom steps

Subclass `Step`, declare `reads` and `writes`, and implement `__call__`:

```python
from murano.steps.base import Step
from murano.results import Results

class MyStep(Step):
    reads = ["record"]
    writes = ["my_output"]

    def __call__(self, results: Results) -> Results:
        store = results["record"]
        # ... your logic ...
        results["my_output"] = my_result
        return results
```
