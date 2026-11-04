---
title: Recording Activations
description: Extract hidden states from any layer, module, or token position.
---

Murano captures residual-stream activations via [nnsight](https://nnsight.net/) tracing. You can use the quick API for one-off exploration or the pipeline `Record` step for reproducible experiments.

## Quick API

```python
import murano

model = murano.Model("meta-llama/Llama-3.2-1B-Instruct")

acts = model.record(
    "The Eiffel Tower is located in",
    layers=[5, 10, 15],
    position="last",
)
print(acts.positive[10].shape)  # [1, d_model]
```

`model.record()` returns an `ActivationStore` with a `.positive` dict mapping layer indices to tensors of shape `[N, d_model]`.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str \| list[str]` | required | One or more input texts |
| `layers` | `list[int] \| "all"` | `"all"` | Which decoder layers to record |
| `position` | `str \| int` | `"last"` | Token position to extract |
| `batch_size` | `int` | `8` | Batch size for processing |

### Token positions

- `"last"` — last non-padding token (most common for causal LMs)
- `"first"` — first non-padding token
- `"mean"` — mean-pool over all non-padding tokens
- integer — specific token index (negative indexing supported)

## Pipeline API

For contrastive recording (positive vs negative texts), use the `Record` step with a `MuranoDataset`:

```python
from murano import MuranoDataset, MuranoModel, Pipeline
from murano.steps import Load, Record

model = MuranoModel("meta-llama/Llama-3.2-1B-Instruct")

dataset = MuranoDataset.contrastive(
    positive=["What a wonderful, delightful day", "This is fantastic and uplifting"],
    negative=["What a miserable, dreadful day", "This is awful and depressing"],
    template_fn=model.chat_template,
)

results = Pipeline([
    Load(dataset),
    Record(model, layers="all", position="last"),
]).run()

store = results["record"]
print(store.positive[10].shape)   # [2, d_model]
print(store.negative[10].shape)   # [2, d_model]
```

### Labeled recording (for probing)

When using a `LabeledDataset`, `Record` produces a `LabeledActivationStore` instead:

```python
from murano import LabeledDataset, Pipeline
from murano.steps import Load, Record

dataset = LabeledDataset.from_hub(
    "stanfordnlp/sst2",
    text_column="sentence",
    label_column="label",
    n=500,
    label_names=["negative", "positive"],
)

results = Pipeline([
    Load(dataset),
    Record(model, layers="all", position="last"),
]).run()

labeled_store = results["record"]
print(labeled_store.activations[10].shape)  # [500, d_model]
print(labeled_store.labels.shape)           # [500]
```

## Loading data from HuggingFace Hub

`MuranoDataset.from_hub()` can load contrastive pairs directly and split them into train/eval:

```python
train_ds, eval_ds = MuranoDataset.from_hub(
    positive=("walledai/HarmBench", "standard", "prompt"),
    negative=("tatsu-lab/alpaca", "instruction"),
    n_train=150,
    n_eval=50,
    template_fn=model.chat_template,
)
```

The tuple format is `(dataset_name, config, column)` or `(dataset_name, column)`.

## Ranking neurons with RACE

`RACE` asks which single neurons a domain keeps using. A neuron here is one input channel of a module's output projection: an MLP down-projection channel or one channel of the concatenated head outputs. On every token, RACE measures how much of the neuron's write points along its module's whole update. It pools these scores over the prompts and ranks each neuron by a lower credible bound on its mean, so a neuron that pushes hard on a few tokens and idles on the rest ranks below a steady one.

```python
from murano import Pipeline
from murano.steps import LoadPrompts, RACE

code = Pipeline([LoadPrompts(code_texts), RACE(model)]).run()["race"]
general = Pipeline([LoadPrompts(wiki_texts), RACE(model)]).run()["race"]

neurons = code.select(fraction=0.01, exclude=general)
print([str(n) for n in neurons][:3])  # e.g. ['L0.mlp#1844', 'L0.mlp#623', ...]
```

`select` keeps the top 1% of each layer and module, because score scales differ between layers. With `exclude=`, it skips the neurons the reference run puts in its own top 1%, which leaves the ones specific to the target domain. The result stores per-neuron counts, means and spreads, so `result.cam(gamma=0.1)` needs no new forward pass. `plot_race_layers` draws each layer's score distribution. The method comes from Wang et al. (2026), [arXiv:2608.24758](https://arxiv.org/abs/2608.24758).
