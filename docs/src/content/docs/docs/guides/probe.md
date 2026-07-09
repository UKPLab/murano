---
title: Probing
description: Train linear probes to test what information is encoded in activations.
---

Linear probing trains a classifier on top of frozen activations to test whether a concept is linearly represented at a given layer. Murano's `Probe` step runs cross-validated logistic regression per layer and reports accuracy.

## Full example

```python
from murano import LabeledDataset, MuranoModel, Pipeline
from murano.steps import Load, Record, Probe

model = MuranoModel("meta-llama/Llama-3.2-1B-Instruct")

# Load a labeled dataset
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
    Probe(cv=5),
]).run()

probe = results["probe"]
print(f"Best layer: {probe.best_layer}")
print(f"Accuracy: {probe.accuracy_per_layer[probe.best_layer]:.3f}")
```

## How it works

1. `Load` puts a `LabeledDataset` into the pipeline
2. `Record` extracts activations and produces a `LabeledActivationStore` with `.activations` and `.labels`
3. `Probe` runs `sklearn.cross_val_score` with a logistic regression classifier for each layer

The result is a `ProbeResult` containing:
- `accuracy_per_layer`: mean CV accuracy per layer
- `cv_scores`: per-fold scores per layer
- `best_layer`: layer with the highest mean accuracy
- `classifiers`: fitted classifiers (only if `refit=True`)
- `label_names`: human-readable class names

## Creating labeled datasets

### From HuggingFace Hub

```python
dataset = LabeledDataset.from_hub(
    "stanfordnlp/sst2",
    text_column="sentence",
    label_column="label",
    n=500,
    label_names=["negative", "positive"],
    template_fn=model.chat_template,  # optional chat wrapping
)
```

### From Python lists

```python
dataset = LabeledDataset.from_lists(
    texts=["I love this", "This is terrible", "Great movie", "Awful taste"],
    labels=[1, 0, 1, 0],
    label_names=["negative", "positive"],
)
```

## Custom classifier

By default `Probe` uses `LogisticRegression(max_iter=1000)`. You can pass any sklearn-compatible classifier:

```python
from sklearn.svm import LinearSVC

results = Pipeline([
    Load(dataset),
    Record(model, layers=[10, 15, 20], position="last"),
    Probe(classifier=LinearSVC(), cv=5, refit=True),
]).run()

# Access the fitted classifier for the best layer
clf = results["probe"].classifiers[results["probe"].best_layer]
```

## Plotting

Use the generic `Plot` step to visualize accuracy across layers. It renders a
plot for whichever results are present, so it picks up the probe automatically:

```python
from murano.steps import Plot

results = Pipeline([
    Load(dataset),
    Record(model, layers="all", position="last"),
    Probe(cv=5),
    Plot(),
]).run()
```

This requires the `plot` extra: `pip install "murano-interp[plot]"`.
