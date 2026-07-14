---
title: Installation
description: Install Murano and set up your environment.
---

## Requirements

- Python 3.10 or higher
- PyTorch 2.7+
- A CUDA-capable GPU is recommended for running models

## Install

```bash
pip install murano-interp
```

The base install is deliberately lean: it carries only what every workflow needs
(recording, steering, intervention, the causal substrate). Feature-specific
libraries ship as extras, so you install for what you actually do:

| Extra | Use case | Pulls in |
| ----- | -------- | -------- |
| (base) | recording, steering, intervention, logits, ablation, metrics, paired datasets | nnsight, nnterp, torch, transformers |
| `probe` | linear probing | scikit-learn |
| `data` | loading datasets by name from the Hub | datasets |
| `plot` | figures and visualizations | plotly, kaleido |
| `sae` | sparse autoencoder features | sae-lens |
| `notebook` | running the example notebooks | jupyter, ipykernel, nltk |
| `all` | everything above | probe, data, plot, sae, notebook |

Combine extras as needed:

```bash
pip install "murano-interp[probe,plot]"   # probing with figures
pip install "murano-interp[all]"          # everything
```

If you call a feature whose extra is not installed, Murano raises a clear error
naming the extra to install.

The PyPI distribution is `murano-interp` (the bare name `murano` is held by an
unrelated OpenStack project). The Python module name is unchanged: `import murano`.

## Development install

If you want to contribute or run from source:

```bash
git clone https://github.com/UKPLab/murano
cd murano
uv sync --all-extras --dev
```

Or with pip:

```bash
pip install -e ".[plot]"
```

## HuggingFace setup

Murano loads models from HuggingFace Hub. On first use, models are downloaded to the local cache (`~/.cache/huggingface/hub`). Subsequent runs load from cache without network access.

If your models require authentication (e.g. gated Llama weights), log in first:

```bash
huggingface-cli login
```

## Verify the install

```python
import murano
model = murano.Model("meta-llama/Llama-3.2-1B-Instruct")
print(model)  # MuranoModel('meta-llama/Llama-3.2-1B-Instruct', layers=16, d=2048)
```
