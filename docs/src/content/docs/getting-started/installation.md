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
pip install murano
```

To include plotting support (matplotlib + seaborn):

```bash
pip install murano[plot]
```

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
