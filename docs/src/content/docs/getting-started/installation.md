---
title: Installation
description: How to install Murano.
---

## Requirements

- Python 3.10+
- PyTorch
- A HuggingFace model or local snapshot

## Install from PyPI

```bash
pip install murano
```

## Install latest release

```bash
pip install https://github.com/UKPLab/murano/releases/latest/download/murano-latest.whl
```

## Development install

```bash
git clone https://github.com/UKPLab/murano
cd murano
uv sync --all-extras --dev
```