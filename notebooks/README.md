# Murano tutorials

Hands-on Jupyter notebooks that walk through Murano workflows end-to-end.

## SAE (Sparse Autoencoders)

Requires the `[sae]` and `[notebook]` extras:

```bash
pip install -e ".[sae,notebook]"
```

- [`sae/01_basics.ipynb`](sae/01_basics.ipynb) — Load a pre-trained SAE from HuggingFace, encode prompts through it, and inspect what each feature fires on.
- [`sae/02_feature_steering.ipynb`](sae/02_feature_steering.ipynb) — Find an SAE feature for a concept (the "Golden Gate Bridge" workflow), extract its decoder direction, and steer model generations with it.
- [`sae/03_sst2_feature_enrichment.ipynb`](sae/03_sst2_feature_enrichment.ipynb) — Rank SST-2 SAE features by sentiment selectivity and visualize feature logit effects plus token activations.
