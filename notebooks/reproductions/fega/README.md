# Feature-Effect Geometry Analysis reproduction

This directory accompanies the Murano reproduction of [Sparse Autoencoders
Encode Both Concepts and Functions](https://arxiv.org/abs/2607.24645). See the
[FEGA project page](https://ukplab.github.io/FEGA/) and
[original implementation](https://github.com/UKPLab/FEGA) for the full study.

The executed notebook does three things:

1. removes Gemma 2 SAE feature `33760` from 64 prompts with Murano;
2. recomputes FEGA geometry for eight saved effect matrices, one per geometry
   family; and
3. compares the new labels, measurements, and plots with the original FEGA
   results.

FEGA-specific analysis code lives in `fega_method/` and is not part of Murano's
public API.

## Run the notebook

From the Murano repository root:

```bash
uv sync --frozen --extra notebook --dev
uv pip install -r notebooks/reproductions/fega/requirements.txt
uv run jupyter lab notebooks/reproductions/fega/hoang2026_feature_effect_geometry.ipynb
```

The live section requires a CUDA GPU and Hugging Face access to
`google/gemma-2-2b`. Keep its batch size at `8` when comparing with the stored
output; reduced-precision model calculations can change slightly under a
different batch partition.

## Tests

Run `uv run pytest notebooks/reproductions/fega/test_fega.py -q` explicitly;
the repository's default test discovery excludes notebook bundles. These CPU
tests cover the intervention, deterministic vMF fitting, geometry reporting,
stability decisions, and visualization routing.

## Included data and scope

`artifacts/effect_clouds.npz` contains 361 prompt-level effects from the
original FEGA runs. `inputs.json` describes their feature and prompt rows.
`expected.json` and `source_figures/` are used only after the notebook has
computed new results.

This reproduces FEGA's downstream analysis for eight examples. It does not
repeat dataset construction, feature selection, model-wide SAE sweeps,
aggregate family frequencies, or the full paper atlas.

## Regenerate the compact artifacts

Maintainers can extract the bundle from a completed source FEGA checkout:

```bash
uv run python notebooks/reproductions/fega/curate_reference.py \
    /path/to/SAEBench-modified
```

The script reads the completed outputs under `results/fega/ravel`, renders the
original comparison figures with that checkout's environment, validates all
eight examples, and then replaces `artifacts/`.

## License

The adapted FEGA code is available under the Apache License 2.0 in
[`LICENSE`](LICENSE). [`NOTICE`](NOTICE) records the spherecluster-derived code,
its original revision, changes, and MIT license terms.
