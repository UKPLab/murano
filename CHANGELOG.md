# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `SelectComponents` step and `ComponentSelection` artifact: rank an attribution result (for example `LogitAttribution`) by magnitude, signed value, or most-negative, keep the top `top_k` or everything past a `threshold`, and write the chosen addresses for a downstream step to read. `Patch` / `PathPatch` / `Ablate` accept a `targets_key` / `senders_key` naming that selection, so attribute-then-patch runs as one pipeline instead of two with a hand-copied node list.
- `Intervene` gains `direction_layers` (`"all"`, `"best"`, or an explicit layer list) for the `direction_key` steering path, so a one-pipeline steer can apply only the best-separating layer's direction instead of every recorded layer, which keeps deep models coherent.
- `Sweep` step and `SweepResult` artifact: run a step chain once per item and harvest one or more metric keys. Every component study has this shape ("patch each head and measure what it restores", "zero each head and measure the damage", "steer at each layer"), and every notebook was hand-rolling it as a closure over the model, the task, and a baseline `Results`. `Sweep` forks the incoming `Results` per item, so the shared prefix runs once and the swept steps' writes stay out of the pipeline, and it derives its own read contract from the chain, so a missing upstream key fails pre-flight validation. A sweep over `Node` addresses publishes the same `{Node: float}` map an attribution does, so it feeds `SelectComponents` and `plot_head_matrix` with no adapter: attribute, sweep, select and path-patch now compose in one pipeline.
- `plot_sweep` in `murano.plotting`, auto-dispatched by `Plot`: a sweep over attention heads renders as the layer-by-head heatmap, anything else as one bar per item. The color scale diverges only when the values actually straddle zero.
- `AnswerRankStep`: how many tokens outscore the correct answer at the answer position. Rank 0 means the model would emit it greedily. Like its sibling metric steps it resolves the answer position from the attention mask, so it stays correct under either padding side.
- `Logits` accepts `fn`, `layers`, `modules` and `per_head`, forwarding them to the backend's `forward_logits`, which already took them. It is the forward-pass analogue of `Intervene`: a twelve-layer steering sweep now costs twelve forward passes instead of twelve decodes.
- `SAEModel.decoder`: the whole decoder matrix, for callers projecting every feature through the unembedding. Previously reachable only as `sae_model._sae.W_dec`.
- `murano.tasks`: the two toy tasks the tutorials share, defined once and tested. `ioi()` builds the indirect-object-identification task as a `CleanCorruptDataset`; `sentiment()` returns contrastive sentences; `positive_word_rate()` is the crude scorer the steering notebooks use.
- `plot_activation_projection` in `murano.plotting`: reduce one component's activations with any scikit-learn-style reducer (PCA, LDA, t-SNE, UMAP) and scatter them by class. It accepts both a contrastive `ActivationStore` and a `LabeledActivationStore`, so a single `Record` can feed both `Probe` and the plot.
- `zmid` on `plot_heatmap` and `plot_head_matrix`, anchoring a diverging colorscale at zero so a signed statistic no longer shades zero as if it had a sign.
- `notebooks/getting_started.ipynb`, plus fourteen runnable notebooks under `notebooks/applications/`: steering, probing, logit lens, logit attribution, attention, ablation, activation patching, circuit discovery, metrics, custom pipeline, weight ablation, and the three sparse-autoencoder notebooks. All fifteen share one template, enforced by `tests/test_notebook_structure.py`, which also rejects a step constructed inside a loop (that is a hand-rolled `Sweep`) and a pipeline built inside a function.
- The notebooks now render on the documentation site, generated from the executed `.ipynb` files by `docs/scripts/gen_notebook_docs.py` at deploy time.

### Changed

- The SAE notebooks move from `notebooks/sae/` into `notebooks/applications/` as `sae_features`, `sae_steering` and `sae_enrichment`, on the shared template. `sae_enrichment` replaces a 150-line bespoke ranking heuristic with the standardized mean difference between classes, which is four lines and surfaces the same sentiment features, along with the negation and junk features that ranking alone cannot tell apart.

### Fixed

- `Intervene`'s docstring recommended `direction_layers="best"` while the code defaulted to `"all"`, and never stated the default. It now documents the default and explains the trade-off: the best-*separating* layer is not necessarily an effective place to intervene, because a concept is often most separable in an early layer whose contribution later layers overwrite.
- `sae_steer`'s docstring quoted `alpha=200` as if it were a usable default. `alpha` is an absolute magnitude added at every decoded token, so it means nothing without the residual norm at the steering site, which spans tens to thousands across models. The docstring now says so, and points at `notebooks/applications/sae_steering.ipynb`, which measures that norm and finds no strength that both invokes the concept and preserves the text. The notebook's old `alpha=2000` output predates the fix that made generation-time interventions apply on every decoded token.
- `tests/test_notebook_structure.py` compared function *names* across notebooks, so `patch_head` and `recovered_fraction`, the same seventeen lines twice, lived in two notebooks unnoticed. It now compares normalized syntax trees.

### Removed

- The `examples/` directory. Its scripts became notebooks under `notebooks/`, and the one reusable class it held (`PlotterLens`) moved into the library as `murano.plotting.plot_activation_projection`.
- `.coveragerc`, which duplicated the `[tool.coverage]` configuration in `pyproject.toml` and took precedence over it.

## [0.1.0a2] - 2026-07-08

### Added

- `Node` addressing for model components: a single scheme naming a layer, submodule (residual / MLP / attention), and optional head, Q/K/V/O side, and token position, with a string form (`L5.self_attn.h3.Q`) and `NodeSet` / `NodeDict` / `Edge` helpers.
- Causal-analysis steps: `Logits` (forward pass exposing output logits and next-token targets), `Patch` (cross-run activation patching), `PathPatch` (direct-path patching, with per-head and per-side Q/K/V receivers), `Ablate` (zero / mean / resample a component, with support for precomputed per-head means), `LogitAttribution` (per-head and per-MLP direct logit attribution), and `LoadPaired` with clean/corrupt paired datasets.
- Generic attention analysis: pattern capture, reductions, OV circuits, and plots (`RecordAttention`, `ov_circuit`).
- Evaluation metrics: logit difference, KL divergence, and recovered-fraction (`LogitDiffStep`, `KLDivergenceStep`, `RecoveredMetricStep`).
- `attn_implementation` and other loader keyword arguments are forwarded through `MuranoModel` to the nnterp loader, so models that need eager attention or `check_renaming=False` (for example GPT-J) can be loaded.
- Packaging metadata: SPDX `license`, `authors` / `maintainers`, project URLs, trove classifiers, and keywords.
- Documentation: SAE tutorials and a reproduction gallery (IOI, Geometry of Truth, Function Vectors).

### Changed

- **BREAKING:** activations are keyed by `Node` rather than `(layer, module)` tuples across `ActivationStore` / `SteeringResult` / `ProbeResult`.
- **BREAKING:** feature dependencies are split into per-use-case extras (`[probe]`, `[data]`, `[plot]`, `[sae]`, `[notebook]`, `[all]`); the base install carries only the recording / steering / intervention core.
- **BREAKING:** the metric value types are renamed to `MetricScore` (one scalar) and `MetricComparison` (two labeled conditions); the `EvalResult` subclass is removed.
- The source distribution is slimmed to the package and its supporting files; the docs site (and its `node_modules`), notebooks, and tutorials are excluded.

### Fixed

- Generation-time interventions apply on every decoded token, so a steering or ablation edit holds across a whole completion.
- Hardened serialization, loading, and architecture preconditions, and deduplicated internal helpers with added test coverage.

## [0.1.0a1] - 2026-06-15

### Added

- SAE support via the optional `[sae]` extra: `SAEEncode` and `SAETopActivations` steps, `SAEModel` wrapper around `sae-lens`, `SAEActivationStore` and `SAEFeatureExamples` artifacts with on-disk persistence (`load_sae_activations`, `load_sae_examples`).
- Full-position recording (`position="none"`, keeps every token) and per-head attention recording (`per_head=True`) in `Record` and `MuranoModel.record`. `ActivationStore` / `LabeledActivationStore` carry `position` and `per_head` metadata plus optional token masks, persisted and reloaded with backward-compatible defaults.
- `murano.keys` module exposing the canonical Results keys (`RECORD`, `STEERING`, ...) as constants.
- `ModelBackend` interface (`murano.backend`) and the model methods behind it on `MuranoModel` (`resolve_module`, `attn_out_proj`, `trace`, `project_on_vocab`, `hf_model`, `generate_with_hooks`). Pipeline steps now reach the model only through this interface instead of nnsight internals. No behavior change.

### Changed

- **BREAKING:** `ActivationKey` is now always `(layer, module)`. Single-module records previously keyed activations by bare `int` (e.g. `store.positive[5]`); they now use `(5, "residual")` like multi-module records. Update any code that indexes `ActivationStore`/`SteeringResult`/`ProbeResult` dicts by a bare layer index. Saved activation stores from earlier versions still load.
- Activation-space interventions reject direction keys that are not `(layer: int, module: str)` tuples, and `SteeringVector` / `Probe` reject full-position or per-head stores, replacing silent no-ops and wrong-shaped results with clear errors.

## [0.1.0] - TBD

Initial release.

### Added

- Quick API on `MuranoModel`: `find_direction()`, `generate(intervention=...)`, direct activation recording.
- Pipeline API: composable `Step` + `Pipeline` with pre-flight validation of `reads`/`writes` contracts.
- Steps: `Load`, `Record`, `SteeringVector`, `Intervene`, `Probe`, `GenerationMetric`.
- Logit lens (`LogitLens` step).
- Datasets: `MuranoDataset` (contrastive) and `LabeledDataset`, with `from_hub()` and `from_template()` factories.
- Direction-based interventions: `ablate_direction`, `steer_direction`.
- I/O: `save_results()` with structured output layout, `load_steering()`, `save_ablated_model()`.
- Top-level `__version__` via `importlib.metadata`.
- `py.typed` marker for PEP 561 type-checker support.
- `murano_version` field in saved artifact metadata for forward-compatible reloading.
