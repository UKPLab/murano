# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- **BREAKING:** the metric value types are renamed to `MetricScore` (one scalar) and `MetricComparison` (two labeled conditions); the refusal-specific `EvalResult` subclass is removed.
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
- Steps: `Load`, `Record`, `SteeringVector`, `Intervene`, `Probe`, `ComplianceRate`, `GenerationMetric`.
- Logit lens (`LogitLens` step).
- Datasets: `MuranoDataset` (contrastive) and `LabeledDataset`, with `from_hub()` and `from_template()` factories.
- Direction-based interventions: `ablate_direction`, `steer_direction`.
- I/O: `save_results()` with structured output layout, `load_steering()`, `save_ablated_model()`.
- Refusal-analysis submodule for safety-behavior analysis.
- Top-level `__version__` via `importlib.metadata`.
- `py.typed` marker for PEP 561 type-checker support.
- `murano_version` field in saved artifact metadata for forward-compatible reloading.
