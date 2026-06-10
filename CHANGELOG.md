# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0a1] - 2026-06-10

### Added

- SAE support via the optional `[sae]` extra: `SAEEncode` and `SAETopActivations` steps, `SAEModel` wrapper around `sae-lens`, `SAEActivationStore` and `SAEFeatureExamples` artifacts with on-disk persistence (`load_sae_activations`, `load_sae_examples`).
- Full-position recording (`position="none"`, keeps every token) and per-head attention recording (`per_head=True`) in `Record` and `MuranoModel.record`. `ActivationStore` / `LabeledActivationStore` carry `position` and `per_head` metadata plus optional token masks, persisted and reloaded with backward-compatible defaults.

### Changed

- **BREAKING:** `ActivationKey` is now always `(layer, module)`. Single-module records previously keyed activations by bare `int` (e.g. `store.positive[5]`); they now use `(5, "residual")` like multi-module records. Update any code that indexes `ActivationStore`/`SteeringResult`/`ProbeResult` dicts by a bare layer index. Saved activation stores from earlier versions still load.
- Activation-space interventions reject direction keys that are not `(layer: int, module: str)` tuples, and `SteeringVector` / `Probe` reject full-position or per-head stores, replacing silent no-ops and wrong-shaped results with clear errors.

## [0.1.0] - TBD

Initial release.

### Added

- Quick API on `MuranoModel`: `find_direction()`, `generate(intervention=...)`, direct activation recording.
- Pipeline API: composable `Step` + `Pipeline` with pre-flight validation of `reads`/`writes` contracts.
- Steps: `Load`, `Record`, `SteeringVector`, `Intervene`, `Probe`, `ComplianceRate`, `GenerationMetric`.
- Logit lens (`murano.lenses`).
- Datasets: `MuranoDataset` (contrastive) and `LabeledDataset`, with `from_hub()` and `from_template()` factories.
- Direction-based interventions: `ablate_direction`, `steer_direction`.
- I/O: `save_results()` with structured output layout, `load_steering()`, `save_ablated_model()`.
- Refusal-analysis submodule for safety-behavior analysis.
- Top-level `__version__` via `importlib.metadata`.
- `py.typed` marker for PEP 561 type-checker support.
- `murano_version` field in saved artifact metadata for forward-compatible reloading.
