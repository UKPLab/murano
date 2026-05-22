# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
