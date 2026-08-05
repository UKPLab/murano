"""Scientific defaults for Murano's native FEGA implementation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FEGAConfig:
    """Configuration values that can change a FEGA result.

    Defaults match the source FEGA dense-CPU paper configuration. Runtime-only
    controls such as checkpoint paths and plot counts live on their phase steps,
    not in this method configuration.

    Attributes:
        seed: Base seed used to derive feature-local stochastic seeds.
        min_contexts: Minimum valid effect rows required for geometry analysis.
        eps: Numerical tolerance used for zero-norm filtering and spectra.
        span_k_values: Candidate dimensions for uncentered span geometry.
        residual_k_values: Candidate dimensions for centered residual geometry.
        vmf_k_values: Candidate component counts for dense vMF fitting.
        vmf_bic_tolerance: Tolerance within which the smaller mixture wins.
        vmf_resample_fraction: Fraction retained by assignment-stability resamples.
        vmf_resample_rounds: Number of assignment-stability resamples.
        vmf_n_init: Seeded EM initializations evaluated per component count.
        vmf_max_iter: Maximum EM iterations per initialization.
        warn_contexts: Context count above which dense vocabulary allocation warns.
        bootstrap_rounds: Scalar bootstrap replicates used by stability.
        ci_quantiles: Lower and upper scalar bootstrap quantiles.
        subspace_resample_fraction: Fraction retained by subspace resamples.
        subspace_resample_rounds: Number of subspace resamples.
        subspace_angle_quantile: Quantile reported for principal angles.
        subspace_eig_floor: Minimum supported eigenvalue for a subspace basis.
        sample_sizes: Context counts evaluated by sample-size stability.
        sample_size_rounds: Default random subsets per sample size.
        strong_sample_size_rounds: Subsets used near a decision boundary.
        max_enumerated_subsets: Exact subsets retained before seeded sampling.
        min_group_count: Minimum groups required for grouped stability.
        min_group_size: Minimum rows required in each stability group.
    """

    seed: int = 42
    min_contexts: int = 8
    eps: float = 1.0e-12
    span_k_values: tuple[int, ...] = (1, 2, 3, 4, 8)
    residual_k_values: tuple[int, ...] = (1, 2, 3, 4)
    vmf_k_values: tuple[int, ...] = (1, 2, 3, 4)
    vmf_bic_tolerance: float = 1.0e-9
    vmf_resample_fraction: float = 0.80
    vmf_resample_rounds: int = 8
    vmf_n_init: int = 4
    vmf_max_iter: int = 200
    warn_contexts: int = 64
    bootstrap_rounds: int = 200
    ci_quantiles: tuple[float, float] = (0.025, 0.975)
    subspace_resample_fraction: float = 0.75
    subspace_resample_rounds: int = 20
    subspace_angle_quantile: float = 0.90
    subspace_eig_floor: float = 1.0e-8
    sample_sizes: tuple[int, ...] = (8, 16, 32, 64, 128, 256)
    sample_size_rounds: int = 20
    strong_sample_size_rounds: int = 50
    max_enumerated_subsets: int = 20
    min_group_count: int = 2
    min_group_size: int = 2
