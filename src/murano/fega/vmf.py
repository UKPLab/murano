"""Dense CPU von Mises--Fisher mixture fitting for FEGA."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import hashlib
import math
import warnings
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import torch
from joblib import Parallel, delayed
from scipy.integrate import quad
from scipy.special import gammaln, i0e, ive, logsumexp
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import adjusted_rand_score
from sklearn.utils.extmath import squared_norm, stable_cumsum


_MAX_CONCENTRATION = 1e10
_MAX_VALIDATED_DIMENSION = 256_000
_SATURATION_RBAR = 1.0 - 1.0e-10
_QUADRATURE_RELATIVE_ERROR_LIMIT = 1.0e-11


@dataclass(frozen=True)
class VMFFit:
    """A fitted finite mixture of von Mises--Fisher distributions."""

    n_components: int
    weights: np.ndarray
    means: np.ndarray
    concentrations: np.ndarray
    labels: np.ndarray
    responsibilities: np.ndarray
    log_likelihood: float
    bic: float
    converged: bool
    n_iter: int


@dataclass(frozen=True)
class VMFSelectedFit:
    """Compact selected mixture state needed by stability and reporting."""

    n_components: int
    weights: np.ndarray
    concentrations: np.ndarray
    labels: np.ndarray
    log_likelihood: float
    bic: float
    converged: bool
    n_iter: int


@dataclass(frozen=True)
class VMFCandidateEvidence:
    """Scalar evidence retained for one attempted component count."""

    n_components: int
    status: Literal["finite", "fit_failed"]
    log_likelihood: float | None = None
    bic: float | None = None
    converged: bool | None = None
    n_iter: int | None = None


@dataclass(frozen=True)
class VMFSelection:
    """Compact selected state and scalar evidence for successful candidates."""

    selected: VMFSelectedFit
    candidates: tuple[VMFCandidateEvidence, ...]


class NoFiniteVMFCandidate(FloatingPointError):
    """Report a fixed candidate schedule with no finite vMF fit."""

    def __init__(self, candidates: tuple[VMFCandidateEvidence, ...]) -> None:
        """Retain source-style failure evidence for the calling feature loop."""
        # Let the pipeline continue per feature without discarding attempted counts.
        super().__init__("all vMF component candidates failed")
        self.candidates = candidates


def feature_seed(base_seed: int, feature_id: int) -> int:
    """Return the deterministic FEGA seed assigned to one feature."""

    # Keep adjacent features on distinct, reproducible random streams.
    return int(base_seed) + int(feature_id) * 104729


def fit_vmf(
    directions: np.ndarray,
    n_components: int,
    seed: int,
    n_init: int = 4,
    max_iter: int = 200,
) -> VMFFit:
    """Fit a dense CPU vMF mixture and retain the best seeded initialization."""

    # Validate once, then compare finite fits in initialization order.
    data = _validate_directions(directions, warn_large=True)
    if not 1 <= n_components <= len(data):
        raise ValueError("n_components must be between 1 and the number of rows")
    if n_init < 1 or max_iter < 1:
        raise ValueError("n_init and max_iter must be positive")

    return _fit_many(data, n_components, seed, n_init, max_iter)


def select_vmf(
    directions: np.ndarray,
    k_values: tuple[int, ...] = (1, 2, 3, 4),
    seed: int = 0,
    n_init: int = 4,
    max_iter: int = 200,
    bic_tolerance: float = 1e-9,
    n_jobs: int = 1,
    warn_large: bool = True,
) -> VMFSelection:
    """Fit valid component counts and select BIC, resolving ties toward smaller k."""

    # Normalize candidate order before parallel fitting so worker count cannot affect selection.
    data = _validate_directions(directions, warn_large=warn_large)
    component_counts = tuple(
        sorted({int(k) for k in k_values if 1 <= int(k) <= len(data)})
    )
    if not component_counts:
        raise ValueError("k_values contains no component count valid for the input")
    if n_init < 1 or max_iter < 1:
        raise ValueError("n_init and max_iter must be positive")
    if bic_tolerance < 0:
        raise ValueError("bic_tolerance must be non-negative")

    attempted = cast(
        list[VMFFit | None],
        Parallel(n_jobs=n_jobs)(
            delayed(_fit_candidate)(
                data,
                k,
                _derived_seed(seed, k, -1, "candidate_fit"),
                n_init,
                max_iter,
            )
            for k in component_counts
        ),
    )
    fits = [fit for fit in attempted if fit is not None]
    evidence = tuple(
        VMFCandidateEvidence(
            k,
            "fit_failed" if fit is None else "finite",
            None if fit is None else fit.log_likelihood,
            None if fit is None else fit.bic,
            None if fit is None else fit.converged,
            None if fit is None else fit.n_iter,
        )
        for k, fit in zip(component_counts, attempted, strict=True)
    )
    if not fits:
        raise NoFiniteVMFCandidate(evidence)
    selected = fits[0]
    for candidate in fits[1:]:
        if candidate.bic < selected.bic - bic_tolerance:
            selected = candidate
    return VMFSelection(
        selected=_compact_selected_fit(selected),
        candidates=evidence,
    )


def assignment_stability(
    directions: np.ndarray,
    fit: VMFSelectedFit,
    seed: int,
    fraction: float = 0.8,
    rounds: int = 8,
    n_jobs: int = 1,
    n_init: int = 4,
    max_iter: int = 200,
) -> float | None:
    """Return mean full-data assignment ARI after seeded subset refits."""

    # A one-component solution has no assignments whose stability can be measured.
    if fit.n_components <= 1:
        return None

    # Refit independent deterministic subsets, then compare their full-data assignments.
    data = _validate_directions(directions, warn_large=False)
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if rounds < 1:
        raise ValueError("rounds must be positive")
    if n_init < 1 or max_iter < 1:
        raise ValueError("n_init and max_iter must be positive")
    subset_size = max(fit.n_components, int(np.ceil(fraction * len(data))))
    if subset_size > len(data):
        raise ValueError("not enough rows to refit all mixture components")

    scores = cast(
        list[float | None],
        Parallel(n_jobs=n_jobs)(
            delayed(_stability_round)(
                data, fit, seed, replicate, subset_size, n_init, max_iter
            )
            for replicate in range(rounds)
        ),
    )
    if any(score is None for score in scores):
        return None
    return float(sum(score for score in scores if score is not None) / rounds)


def _fit_many(
    data: np.ndarray,
    n_components: int,
    seed: int,
    n_init: int,
    max_iter: int,
) -> VMFFit:
    """Fit multiple starts without repeating public boundary validation."""

    # Preserve initialization order so exact likelihood ties prefer the earlier start.
    best: VMFFit | None = None
    init_seeds = np.random.RandomState(seed % 2**32).randint(
        np.iinfo(np.int32).max, size=n_init
    )
    for init_seed in init_seeds:
        try:
            candidate = _fit_once(data, n_components, int(init_seed), max_iter)
        except (FloatingPointError, np.linalg.LinAlgError):
            continue
        if np.isfinite(candidate.log_likelihood) and (
            best is None or candidate.log_likelihood > best.log_likelihood
        ):
            best = candidate
    if best is None:
        raise FloatingPointError(
            "all vMF initializations produced non-finite likelihoods"
        )
    return best


def _fit_candidate(
    data: np.ndarray,
    n_components: int,
    seed: int,
    n_init: int,
    max_iter: int,
) -> VMFFit | None:
    """Return one finite fixed-component candidate or source-style failure evidence."""

    # Isolate candidate failures so another feasible component count can still win BIC.
    try:
        return _fit_many(data, n_components, seed, n_init, max_iter)
    except (FloatingPointError, np.linalg.LinAlgError):
        return None


def _compact_selected_fit(fit: VMFFit) -> VMFSelectedFit:
    """Drop vocabulary-sized centers and responsibilities after BIC selection."""

    # Retain only the selected state consumed by stability, reporting, and plots.
    return VMFSelectedFit(
        fit.n_components,
        fit.weights,
        fit.concentrations,
        fit.labels,
        fit.log_likelihood,
        fit.bic,
        fit.converged,
        fit.n_iter,
    )


def _fit_once(data: np.ndarray, n_components: int, seed: int, max_iter: int) -> VMFFit:
    """Run one source-equivalent spherical mixture EM initialization."""

    # Seed source-equivalent EM with legacy k-means++ unit centers.
    means = _kmeans_plusplus(data, n_components, np.random.RandomState(seed))
    weights = np.full(n_components, 1.0 / n_components)
    concentrations = np.ones(n_components)
    responsibilities = np.empty((n_components, len(data)), dtype=np.float64)
    converged = False
    convergence_tolerance = float(np.mean(np.var(data, axis=0)) * 1e-6)
    iteration = 0

    for iteration in range(1, max_iter + 1):
        # Compute the source posterior before updating mixture parameters.
        previous_means = means.copy()
        logged_weights = np.full(n_components, -np.inf, dtype=np.float64)
        positive_weights = weights > 0.0
        logged_weights[positive_weights] = np.log(weights[positive_weights])
        log_scores = np.empty((n_components, len(data)), dtype=np.float64)
        shifted_normalizers = _log_normalizer_plus_kappa(concentrations, data.shape[1])
        for component in range(n_components):
            log_scores[component] = (
                concentrations[component] * (data.dot(means[component]).T - 1.0)
                + shifted_normalizers[component]
            )
            log_scores[component] += logged_weights[component]
        for row in range(len(data)):
            responsibilities[:, row] = np.exp(
                log_scores[:, row] - logsumexp(log_scores[:, row])
            )

        # Update weights, unit means, and the closed-form concentration approximation.
        resultants = np.einsum("kn,nd->kd", responsibilities, data, optimize=False)
        means = np.zeros_like(resultants)
        concentrations = np.zeros(n_components, dtype=np.float64)
        for component in range(n_components):
            weights[component] = np.mean(responsibilities[component])
            length = np.linalg.norm(resultants[component])
            if (
                not np.isfinite(weights[component])
                or weights[component] <= np.finfo(np.float64).eps
                or not np.isfinite(length)
                or length <= 1.0e-8
            ):
                weights[component] = (
                    max(0.0, weights[component])
                    if np.isfinite(weights[component])
                    else 0.0
                )
                means[component, component % data.shape[1]] = 1.0
                continue
            means[component] = resultants[component] / length
            rbar = length / (len(data) * weights[component])
            if not np.isfinite(rbar):
                continue
            if rbar >= _SATURATION_RBAR:
                concentrations[component] = _MAX_CONCENTRATION
            else:
                rbar = max(0.0, float(rbar))
                concentrations[component] = rbar * data.shape[1] - np.power(rbar, 3.0)
                concentrations[component] /= 1.0 - np.power(rbar, 2.0)
        if squared_norm(means - previous_means) <= convergence_tolerance:
            converged = True
            break

    # Preserve the last E-step posterior while scoring the final M-step parameters.
    mean_norms = np.linalg.norm(means, axis=1, keepdims=True)
    means = np.asarray(means / mean_norms, dtype=np.float64)
    normalized_weights = weights / math.fsum(float(weight) for weight in weights)
    logged_weights = np.full(n_components, -np.inf, dtype=np.float64)
    positive_weights = normalized_weights > 0.0
    logged_weights[positive_weights] = np.log(normalized_weights[positive_weights])
    dots = data @ means.T
    component_logs = np.empty((len(data), n_components), dtype=np.float64)
    shifted_normalizers = _log_normalizer_plus_kappa(concentrations, data.shape[1])
    for component in range(n_components):
        component_logs[:, component] = (
            logged_weights[component]
            + shifted_normalizers[component]
            + concentrations[component] * (dots[:, component] - 1.0)
        )
    row_log_likelihoods = np.asarray(
        logsumexp(component_logs, axis=1), dtype=np.float64
    )
    likelihood = float(math.fsum(float(value) for value in row_log_likelihoods))
    labels = np.argmax(responsibilities, axis=0)
    parameters = n_components * (data.shape[1] - 1) + n_components + (n_components - 1)
    bic = float(-2.0 * likelihood + parameters * np.log(len(data)))
    return VMFFit(
        n_components=n_components,
        weights=weights,
        means=means,
        concentrations=concentrations,
        labels=labels,
        responsibilities=responsibilities,
        log_likelihood=likelihood,
        bic=bic,
        converged=converged,
        n_iter=iteration,
    )


def _log_normalizer_plus_kappa(
    concentrations: np.ndarray, dimension: int
) -> np.ndarray:
    """Compute log C_d(kappa) + kappa stably for non-negative concentrations."""

    # Evaluate the source scalar path so high-dimensional Bessel underflow falls back.
    result = np.empty_like(concentrations, dtype=np.float64)
    for index, concentration in np.ndenumerate(concentrations):
        result[index] = _scalar_log_normalizer_plus_kappa(
            dimension, float(concentration)
        )
    return result


def _scalar_log_normalizer_plus_kappa(dimension: int, kappa: float) -> float:
    """Return the source shifted vMF normalizer over its validated dense domain."""
    # Validate the exact production-reachable dimension and concentration range.
    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, int | np.integer)
        or dimension < 2
        or dimension > _MAX_VALIDATED_DIMENSION
    ):
        raise ValueError(
            f"vMF dimension must be an integer in [2, {_MAX_VALIDATED_DIMENSION}]"
        )
    if not math.isfinite(kappa) or kappa < 0.0:
        raise FloatingPointError("vMF concentration must be finite and non-negative")
    rbar = math.nextafter(_SATURATION_RBAR, 0.0)
    maximum = max(
        _MAX_CONCENTRATION,
        (rbar * float(dimension) - rbar**3.0) / (1.0 - rbar**2.0),
    )
    if kappa > maximum:
        raise FloatingPointError("vMF concentration exceeds validated coverage")

    # Prefer the scaled-Bessel route and use the exact quadrature only on underflow.
    half_dimension = dimension / 2.0
    zero_limit = float(
        gammaln(half_dimension) - math.log(2.0) - half_dimension * math.log(math.pi)
    )
    if kappa == 0.0:
        return zero_limit
    order = half_dimension - 1.0
    scaled = float(i0e(kappa) if order == 0.0 else ive(order, kappa))
    if math.isfinite(scaled) and scaled > 0.0:
        value = (
            order * math.log(kappa)
            - half_dimension * math.log(2.0 * math.pi)
            - math.log(scaled)
        )
    else:
        value = zero_limit + _quadrature_kappa_minus_log_h(dimension, kappa)
    if not math.isfinite(value):
        raise FloatingPointError("vMF shifted log normalizer was not finite")
    return float(value)


def _quadrature_kappa_minus_log_h(dimension: int, kappa: float) -> float:
    """Return the source exact ``kappa - log(0F1)`` fallback by quadrature."""
    # Center the transformed spherical integral at its analytic mode.
    coefficient = float(dimension - 1)
    mode_tanh = 2.0 * kappa / (math.hypot(coefficient, 2.0 * kappa) + coefficient)
    root_coefficient = math.sqrt(coefficient)

    def scaled_integrand(scaled_offset: float) -> float:
        """Evaluate the exact mode-centered integrand in scaled coordinates."""
        # Apply hyperbolic addition identities before concentration multiplication.
        offset = scaled_offset / root_coefficient
        offset_tanh = math.tanh(offset)
        denominator = 1.0 + mode_tanh * offset_tanh
        magnitude = abs(offset)
        log_cosh = magnitude + math.log1p(math.exp(-2.0 * magnitude)) - math.log(2.0)
        exponent = coefficient * (
            mode_tanh * offset_tanh / denominator - log_cosh - math.log(denominator)
        )
        return math.exp(exponent) / root_coefficient

    integral, absolute_error = quad(
        scaled_integrand,
        -math.inf,
        math.inf,
        epsabs=1.0e-13,
        epsrel=1.0e-13,
        limit=300,
    )
    relative_error = absolute_error / integral if integral > 0.0 else math.inf
    if (
        not math.isfinite(integral)
        or integral <= 0.0
        or not math.isfinite(relative_error)
        or relative_error > _QUADRATURE_RELATIVE_ERROR_LIMIT
    ):
        raise FloatingPointError(
            "exact vMF normalizer quadrature did not meet its error contract"
        )

    # Assemble the shifted log-H value without concentration-scale subtraction.
    half_dimension = dimension / 2.0
    log_density_constant = float(
        gammaln(half_dimension)
        - 0.5 * math.log(math.pi)
        - gammaln(half_dimension - 0.5)
    )
    kappa_minus_mode = coefficient * mode_tanh / (1.0 + mode_tanh)
    kappa_minus_mode += 0.5 * coefficient * math.log(kappa / (coefficient * mode_tanh))
    return kappa_minus_mode - log_density_constant - math.log(integral)


def _validate_directions(directions: np.ndarray, *, warn_large: bool) -> np.ndarray:
    """Return the source-equivalent float64 normalized direction provider."""

    # Reject malformed retained rows at the public CPU boundary.
    values = np.asarray(directions)
    if values.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("directions must have float32 or float64 dtype")
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("directions must be a two-dimensional direction matrix")
    if len(values) < 8:
        raise ValueError("at least 8 valid direction rows are required")
    if not np.all(np.isfinite(values)):
        raise ValueError("directions must contain only finite values")
    normalized_rows: list[torch.Tensor] = []
    for row in torch.from_numpy(np.ascontiguousarray(values)):
        norm = torch.linalg.vector_norm(row)
        if not bool(torch.isfinite(norm)) or bool(norm <= 0.0):
            raise ValueError("direction rows must have finite positive norms")
        normalized_rows.append(row * (1.0 / norm))
    data = np.asarray(torch.stack(normalized_rows).numpy(), dtype=np.float64)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data = np.asarray(data / norms, dtype=np.float64)
    if warn_large and len(data) > 64:
        estimated_bytes = int(
            data.shape[0] * data.shape[1] * np.dtype(np.float32).itemsize
        )
        warnings.warn(
            f"retaining {len(data)} dense direction rows (estimated float32 bytes: {estimated_bytes})",
            RuntimeWarning,
            stacklevel=2,
        )
    return data


def _derived_seed(feature_seed_value: int, k: int, replicate_id: int, role: str) -> int:
    """Derive the source-compatible deterministic seed for one vMF operation."""

    # Hash semantic seed coordinates; legacy fitting folds only at RandomState.
    identity = f"vmf|{feature_seed_value}|{k}|{replicate_id}|{role}"
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")


def _stability_round(
    data: np.ndarray,
    fit: VMFSelectedFit,
    seed: int,
    replicate: int,
    subset_size: int,
    n_init: int,
    max_iter: int,
) -> float | None:
    """Compute one deterministic subset-refit assignment agreement score."""

    # Sample without replacement, refit, and classify every original direction.
    subset_seed = _derived_seed(seed, fit.n_components, replicate, "subset")
    refit_seed = _derived_seed(seed, fit.n_components, replicate, "refit")
    indices = np.sort(
        np.random.default_rng(subset_seed).choice(len(data), subset_size, replace=False)
    )
    try:
        refit = _fit_many(data[indices], fit.n_components, refit_seed, n_init, max_iter)
        score = float(adjusted_rand_score(fit.labels[indices], refit.labels))
    except (FloatingPointError, np.linalg.LinAlgError):
        return None
    return score if math.isfinite(score) else None


def _kmeans_plusplus(
    data: np.ndarray, n_components: int, random_state: np.random.RandomState
) -> np.ndarray:
    """Choose legacy sklearn-style k-means++ centers directly from unit rows."""

    # Match the old local-trial initializer used by spherecluster.
    centers = np.empty((n_components, data.shape[1]), dtype=data.dtype)
    centers[0] = data[random_state.randint(len(data))]
    closest_distances = euclidean_distances(
        centers[0, np.newaxis],
        data,
        Y_norm_squared=np.ones(len(data)),
        squared=True,
    )[0]
    current_potential = float(closest_distances.sum())
    local_trials = 2 + int(np.log(n_components))
    for center_id in range(1, n_components):
        random_values = random_state.random_sample(local_trials) * current_potential
        candidate_ids = np.searchsorted(stable_cumsum(closest_distances), random_values)
        np.clip(candidate_ids, None, len(data) - 1, out=candidate_ids)
        candidate_distances = euclidean_distances(
            data[candidate_ids],
            data,
            Y_norm_squared=np.ones(len(data)),
            squared=True,
        )
        candidate_distances = np.minimum(candidate_distances, closest_distances)
        candidate_potentials = candidate_distances.sum(axis=1)
        best = int(np.argmin(candidate_potentials))
        centers[center_id] = data[candidate_ids[best]]
        closest_distances = candidate_distances[best]
        current_potential = float(candidate_potentials[best])
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    return centers
