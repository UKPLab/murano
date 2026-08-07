"""Deterministic numerical primitives for FEGA stability analysis."""

from __future__ import annotations

import math
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import FEGAConfig
from .geometry import GeometryMetrics, compute_geometry_metrics


_SPAN_K = (2, 3, 4, 8)
_RESIDUAL_K = (2, 3, 4)
_ANGLE_FAMILIES = {
    "axis_or_antipodal",
    "global_2D_directional_subspace",
    "global_kD_directional_subspace",
    "residual_lowD_k",
}


def derive_seed(
    base_seed: int, feature_id: int, effect_space: str, salt: int = 0
) -> int:
    """Derive a stable 31-bit seed from a feature and effect-space identifier."""
    # Mix integer identifiers with a position-sensitive string checksum.
    effect_code = sum(
        (index + 1) * ord(char) for index, char in enumerate(effect_space)
    )
    return (base_seed + feature_id * 104729 + effect_code * 37 + salt) % (2**31 - 1)


def low_context_protocol(n_valid: int) -> dict[str, str | int]:
    """Select the permitted stability protocol for the available context count."""
    # Apply the fixed evidence tiers from most restrictive to principal-angle analysis.
    if n_valid < 8:
        return {
            "status": "insufficient_contexts",
            "protocol": "descriptive",
            "n_valid": int(n_valid),
        }
    if n_valid < 16:
        return {
            "status": "exploratory",
            "protocol": "leave_out_sensitivity",
            "n_valid": int(n_valid),
        }
    if n_valid < 32:
        return {
            "status": "exploratory",
            "protocol": "exploratory_subsampling",
            "n_valid": int(n_valid),
        }
    return {"status": "ok", "protocol": "principal_angle", "n_valid": int(n_valid)}


def g_orthonormal_basis(
    rows: ArrayLike,
    gram: ArrayLike,
    *,
    eig_floor: float = 1e-10,
) -> dict[str, object]:
    """Build a Gram-orthonormal row-span basis using a symmetric dual kernel.

    The eigendecomposition is performed in float64 on ``rows @ gram @ rows.T``.
    Tiny negative eigenvalues from roundoff are clamped to zero, and directions
    at or below ``eig_floor`` are omitted.
    """
    # Diagonalize the symmetric dual kernel and lift retained vectors to feature space.
    row_array = np.asarray(rows, dtype=np.float64)
    gram_array = np.asarray(gram, dtype=np.float64)
    if row_array.ndim != 2 or gram_array.shape != (row_array.shape[1],) * 2:
        raise ValueError("rows must be 2D and gram must match their feature dimension")
    kernel = row_array @ gram_array @ row_array.T
    eigenvalues, eigenvectors = np.linalg.eigh((kernel + kernel.T) * 0.5)
    if np.any(eigenvalues < -1.0e-5):
        raise ValueError("gram kernel has eigenvalues below the negative tolerance")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    retained = eigenvalues > eig_floor
    kept_values = eigenvalues[retained]
    basis = row_array.T @ (eigenvectors[:, retained] / np.sqrt(kept_values))
    return {"basis": basis, "eigenvalues": kept_values, "rank": int(kept_values.size)}


def principal_angles_degrees(
    basis_a: ArrayLike,
    basis_b: ArrayLike,
    gram: ArrayLike,
    k: int | None = None,
) -> list[float]:
    """Return principal angles in degrees between two Gram-orthonormal bases."""
    # Convert singular values of the Gram cross-product into clipped angles.
    a = np.asarray(basis_a, dtype=np.float64)
    b = np.asarray(basis_b, dtype=np.float64)
    gram_array = np.asarray(gram, dtype=np.float64)
    resolved_k = min(a.shape[1], b.shape[1]) if k is None else int(k)
    if resolved_k <= 0 or a.shape[1] < resolved_k or b.shape[1] < resolved_k:
        raise ValueError("k must be positive and no larger than either basis rank")
    singular_values = np.linalg.svd(
        a[:, :resolved_k].T @ gram_array @ b[:, :resolved_k], compute_uv=False
    )
    return np.degrees(np.arccos(np.clip(singular_values, -1.0, 1.0))).tolist()


def subspace_resample_indices(
    n_valid: int,
    fraction: float,
    rounds: int,
    seed: int,
) -> list[NDArray[np.int64]]:
    """Draw a deterministic schedule of fixed-size subsets without replacement."""
    # Use one local generator so the same arguments reproduce the entire schedule.
    if n_valid <= 0 or rounds <= 0:
        return []
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    sample_size = math.ceil(fraction * n_valid)
    generator = np.random.default_rng(seed)
    return [
        np.sort(generator.choice(n_valid, size=sample_size, replace=False)).astype(
            np.int64
        )
        for _ in range(rounds)
    ]


def _c_ray_from_kernel(kernel: NDArray[np.float64]) -> float:
    """Compute source pairwise concentration from a unit-row kernel."""
    # Exclude the diagonal self-similarities from the ordered pair mean.
    n_rows = int(kernel.shape[0])
    if n_rows < 2:
        raise ValueError("c_ray requires at least two rows")
    return float((kernel.sum() - np.trace(kernel)) / (n_rows * (n_rows - 1)))


def group_sampling_status(
    labels: ArrayLike | None,
    min_group_count: int,
    min_group_size: int,
) -> dict[str, object]:
    """Report whether enough label groups are large enough for grouped sampling."""
    # Count labels and compare the eligible group total with the requested minimum.
    if labels is None:
        return {"status": "group_sampling_unavailable", "groups": {}}
    label_array = np.asarray(
        [
            label
            for label in np.asarray(labels, dtype=object).tolist()
            if label is not None
        ],
        dtype=object,
    )
    if label_array.size == 0:
        return {"status": "group_sampling_unavailable", "groups": {}}
    unique_labels, sizes = np.unique(label_array, return_counts=True)
    eligible = sizes >= min_group_size
    eligible_labels = unique_labels[eligible].tolist()
    return {
        "status": (
            "ok"
            if len(eligible_labels) >= min_group_count
            else "group_sampling_unavailable"
        ),
        "group_count": int(unique_labels.size),
        "eligible_group_count": len(eligible_labels),
        "eligible_labels": eligible_labels,
        "group_sizes": dict(
            zip(unique_labels.tolist(), sizes.astype(int).tolist(), strict=True)
        ),
    }


def selected_family_stability(
    family: str,
    selection_mode: str,
    selected_k: int | None,
    rows: ArrayLike,
    gram: ArrayLike,
    point: GeometryMetrics,
    group_labels: Sequence[str | None],
    config: FEGAConfig,
    feature_id: int,
    assignment_stability: float | None = None,
) -> dict[str, Any]:
    """Run the source FEGA protocols for one already selected point family.

    Fallback and terminal labels are deliberately not profiled. Strict mixtures
    reuse standalone assignment stability; other strict families run only their
    declared scalar, subset, and selected-dimension angle checks.
    """
    # Freeze the selected family before resampling so stability cannot relabel it.
    row_array = np.asarray(rows, dtype=np.float64)
    gram_array = np.asarray(gram, dtype=np.float64)
    protocol = low_context_protocol(len(row_array))
    evidence: dict[str, Any] = {
        "family": family,
        "selection_mode": selection_mode,
        "selected_k": selected_k,
        "protocol": protocol,
        "groups": group_sampling_status(
            np.asarray(group_labels, dtype=object),
            config.min_group_count,
            config.min_group_size,
        ),
    }
    if selection_mode != "strict":
        evidence.update(decision="not_evaluated", reason="point_fallback_or_terminal")
        return evidence
    if family == "multi_mode_directional_geometry":
        evidence.update(
            decision="stable",
            assignment_stability="reused",
            assignment_stability_value=assignment_stability,
        )
        return evidence

    # Retain only protocols requested by the source selected-family schedule.
    seed = derive_seed(config.seed, feature_id, "final_resid")
    outcomes: dict[str, str] = {}
    if family in {"directed_ray", "axis_or_antipodal"}:
        kernel = row_array @ gram_array @ row_array.T
        scalar = _identity_bootstrap_c_ray(
            kernel,
            family,
            config.bootstrap_rounds,
            seed,
            feature_id,
            config.ci_quantiles,
        )
        evidence["scalar_ci"] = scalar
        outcomes["scalar_ci"] = str(scalar["status"])

    point_margins = _family_margins(family, selected_k, point)
    leave_indices = [
        np.delete(np.arange(len(row_array), dtype=np.int64), omitted)
        for omitted in range(len(row_array))
    ]
    groups = sorted({str(label) for label in group_labels if label is not None})
    leave_indices.extend(
        np.asarray(
            [index for index, label in enumerate(group_labels) if str(label) != group],
            dtype=np.int64,
        )
        for group in groups
    )
    leave = _subset_protocol(
        row_array, gram_array, leave_indices, family, selected_k, point_margins, config
    )
    evidence["leave_out"] = leave
    outcomes["leave_out"] = str(leave["status"])

    if len(row_array) >= 16:
        rounds = (
            config.strong_sample_size_rounds
            if len(row_array) >= 32
            else min(config.sample_size_rounds, config.max_enumerated_subsets)
        )
        samples = _sample_size_indices(
            len(row_array), config.sample_sizes, rounds, seed + 31, feature_id
        )
        sample_size = _subset_protocol(
            row_array,
            gram_array,
            samples,
            family,
            selected_k,
            point_margins,
            config,
        )
        evidence["sample_size"] = sample_size
        outcomes["sample_size"] = str(sample_size["status"])

    if family in {
        "axis_or_antipodal",
        "global_2D_directional_subspace",
        "global_kD_directional_subspace",
        "residual_lowD_k",
    }:
        angle = _selected_angle(
            row_array,
            gram_array,
            family,
            selected_k,
            seed,
            config,
        )
        evidence["principal_angle"] = angle
        outcomes["principal_angle"] = str(angle["status"])

    # Apply the source precedence: observed instability outranks unavailable evidence.
    if "unstable" in outcomes.values():
        decision = "unstable"
    elif "unavailable" in outcomes.values():
        decision = "unavailable"
    else:
        decision = "stable"
    evidence["decision"] = decision
    return evidence


def _identity_bootstrap_c_ray(
    kernel: NDArray[np.float64],
    family: str,
    rounds: int,
    seed: int,
    feature_id: int,
    quantiles: tuple[float, float],
) -> dict[str, Any]:
    """Bootstrap c-ray with the source plan-identity RNG for each replicate."""
    # Derive each resample independently so execution order cannot alter the draws.
    if rounds <= 0:
        return {
            "status": "not_applicable",
            "ci_low": None,
            "ci_high": None,
            "rounds": 0,
            "requested_count": 0,
            "valid_count": 0,
            "failed_count": 0,
            "non_applicable_count": 0,
            "skipped_count": 0,
            "instability_count": 0,
            "required_failure_count": 0,
            "counters": _protocol_counters(0, 0),
        }
    values = []
    n_rows = len(kernel)
    for replicate in range(rounds):
        rng = np.random.default_rng(
            _plan_seed(seed, feature_id, "bootstrap", str(n_rows), replicate)
        )
        indices = np.sort(rng.choice(n_rows, size=n_rows, replace=True))
        values.append(_c_ray_from_kernel(kernel[np.ix_(indices, indices)]))
    low, high = np.quantile(values, quantiles)
    status = (
        "stable"
        if (
            family == "directed_ray"
            and low >= 0.80
            or family == "axis_or_antipodal"
            and high < 0.80
        )
        else "unstable"
    )
    return {
        "status": status,
        "ci_low": float(low),
        "ci_high": float(high),
        "rounds": int(rounds),
        "requested_count": int(rounds),
        "valid_count": int(rounds),
        "failed_count": 0,
        "non_applicable_count": 0,
        "skipped_count": 0,
        "instability_count": int(status == "unstable"),
        "required_failure_count": 0,
        "counters": _protocol_counters(rounds, rounds),
    }


def _sample_size_indices(
    n_rows: int,
    targets: Sequence[int],
    rounds: int,
    seed: int,
    feature_id: int,
) -> list[NDArray[np.int64]]:
    """Build the source deterministic sample-size subset inventory."""
    # Bind each RNG to its target and replicate rather than worker scheduling.
    subsets: list[NDArray[np.int64]] = []
    for target in sorted({int(value) for value in targets if 0 < value <= n_rows}):
        target_rounds = 1 if target == n_rows else rounds
        for replicate in range(target_rounds):
            rng = np.random.default_rng(
                _plan_seed(seed, feature_id, "sample_size", str(target), replicate)
            )
            subsets.append(
                np.sort(rng.choice(n_rows, size=target, replace=False)).astype(np.int64)
            )
    return subsets


def _plan_seed(
    seed: int, feature_id: int, protocol: str, target: str, replicate: int
) -> int:
    """Return the source plan-identity seed from its scientific inputs."""
    # Hash only the immutable sampling identity used by the source RNG.
    identity = {
        "global_seed": int(seed),
        "feature_id": int(feature_id),
        "protocol": protocol,
        "target_or_group_identity": target,
        "replicate_id": int(replicate),
        "purpose": "all_scalars" if protocol == "bootstrap" else "gate_margins",
        "indices": (),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return int(digest[:16], 16)


def _subset_protocol(
    rows: NDArray[np.float64],
    gram: NDArray[np.float64],
    subsets: Sequence[NDArray[np.int64]],
    family: str,
    selected_k: int | None,
    point_margins: Mapping[str, float | None],
    config: FEGAConfig,
) -> dict[str, Any]:
    """Count gate-side changes across one declared subset protocol."""
    # Compare only the selected family's source margin inventory.
    crossings = 0
    instability_count = 0
    valid = 0
    failed = 0
    unavailable = 0
    gate_crossing_counts = {key: 0 for key in point_margins}
    selected_k_mismatch_count = 0
    strict_k_values = _strict_k_values(family, selected_k)
    for indices in subsets:
        try:
            metrics = compute_geometry_metrics(
                rows[indices],
                gram,
                k_values=config.span_k_values,
                residual_k_values=config.residual_k_values,
                eps=config.eps,
            )
            subset_margins = _family_margins(family, selected_k, metrics)
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            failed += 1
            continue
        valid += 1
        family_result = _locked_family_result(
            family,
            selected_k=selected_k,
            strict_k_values=strict_k_values,
            margins=subset_margins,
        )
        subset_crossed = False
        for key, point_margin in point_margins.items():
            subset_margin = subset_margins.get(key)
            if point_margin is None or subset_margin is None:
                continue
            if _margin_passes(key, point_margin) != _margin_passes(key, subset_margin):
                crossings += 1
                gate_crossing_counts[key] += 1
                subset_crossed = True
        if family_result["missing_required_margins"]:
            unavailable += 1
        mismatch = bool(family_result["selected_k_mismatch"])
        selected_k_mismatch_count += int(mismatch)
        if subset_crossed or mismatch:
            instability_count += 1
    status = (
        "unstable"
        if instability_count
        else "unavailable"
        if unavailable or failed
        else "not_applicable"
        if not subsets
        else "stable"
    )
    requested = int(len(subsets))
    return {
        "status": status,
        "requested": requested,
        "requested_count": requested,
        "valid_count": int(valid),
        "failed_count": int(failed),
        "non_applicable_count": 0,
        "skipped_count": 0,
        "instability_count": int(instability_count),
        "selected_k_mismatch_count": int(selected_k_mismatch_count),
        "required_margin_unavailable_count": int(unavailable),
        "gate_crossing_count": int(crossings),
        "gate_crossing_counts": gate_crossing_counts,
        "required_failure_count": int(failed + unavailable),
        "unavailable_count": int(unavailable),
        "counters": _protocol_counters(
            requested,
            valid,
            failed=failed,
        ),
    }


def _selected_angle(
    rows: NDArray[np.float64],
    gram: NDArray[np.float64],
    family: str,
    selected_k: int | None,
    seed: int,
    config: FEGAConfig,
) -> dict[str, Any]:
    """Evaluate the one raw or centered-residual angle requested by a family."""
    # Low-context schedules intentionally omit principal-angle resampling.
    if len(rows) < 32:
        return {
            "status": "exploratory",
            "angle_p90_deg": None,
            "requested_count": 0,
            "valid_count": 0,
            "failed_count": 0,
            "skipped_count": 0,
            "k": 1 if family == "axis_or_antipodal" else selected_k,
            "n_valid": int(len(rows)),
            "numerical_rank": int(len(rows)),
            "threshold": None,
            "decision": "exploratory",
            "instability_count": 0,
            "required_failure_count": 0,
            "counters": _protocol_counters(0, 0, non_applicable=1),
        }
    k = 1 if family == "axis_or_antipodal" else selected_k
    if k is None:
        return {
            "status": "unavailable",
            "angle_p90_deg": None,
            "requested_count": 0,
            "valid_count": 0,
            "failed_count": 0,
            "skipped_count": 0,
            "k": None,
            "n_valid": int(len(rows)),
            "numerical_rank": int(len(rows)),
            "threshold": None,
            "decision": "unavailable",
            "instability_count": 0,
            "required_failure_count": 0,
            "counters": _protocol_counters(0, 0, non_applicable=1),
        }
    centered = family == "residual_lowD_k"
    offset = 53 if centered else 47
    schedules = subspace_resample_indices(
        len(rows),
        config.subspace_resample_fraction,
        config.subspace_resample_rounds,
        seed + offset,
    )
    basis_rows = rows - rows.mean(axis=0, keepdims=True) if centered else rows
    full = g_orthonormal_basis(basis_rows, gram, eig_floor=config.subspace_eig_floor)
    full_rank = int(np.asarray(full["rank"]).item())
    full_basis = np.asarray(full["basis"], dtype=np.float64)
    threshold = 30.0 if k <= 2 else 35.0
    requested = len(schedules)
    if not schedules:
        return {
            "status": "not_applicable",
            "angle_p90_deg": None,
            "requested_count": 0,
            "valid_count": 0,
            "failed_count": 0,
            "skipped_count": 0,
            "k": int(k),
            "n_valid": int(len(rows)),
            "numerical_rank": full_rank,
            "threshold": threshold,
            "decision": "not_applicable",
            "instability_count": 0,
            "required_failure_count": 0,
            "counters": _protocol_counters(0, 0, non_applicable=1),
        }
    if full_rank < k:
        return {
            "status": "unavailable",
            "angle_p90_deg": None,
            "requested_count": requested,
            "valid_count": 0,
            "failed_count": 1,
            "skipped_count": max(0, requested - 1),
            "k": int(k),
            "n_valid": int(len(rows)),
            "numerical_rank": full_rank,
            "threshold": threshold,
            "decision": "unavailable",
            "instability_count": 0,
            "required_failure_count": int(requested > 0),
            "counters": _protocol_counters(
                requested,
                0,
                failed=int(requested > 0),
                skipped=max(0, requested - 1),
            ),
        }
    angles = []
    for indices in schedules:
        try:
            sampled_rows = rows[indices]
            if centered:
                sampled_rows = sampled_rows - sampled_rows.mean(axis=0, keepdims=True)
            sampled = g_orthonormal_basis(
                sampled_rows, gram, eig_floor=config.subspace_eig_floor
            )
            sampled_rank = int(np.asarray(sampled["rank"]).item())
            sampled_basis = np.asarray(sampled["basis"], dtype=np.float64)
            if sampled_rank < k:
                raise ValueError("insufficient_rank")
            angles.append(
                max(principal_angles_degrees(full_basis, sampled_basis, gram, k))
            )
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            skipped = max(requested - len(angles) - 1, 0)
            return {
                "status": "unavailable",
                "angle_p90_deg": None,
                "requested_count": requested,
                "valid_count": len(angles),
                "failed_count": 1,
                "skipped_count": skipped,
                "k": int(k),
                "n_valid": int(len(rows)),
                "numerical_rank": full_rank,
                "threshold": threshold,
                "decision": "unavailable",
                "instability_count": 0,
                "required_failure_count": 1,
                "counters": _protocol_counters(
                    requested,
                    len(angles),
                    failed=1,
                    skipped=skipped,
                ),
            }
    angle = float(np.quantile(angles, config.subspace_angle_quantile))
    status = "stable" if angle <= threshold else "unstable"
    return {
        "status": status,
        "angle_p90_deg": angle,
        "threshold": threshold,
        "requested_count": len(schedules),
        "valid_count": len(angles),
        "failed_count": 0,
        "skipped_count": 0,
        "k": int(k),
        "n_valid": int(len(rows)),
        "numerical_rank": full_rank,
        "decision": status,
        "instability_count": int(status == "unstable"),
        "required_failure_count": 0,
        "counters": _protocol_counters(len(schedules), len(angles)),
    }


def _family_margins(
    family: str, selected_k: int | None, metrics: GeometryMetrics
) -> dict[str, float | None]:
    """Return positive-on-pass margins for one source-selected family."""
    # Keep each family on its exact point-gate comparisons and dimensions.
    if family == "directed_ray":
        return {
            "c_ray_ge": _margin(metrics.c_ray, 0.80, 1),
            "s_span_1_axis": _margin(metrics.s_span.get(1), 0.80, 1),
        }
    if family == "axis_or_antipodal":
        return {
            "c_ray_lt": _margin(metrics.c_ray, 0.80, -1),
            "s_span_1_axis": _margin(metrics.s_span.get(1), 0.80, 1),
            "b_axis": _margin(metrics.b_axis, 0.15, 1),
        }
    if selected_k is None:
        return {}
    ks = (
        _SPAN_K[: _SPAN_K.index(selected_k) + 1]
        if family.startswith("global_")
        else _RESIDUAL_K[: _RESIDUAL_K.index(selected_k) + 1]
    )
    margins: dict[str, float | None] = {}
    for k in ks:
        if family.startswith("global_"):
            margins.update(
                {
                    f"s_span_{k}": _margin(metrics.s_span.get(k), 0.90, 1),
                    f"r_span_pr_k{k}": _margin(
                        metrics.r_span_pr, {2: 1.6, 3: 2.3, 4: 3.0, 8: 5.0}[k], 1
                    ),
                    f"u_span_{k}": _margin(
                        metrics.u_span.get(k),
                        {2: 0.08, 3: 0.05, 4: 0.03, 8: 0.01}[k],
                        1,
                    ),
                    f"d_span_{k}": _margin(metrics.d_span.get(k), 0.60, -1),
                }
            )
        else:
            margins.update(
                {
                    "e_res": _margin(metrics.e_res, 0.10, 1),
                    f"s_res_{k}": _margin(metrics.s_res.get(k), 0.80, 1),
                    f"r_ctr_pr_k{k}": _margin(
                        metrics.r_ctr_pr, {2: 1.5, 3: 2.2, 4: 2.9}[k], 1
                    ),
                }
            )
    return margins


def _margin(value: float | None, threshold: float, direction: int) -> float | None:
    """Return one finite signed threshold margin or explicit missingness."""
    # Positive margins pass; the axis c-ray complement remains strict at equality.
    return (
        float(direction) * (float(value) - threshold)
        if value is not None and math.isfinite(float(value))
        else None
    )


def _margin_passes(name: str, margin: float) -> bool:
    """Apply inclusive source gates except the strict c-ray complement."""
    # Equality fails only c_ray < tau; every other paper comparison is inclusive.
    return margin > 0.0 if name.endswith("_lt") else margin >= 0.0


def _strict_k_values(family: str, selected_k: int | None) -> tuple[int, ...]:
    """Return the cumulative strict-k candidate list for one family."""
    # Global and residual families compare every candidate up to the selected k.
    if selected_k is None:
        return ()
    if family.startswith("global_"):
        return _SPAN_K[: _SPAN_K.index(selected_k) + 1]
    if family == "residual_lowD_k":
        return _RESIDUAL_K[: _RESIDUAL_K.index(selected_k) + 1]
    return ()


def _required_family_margin_keys(
    family: str, strict_k_values: Sequence[int]
) -> tuple[str, ...]:
    """Return the complete source margin set required for one family."""
    # These families require every candidate margin up to the selected dimension.
    if family == "directed_ray":
        return ("c_ray_ge", "s_span_1_axis")
    if family == "axis_or_antipodal":
        return ("c_ray_lt", "s_span_1_axis", "b_axis")
    keys: list[str] = []
    for k in strict_k_values:
        keys.extend(_candidate_margin_keys(family, int(k)))
    return tuple(dict.fromkeys(keys))


def _candidate_margin_keys(family: str, k: int) -> tuple[str, ...]:
    """Return one strict-k candidate's source margin keys."""
    # Global and residual candidates intentionally use different effective-rank gates.
    if family.startswith("global_"):
        return (f"s_span_{k}", f"r_span_pr_k{k}", f"u_span_{k}", f"d_span_{k}")
    if family == "residual_lowD_k":
        return ("e_res", f"s_res_{k}", f"r_ctr_pr_k{k}")
    return ()


def _locked_family_result(
    family: str,
    *,
    selected_k: int | None,
    strict_k_values: Sequence[int],
    margins: Mapping[str, float | None],
) -> dict[str, object]:
    """Derive missing-margin and selected-k mismatch facts for one subset."""
    # Track selected-k mismatches separately from missing margins and gate crossings.
    required = _required_family_margin_keys(family, strict_k_values)
    missing = [key for key in required if margins.get(key) is None]
    if family in {"directed_ray", "axis_or_antipodal"}:
        return {
            "selected_k_mismatch": False,
            "missing_required_margins": missing,
        }
    derived = next(
        (
            int(k)
            for k in strict_k_values
            if all(
                margins.get(key) is not None
                for key in _candidate_margin_keys(family, int(k))
            )
            and all(
                _margin_passes(key, cast(float, margins[key]))
                for key in _candidate_margin_keys(family, int(k))
            )
        ),
        None,
    )
    return {
        "selected_k_mismatch": derived is not None and derived != selected_k,
        "missing_required_margins": missing,
    }


def _protocol_counters(
    requested: int,
    valid: int,
    *,
    failed: int = 0,
    non_applicable: int = 0,
    skipped: int = 0,
) -> dict[str, int]:
    """Build the source counter block for one protocol."""
    # Keep every protocol on the same requested/valid/failed/non-applicable/skipped schema.
    return {
        "requested": int(requested),
        "valid": int(valid),
        "failed": int(failed),
        "non_applicable": int(non_applicable),
        "skipped": int(skipped),
    }
