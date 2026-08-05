"""Compare current FEGA and Murano numerics in the source dependency stack."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch

from murano.fega.config import FEGAConfig
from murano.fega.geometry import GeometryMetrics, compute_geometry_metrics
from murano.fega.reporting import (
    GeometryRecord,
    classify_geometry,
    qualify_geometry,
)
from murano.fega.stability import (
    _plan_seed,
    _sample_size_indices,
    derive_seed as murano_stability_seed,
    final_resid_unit_rows as murano_final_resid_unit_rows,
    selected_family_stability,
)
from murano.fega.visualization import (
    CLASS_PALETTE,
    project_directions as murano_project_directions,
    render_projection_2d as murano_render_projection_2d,
    render_sphere_surface,
    surface_coordinates as murano_surface_coordinates,
)
from murano.fega.vmf import (
    _derived_seed,
    _validate_directions,
    assignment_stability as murano_assignment_stability,
    feature_seed,
    fit_vmf,
    select_vmf,
)


def _comparison(
    actual: torch.Tensor | np.ndarray,
    expected: torch.Tensor | np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Return compact float64 diagnostics under an authorized tolerance."""
    # Compare without retaining either full array in the JSON result.
    left = torch.as_tensor(actual, dtype=torch.float64).cpu()
    right = torch.as_tensor(expected, dtype=torch.float64).cpu()
    if left.shape != right.shape:
        return {
            "status": "mismatch",
            "shape_match": False,
            "actual_shape": list(left.shape),
            "expected_shape": list(right.shape),
        }
    error = (left - right).abs()
    close = torch.isclose(left, right, rtol=rtol, atol=atol, equal_nan=False)
    return {
        "status": "match" if bool(close.all()) else "mismatch",
        "shape_match": True,
        "count": left.numel(),
        "max_absolute_error": float(error.max()) if error.numel() else 0.0,
        "rtol": rtol,
        "atol": atol,
        "exceedance_count": int((~close).sum()),
    }


def _source_api(source_root: Path) -> dict[str, Any]:
    """Import the current source authorities from one explicit checkout."""
    # Put the method checkout and its vendored dependencies before installed copies.
    if not (source_root / "fega").is_dir():
        raise FileNotFoundError("source FEGA checkout is unavailable")
    sys.path[:0] = [str(source_root), str(source_root / "external")]
    from fega.config_schema import DirectionalMixtureFitConfig, FEGAPipelineConfig
    from fega.core.compute_effect.artifacts import summarize_magnitudes
    from fega.core.geometry_metrics.metrics import (
        c_ray_fast_final_resid,
        centered_residual_spectrum_final_resid,
        effective_rank_from_spectrum,
        normalize_logit_deltas,
        span_spectrum_final_resid,
    )
    from fega.core.geometry_reporting.classifier import classify_record
    from fega.core.geometry_reporting.point_selection import (
        point_selection_identity,
        resolve_point_selection,
    )
    from fega.core.geometry_reporting.records import _build_record
    from fega.core.geometry_reporting.thresholds import get_threshold_profile
    from fega.core.stability.artifacts import (
        StabilityGroupLookup,
        _group_labels_for_pairs,
    )
    from fega.core.stability.protocols import (
        bootstrap_plans,
        leave_out_plans,
        sample_size_plans,
    )
    from fega.core.stability.sampling import derive_seed
    from fega.core.stability.metrics import final_resid_unit_rows
    from fega.core.stability.runner import _ScheduledFeature, _execute_one
    from fega.core.stability.schedule import build_selected_family_schedule
    from fega.core.visualizations.projection import (
        project_directions,
        surface_coordinates,
    )
    from fega.core.visualizations.render import render_projection_2d, render_sphere
    from fega.core.vmf.factor_reuse import normalized_numpy_rows
    from fega.core.vmf.fit import fit_vmf_mixture
    from fega.core.vmf.metrics import (
        assignment_stability,
        default_fit_fn,
        derived_vmf_seed,
        feature_fit_seed,
        metrics_from_fit,
        score_vmf_feature,
    )

    return {
        "DirectionalMixtureFitConfig": DirectionalMixtureFitConfig,
        "FEGAPipelineConfig": FEGAPipelineConfig,
        "summarize_magnitudes": summarize_magnitudes,
        "c_ray_fast_final_resid": c_ray_fast_final_resid,
        "centered_residual_spectrum_final_resid": centered_residual_spectrum_final_resid,
        "effective_rank_from_spectrum": effective_rank_from_spectrum,
        "normalize_logit_deltas": normalize_logit_deltas,
        "span_spectrum_final_resid": span_spectrum_final_resid,
        "classify_record": classify_record,
        "point_selection_identity": point_selection_identity,
        "resolve_point_selection": resolve_point_selection,
        "_build_record": _build_record,
        "get_threshold_profile": get_threshold_profile,
        "StabilityGroupLookup": StabilityGroupLookup,
        "_group_labels_for_pairs": _group_labels_for_pairs,
        "bootstrap_plans": bootstrap_plans,
        "leave_out_plans": leave_out_plans,
        "sample_size_plans": sample_size_plans,
        "derive_seed": derive_seed,
        "final_resid_unit_rows": final_resid_unit_rows,
        "_ScheduledFeature": _ScheduledFeature,
        "_execute_one": _execute_one,
        "build_selected_family_schedule": build_selected_family_schedule,
        "project_directions": project_directions,
        "surface_coordinates": surface_coordinates,
        "render_projection_2d": render_projection_2d,
        "render_sphere": render_sphere,
        "normalized_numpy_rows": normalized_numpy_rows,
        "fit_vmf_mixture": fit_vmf_mixture,
        "assignment_stability": assignment_stability,
        "default_fit_fn": default_fit_fn,
        "derived_vmf_seed": derived_vmf_seed,
        "feature_fit_seed": feature_fit_seed,
        "metrics_from_fit": metrics_from_fit,
        "score_vmf_feature": score_vmf_feature,
    }


def _component_permutation(
    source_labels: np.ndarray, murano_labels: np.ndarray, k: int
) -> tuple[int, ...] | None:
    """Return the Murano component corresponding to each source component."""
    # Exhaust at most four labels, which is clearer than another matching dependency.
    for permutation in itertools.permutations(range(k)):
        if np.array_equal(np.asarray(permutation)[source_labels], murano_labels):
            return permutation
    return None


def _schedule_checks(
    api: dict[str, Any], metadata: dict[str, Any], config: FEGAConfig
) -> dict[str, Any]:
    """Compare explicit source and Murano subset indices with multiplicity."""
    # Rebuild each schedule independently from its published seed identity.
    feature_id = int(metadata["feature_id"])
    source_labels = metadata["source_labels"]
    n_rows = len(source_labels)
    source_seed = api["derive_seed"](
        config.seed, feature_id=feature_id, effect_space="final_resid"
    )
    murano_seed = murano_stability_seed(config.seed, feature_id, "final_resid")
    source_bootstrap = [
        plan.indices
        for plan in api["bootstrap_plans"](
            seed=source_seed,
            feature_id=feature_id,
            n_rows=n_rows,
            rounds=config.bootstrap_rounds,
        )
    ]
    murano_bootstrap = []
    for replicate in range(config.bootstrap_rounds):
        generator = np.random.default_rng(
            _plan_seed(source_seed, feature_id, "bootstrap", str(n_rows), replicate)
        )
        murano_bootstrap.append(
            tuple(sorted(generator.choice(n_rows, n_rows, replace=True).tolist()))
        )
    source_leave = [
        plan.indices
        for plan in api["leave_out_plans"](
            seed=source_seed + 17,
            feature_id=feature_id,
            n_rows=n_rows,
            group_labels=source_labels,
        )
    ]
    murano_leave = [
        tuple(index for index in range(n_rows) if index != omitted)
        for omitted in range(n_rows)
    ]
    for group in sorted({str(label) for label in source_labels if label is not None}):
        murano_leave.append(
            tuple(
                index
                for index, label in enumerate(source_labels)
                if str(label) != group
            )
        )
    rounds = (
        config.strong_sample_size_rounds
        if n_rows >= 32
        else min(config.sample_size_rounds, config.max_enumerated_subsets)
    )
    source_sample = [
        plan.indices
        for plan in api["sample_size_plans"](
            seed=source_seed + 31,
            feature_id=feature_id,
            n_rows=n_rows,
            targets=config.sample_sizes,
            rounds=rounds,
        )
    ]
    murano_sample = [
        tuple(indices.tolist())
        for indices in _sample_size_indices(
            n_rows,
            config.sample_sizes,
            rounds,
            murano_seed + 31,
            feature_id,
        )
    ]
    return {
        "status": (
            "match"
            if source_seed == murano_seed
            and source_bootstrap == murano_bootstrap
            and source_leave == murano_leave
            and source_sample == murano_sample
            else "mismatch"
        ),
        "feature_seed_match": source_seed == murano_seed,
        "bootstrap_indices_match": source_bootstrap == murano_bootstrap,
        "bootstrap_replicates": len(source_bootstrap),
        "leave_out_indices_match": source_leave == murano_leave,
        "leave_out_replicates": len(source_leave),
        "sample_size_indices_match": source_sample == murano_sample,
        "sample_size_replicates": len(source_sample),
    }


def _group_labels(api: dict[str, Any], metadata: dict[str, Any]) -> list[str | None]:
    """Resolve current source labels from the exact pair/context lookup inputs."""
    # Reconstruct tuple-keyed lookups from the transient JSON representation.
    context_labels = {int(index): label for index, label in metadata["context_labels"]}
    pair_labels = {
        (str(attribute), str(role), int(index)): label
        for attribute, role, index, label in metadata["pair_labels"]
    }
    lookup = api["StabilityGroupLookup"](
        context_labels=context_labels,
        pair_labels=pair_labels,
        source_paths=(),
    )
    labels = api["_group_labels_for_pairs"](
        metadata["context_indices"],
        metadata["pair_indices"],
        metadata["attribute_labels"],
        metadata["pair_roles"],
        lookup,
    )
    return labels or [None] * len(metadata["context_indices"])


def _geometry_checks(
    api: dict[str, Any], rows: np.ndarray, gram: np.ndarray, config: FEGAConfig
) -> tuple[dict[str, Any], GeometryMetrics, dict[str, Any]]:
    """Compare every source and Murano point-geometry field on one row cloud."""
    # Execute both public metric paths with the same dimensions and epsilon.
    source_rows = torch.from_numpy(np.asarray(rows))
    source_gram = torch.from_numpy(np.asarray(gram))
    ray = api["c_ray_fast_final_resid"](source_rows, source_gram, eps=config.eps)
    span = api["span_spectrum_final_resid"](
        source_rows,
        source_gram,
        k_values=config.span_k_values,
        eps=config.eps,
    )
    residual = api["centered_residual_spectrum_final_resid"](
        source_rows,
        source_gram,
        k_values=config.residual_k_values,
        eps=config.eps,
    )
    span_rank = api["effective_rank_from_spectrum"](span.eigenvalues, eps=config.eps)
    residual_rank = api["effective_rank_from_spectrum"](
        residual.eigenvalues, eps=config.eps
    )
    murano = compute_geometry_metrics(
        rows,
        gram,
        k_values=config.span_k_values,
        residual_k_values=config.residual_k_values,
        eps=config.eps,
    )

    # Flatten source records once for both diagnostics and reporting assembly.
    source_record: dict[str, Any] = {
        "feature_id": None,
        "n_valid": ray.n_valid,
        "r2": ray.r2,
        "c_ray": ray.c_ray,
        "r_span_pr": span_rank.r_pr,
        "r_span_ent": span_rank.r_ent,
        "b_axis": span.b_axis,
        "e_res": residual.e_res,
        "r_ctr_pr": residual_rank.r_pr,
        "r_ctr_ent": residual_rank.r_ent,
    }
    for key in config.span_k_values:
        source_record[f"s_span_{key}"] = span.s_span[key]
        source_record[f"u_span_{key}"] = span.u_span[key]
        source_record[f"d_span_{key}"] = span.d_span[key]
    for key in config.residual_k_values:
        source_record[f"s_res_{key}"] = residual.s_res[key]

    source_scalars = [
        ray.c_ray,
        ray.r2,
        span.b_axis,
        span_rank.r_ent,
        span_rank.r_pr,
        residual.e_res,
        residual_rank.r_ent,
        residual_rank.r_pr,
        *(span.s_span[key] for key in config.span_k_values),
        *(span.u_span[key] for key in config.span_k_values),
        *(span.d_span[key] for key in config.span_k_values),
        *(residual.s_res[key] for key in config.residual_k_values),
    ]
    murano_scalars = [
        murano.c_ray,
        murano.r2,
        murano.b_axis,
        murano.r_span_ent,
        murano.r_span_pr,
        murano.e_res,
        murano.r_ctr_ent,
        murano.r_ctr_pr,
        *(murano.s_span[key] for key in config.span_k_values),
        *(murano.u_span[key] for key in config.span_k_values),
        *(murano.d_span[key] for key in config.span_k_values),
        *(murano.s_res[key] for key in config.residual_k_values),
    ]
    counts_match = (
        ray.n_total,
        ray.n_valid,
        ray.skipped_nonfinite,
        ray.skipped_zero_norm,
    ) == (
        murano.n_total,
        murano.n_valid,
        murano.skipped_nonfinite,
        murano.skipped_zero_norm,
    )
    scalars = _comparison(
        np.asarray(source_scalars, dtype=np.float64),
        np.asarray(murano_scalars, dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    spectrum = _comparison(
        np.asarray(span.eigenvalues), murano.eigenvalues, rtol=1.0e-10, atol=1.0e-10
    )
    centered = _comparison(
        np.asarray(residual.eigenvalues),
        murano.centered_eigenvalues,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    return (
        {
            "status": (
                "match"
                if counts_match
                and scalars["status"] == "match"
                and spectrum["status"] == "match"
                and centered["status"] == "match"
                else "mismatch"
            ),
            "counts_match": counts_match,
            "scalars": scalars,
            "span_spectrum": spectrum,
            "centered_spectrum": centered,
        },
        murano,
        source_record,
    )


def _mixture_metrics(
    kernel: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    concentrations: np.ndarray,
    assignment: float | None,
) -> dict[str, float | int | None]:
    """Derive Murano's reporting mixture scalars from one selected live fit."""
    # Use hard labels only for per-mode rays and fitted weights for mixture gain.
    count = len(kernel)
    global_ray = float((kernel.sum() - np.trace(kernel)) / (count * (count - 1)))
    rays = []
    for mode in range(len(weights)):
        indices = np.flatnonzero(labels == mode)
        if len(indices) < 2:
            return {
                "selected_mode_count": len(weights),
                "mode_mass_min": float(np.min(weights)),
                "mode_kappa_min": float(np.min(concentrations)),
                "assignment_stability": assignment,
            }
        block = kernel[np.ix_(indices, indices)]
        rays.append(
            float((block.sum() - np.trace(block)) / (len(indices) * (len(indices) - 1)))
        )
    return {
        "selected_mode_count": len(weights),
        "delta_mix": float(np.dot(weights, rays) - global_ray),
        "mode_mass_min": float(np.min(weights)),
        "min_mode_c_ray": min(rays),
        "mode_kappa_min": float(np.min(concentrations)),
        "assignment_stability": assignment,
    }


def _config_checks(source: Any, murano: FEGAConfig) -> dict[str, Any]:
    """Verify every scientific setting consumed by the live comparison.

    Args:
        source: Parsed source ``FEGAPipelineConfig``.
        murano: Murano's native scientific configuration.

    Returns:
        A compact inventory of matched and mismatched scientific fields.
    """
    # Compare method settings only; devices, workers, batches, and checkpoints are free.
    phases = source.phases
    checks = {
        "global_seed": int(source.seed.global_) == murano.seed,
        "selection_seed": int(source.seed.selection_seed) == murano.seed,
        "minimum_contexts": int(phases.data_prep.min_contexts) == murano.min_contexts,
        "effect_normalization_eps": float(phases.compute_effect.normalization_eps)
        == murano.eps,
        "effect_tau_zero": float(phases.compute_effect.tau_zero) == murano.eps,
        "geometry_c_ray_eps": float(phases.geometry_metrics.c_ray.eps) == murano.eps,
        "geometry_span_eps": float(phases.geometry_metrics.span.eps) == murano.eps,
        "geometry_residual_eps": float(phases.geometry_metrics.resid.eps) == murano.eps,
        "geometry_rank_eps": float(phases.geometry_metrics.effective_rank.eps)
        == murano.eps,
        "span_dimensions": tuple(phases.geometry_metrics.span.k_values)
        == murano.span_k_values,
        "residual_dimensions": tuple(phases.geometry_metrics.resid.k_values)
        == murano.residual_k_values,
        "vmf_dimensions": tuple(phases.vmf.k_values) == murano.vmf_k_values,
        "vmf_bic_tolerance": float(phases.vmf.bic_tolerance)
        == murano.vmf_bic_tolerance,
        "vmf_resample_fraction": float(phases.vmf.resample_fraction)
        == murano.vmf_resample_fraction,
        "vmf_resample_rounds": int(phases.vmf.resample_rounds)
        == murano.vmf_resample_rounds,
        "vmf_initializations": int(phases.vmf.n_init) == murano.vmf_n_init,
        "vmf_iterations": int(phases.vmf.max_iter) == murano.vmf_max_iter,
        "bootstrap_rounds": int(phases.stability.scalar.bootstrap_rounds)
        == murano.bootstrap_rounds,
        "bootstrap_quantiles": tuple(phases.stability.scalar.ci_quantiles)
        == murano.ci_quantiles,
        "subspace_fraction": float(phases.stability.subspace.resample_fraction)
        == murano.subspace_resample_fraction,
        "subspace_rounds": int(phases.stability.subspace.resample_rounds)
        == murano.subspace_resample_rounds,
        "subspace_quantile": float(phases.stability.subspace.angle_p90_quantile)
        == murano.subspace_angle_quantile,
        "subspace_eigenvalue_floor": float(phases.stability.subspace.eig_floor)
        == murano.subspace_eig_floor,
        "sample_sizes": tuple(phases.stability.sample_size.target_sizes)
        == murano.sample_sizes,
        "sample_rounds": int(phases.stability.sample_size.subset_rounds)
        == murano.sample_size_rounds,
        "strong_sample_rounds": int(phases.stability.sample_size.strong_subset_rounds)
        == murano.strong_sample_size_rounds,
        "maximum_enumerated_subsets": int(
            phases.stability.sample_size.max_enumerated_subsets
        )
        == murano.max_enumerated_subsets,
        "minimum_group_count": int(phases.stability.leave_out.min_group_count)
        == murano.min_group_count,
        "minimum_group_size": int(phases.stability.leave_out.min_group_size)
        == murano.min_group_size,
        "vmf_effect_space": phases.vmf.effect_space == "pre_softcap_logits",
        "stability_effect_space": phases.stability.effect_space == "final_resid",
    }
    mismatches = sorted(name for name, matched in checks.items() if not matched)
    return {
        "status": "match" if not mismatches else "mismatch",
        "checked_fields": len(checks),
        "mismatches": mismatches,
    }


def _protocol_checks(
    source_evidence: dict[str, Any], murano_evidence: dict[str, Any]
) -> dict[str, Any]:
    """Compare executed source and Murano selected-family protocol summaries.

    Args:
        source_evidence: Raw source selected-family execution evidence.
        murano_evidence: Native Murano stability evidence.

    Returns:
        Status and field-level results for the shared protocol inventory.
    """
    # Map only equivalent summaries; source per-replicate arrays stay out of the report.
    protocols = source_evidence["protocols"]
    bootstrap = protocols["bootstrap"]
    scalar = murano_evidence["scalar_ci"]
    bootstrap_values = _comparison(
        np.asarray([bootstrap["ci_low"], bootstrap["ci_high"]]),
        np.asarray([scalar["ci_low"], scalar["ci_high"]]),
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    bootstrap_match = (
        bootstrap["status"] == scalar["status"]
        and int(bootstrap["counters"]["requested"]) == int(scalar["rounds"])
        and bootstrap_values["status"] == "match"
    )

    # Source instability counts one affected subset, matching Murano's crossing count.
    subset_results: dict[str, Any] = {}
    for source_name, murano_name in (
        ("leave_out", "leave_out"),
        ("sample_size", "sample_size"),
    ):
        source_block = protocols[source_name]
        murano_block = murano_evidence[murano_name]
        matched = (
            source_block["status"] == murano_block["status"]
            and int(source_block["counters"]["requested"])
            == int(murano_block["requested"])
            and int(source_block["instability_count"])
            == int(murano_block["gate_crossing_count"])
            and int(source_block["required_failure_count"])
            == int(murano_block["unavailable_count"])
        )
        subset_results[source_name] = {
            "status": "match" if matched else "mismatch",
            "requested": int(source_block["counters"]["requested"]),
        }

    # Low-context qualification is categorical and has the same source authority.
    source_low = protocols["low_context_qualification"]
    murano_low = murano_evidence["protocol"]
    low_context_match = all(
        source_low[key] == murano_low[key] for key in ("status", "protocol", "n_valid")
    )
    matched = (
        bootstrap_match
        and low_context_match
        and all(item["status"] == "match" for item in subset_results.values())
    )
    return {
        "status": "match" if matched else "mismatch",
        "bootstrap": {
            "status": "match" if bootstrap_match else "mismatch",
            "interval": bootstrap_values,
            "replicates": int(bootstrap["counters"]["requested"]),
        },
        "leave_out": subset_results["leave_out"],
        "sample_size": subset_results["sample_size"],
        "low_context_status_match": low_context_match,
    }


def _projection_checks(
    api: dict[str, Any],
    rows: np.ndarray,
    gram: np.ndarray,
    family: str,
) -> dict[str, Any]:
    """Compare live spectral projections and smoke-test both render paths.

    Args:
        api: Imported source FEGA authorities.
        rows: Shared retained final-residual directions.
        gram: Shared unembedding Gram matrix.
        family: Source-selected geometry family used only for plot color.

    Returns:
        Numerical projection, surface, and temporary render diagnostics.
    """
    # Project the same tensors independently in Torch and NumPy for both public views.
    source_rows = torch.from_numpy(np.asarray(rows))
    source_gram = torch.from_numpy(np.asarray(gram))
    comparisons: dict[str, Any] = {}
    projections: dict[int, tuple[Any, Any]] = {}
    for dimensions in (2, 3):
        source = api["project_directions"](
            source_rows, source_gram, dimensions=dimensions, centered=False
        )
        murano = murano_project_directions(
            rows, gram, dimensions=dimensions, centered=False
        )
        source_coordinates = source.coordinates.numpy()
        coordinate_frame = _comparison(
            source_coordinates,
            murano.coordinates,
            rtol=1.0e-8,
            atol=1.0e-10,
        )
        coordinate_geometry = _comparison(
            source_coordinates @ source_coordinates.T,
            murano.coordinates @ murano.coordinates.T,
            rtol=1.0e-8,
            atol=1.0e-10,
        )
        kernel = _comparison(
            source.kernel.numpy(), murano.kernel, rtol=1.0e-10, atol=1.0e-10
        )
        eigenvalues = _comparison(
            source.eigenvalues.numpy(),
            murano.eigenvalues,
            rtol=1.0e-9,
            atol=1.0e-10,
        )
        comparisons[str(dimensions)] = {
            "status": (
                "match"
                if kernel["status"] == "match"
                and eigenvalues["status"] == "match"
                and coordinate_geometry["status"] == "match"
                else "mismatch"
            ),
            "kernel": kernel,
            "eigenvalues": eigenvalues,
            "coordinate_frame": coordinate_frame,
            "coordinate_geometry": coordinate_geometry,
            "numerical_rank_match": source.numerical_rank == murano.numerical_rank,
        }
        projections[dimensions] = (source, murano)

    # Normalize the three-dimensional display rows and compare their retained geometry.
    source_surface, source_mask = api["surface_coordinates"](
        projections[3][0].coordinates
    )
    murano_surface, murano_mask = murano_surface_coordinates(
        projections[3][1].coordinates
    )
    mask_match = np.array_equal(source_mask.numpy(), murano_mask)
    surface_geometry = _comparison(
        source_surface.numpy() @ source_surface.numpy().T,
        murano_surface @ murano_surface.T,
        rtol=1.0e-8,
        atol=1.0e-10,
    )

    # Render only to a temporary directory; visual-reference review is a separate gate.
    color = CLASS_PALETTE[family]
    with TemporaryDirectory(prefix="fega-live-render-") as directory:
        root = Path(directory)
        paths = {
            "source_sphere": root / "source-sphere.png",
            "murano_sphere": root / "murano-sphere.png",
            "source_2d": root / "source-2d.png",
            "murano_2d": root / "murano-2d.png",
        }
        api["render_sphere"](
            paths["source_sphere"], source_surface.numpy(), color=color, dpi=96
        )
        render_sphere_surface(
            paths["murano_sphere"], murano_surface, color=color, dpi=96
        )
        api["render_projection_2d"](
            paths["source_2d"],
            projections[2][0].coordinates.numpy(),
            color=color,
            dpi=96,
        )
        murano_render_projection_2d(
            paths["murano_2d"],
            projections[2][1].coordinates,
            color=color,
            dpi=96,
        )
        renders_nonempty = all(
            path.is_file() and path.stat().st_size > 0 for path in paths.values()
        )

    matched = (
        all(item["status"] == "match" for item in comparisons.values())
        and mask_match
        and surface_geometry["status"] == "match"
        and renders_nonempty
    )
    return {
        "status": "match" if matched else "mismatch",
        "projections": comparisons,
        "surface_mask_match": mask_match,
        "surface_geometry": surface_geometry,
        "temporary_renders_nonempty": renders_nonempty,
        "pixel_equality_claimed": False,
    }


def _live_checks(
    api: dict[str, Any],
    config: FEGAConfig,
    source_pipeline: Any,
    metadata: dict[str, Any],
    source_labels: list[str | None],
    residual_rows: np.ndarray,
    residual_gram: np.ndarray,
    magnitudes: np.ndarray,
    source_provider: np.ndarray,
    source_score: Any,
    source_fit: Any,
    source_assignment: dict[str, Any],
    murano_selection: Any,
    murano_assignment: float | None,
) -> dict[str, Any]:
    """Compare geometry, executed stability, final reporting, and visualization.

    Args:
        api: Imported source FEGA authorities.
        config: Murano scientific defaults.
        source_pipeline: Parsed source run configuration.
        metadata: Shared feature and context identities.
        source_labels: Source-resolved group labels in retained-row order.
        residual_rows: Shared retained final-residual directions.
        residual_gram: Shared unembedding Gram matrix.
        magnitudes: Shared retained effect magnitudes.
        source_provider: Source-normalized vocabulary coordinates.
        source_score: Source vMF model-selection result.
        source_fit: Independently rerun selected source vMF fit.
        source_assignment: Executed source assignment-stability result.
        murano_selection: Murano vMF model-selection result.
        murano_assignment: Executed Murano assignment-stability value.

    Returns:
        A compact path-free report for the remaining native FEGA phases.
    """
    # Establish configuration and point-geometry agreement before stability executes.
    feature_id = int(metadata["feature_id"])
    configuration = _config_checks(source_pipeline, config)
    geometry, murano_geometry, source_geometry = _geometry_checks(
        api, residual_rows, residual_gram, config
    )
    source_geometry["feature_id"] = feature_id

    # Build source and Murano point records from their independently computed summaries.
    source_magnitudes = api["summarize_magnitudes"](
        np.asarray(magnitudes, dtype=np.float32).tolist()
    )
    loaded_contexts = len(metadata["context_indices"])
    retained_contexts = len(residual_rows)
    compute_record = {
        "feature_id": feature_id,
        "loaded_contexts": loaded_contexts,
        "skipped_near_zero": loaded_contexts - retained_contexts,
        "usable_effects": retained_contexts,
        **source_magnitudes,
    }
    source_metrics = api["metrics_from_fit"](
        torch.from_numpy(source_provider), source_fit, source_assignment
    )
    source_vmf = {
        "feature_id": feature_id,
        "metrics": source_metrics,
        "fit_status": source_score.fit_status,
        "model_selection": source_score.model_selection,
        "selected_fit": source_score.selected_fit,
        "assignment_stability": source_assignment,
    }
    source_point = api["_build_record"](
        str(feature_id),
        compute_record,
        source_geometry,
        source_vmf,
        None,
        tau_zero=config.eps,
        eps=config.eps,
    )
    source_selection = api["resolve_point_selection"](
        source_point, "paper", vmf_provenance_valid=True
    )
    source_identity = api["point_selection_identity"](source_selection)

    magnitudes_array = np.asarray(magnitudes, dtype=np.float32)
    mean_magnitude = float(np.mean(magnitudes_array)) if len(magnitudes_array) else 0.0
    selected = murano_selection.selected
    murano_mix = _mixture_metrics(
        residual_rows @ residual_gram @ residual_rows.T,
        selected.labels,
        selected.weights,
        selected.concentrations,
        murano_assignment,
    )
    murano_point_record = GeometryRecord(
        n_valid=murano_geometry.n_valid,
        zero_filter_frac=(loaded_contexts - retained_contexts)
        / max(loaded_contexts, 1),
        c_ray=murano_geometry.c_ray,
        b_axis=murano_geometry.b_axis,
        s_span={
            key: value
            for key, value in murano_geometry.s_span.items()
            if value is not None
        },
        u_span={
            key: value
            for key, value in murano_geometry.u_span.items()
            if value is not None
        },
        d_span={
            key: value
            for key, value in murano_geometry.d_span.items()
            if value is not None
        },
        r_span_ent=murano_geometry.r_span_ent,
        r_span_pr=murano_geometry.r_span_pr,
        m_cv=(
            float(np.std(magnitudes_array) / mean_magnitude)
            if mean_magnitude > 0.0
            else None
        ),
        selected_mode_count=int(murano_mix["selected_mode_count"]),
        delta_mix=murano_mix.get("delta_mix"),
        mode_mass_min=murano_mix.get("mode_mass_min"),
        min_mode_c_ray=murano_mix.get("min_mode_c_ray"),
        mode_kappa_min=murano_mix.get("mode_kappa_min"),
        assignment_stability=murano_assignment,
        e_res=murano_geometry.e_res,
        s_res={
            key: value
            for key, value in murano_geometry.s_res.items()
            if value is not None
        },
        r_ctr_pr=murano_geometry.r_ctr_pr,
    )
    murano_point = classify_geometry(murano_point_record)
    point_selection_match = (
        source_identity["family"] == murano_point.primary_label
        and source_identity["selected_k"] == murano_point.selected_k
        and source_identity["mode"] == murano_point.selection_mode
    )

    # Re-normalize the same stability input independently and prove row identity.
    raw_rows = torch.from_numpy(np.asarray(residual_rows))
    gram = torch.from_numpy(np.asarray(residual_gram))
    source_unit, source_counts, source_indices = api["final_resid_unit_rows"](
        raw_rows,
        gram,
        eps=float(source_pipeline.phases.geometry_metrics.c_ray.eps),
    )
    murano_unit, murano_counts, murano_indices = murano_final_resid_unit_rows(
        residual_rows, residual_gram, eps=config.eps
    )
    valid_indices_match = source_indices == murano_indices.tolist()
    counts_match = source_counts == murano_counts
    unit_rows = _comparison(source_unit.numpy(), murano_unit, rtol=1.0e-6, atol=1.0e-7)
    filtered_labels = [source_labels[index] for index in source_indices]

    # Execute the source's frozen schedule and Murano's matching selected-family path.
    base_seed = int(
        source_pipeline.phases.stability.seed
        if source_pipeline.phases.stability.seed is not None
        else source_pipeline.seed.global_
    )
    schedule = api["build_selected_family_schedule"](
        selection=source_selection,
        feature_id=feature_id,
        point_record_sha256="live-oracle",
        base_seed=base_seed,
        effect_space=source_pipeline.phases.stability.effect_space,
        n_rows=len(source_unit),
        group_labels=filtered_labels,
        stability_config=source_pipeline.phases.stability,
    )
    source_item = api["_ScheduledFeature"](
        schedule=schedule,
        point_record=source_point,
        raw_rows=raw_rows.index_select(0, torch.as_tensor(source_indices)),
        unit_rows=source_unit,
        gram=gram,
        valid_counts=source_counts,
    )
    source_execution = api["_execute_one"](
        source_item,
        source_pipeline,
        api["get_threshold_profile"]("paper"),
    ).record
    source_evidence = source_execution["selected_family_evidence"]
    murano_evidence = selected_family_stability(
        murano_point.primary_label,
        murano_point.selection_mode,
        murano_point.selected_k,
        source_unit.numpy(),
        residual_gram,
        murano_geometry,
        filtered_labels,
        config,
        feature_id,
    )
    protocols = _protocol_checks(source_evidence, murano_evidence)

    # Run both final reporting authorities and compare every published label or flag.
    source_reporting_record = {
        **source_point,
        "point_selection": source_identity,
        "selected_family_evidence": source_evidence,
    }
    source_final = api["classify_record"](source_reporting_record, "paper")
    murano_final = qualify_geometry(murano_point, murano_evidence)
    source_decision = source_final["selected_family_stability"]["decision"]
    final_fields = {
        "primary_label": source_final["primary_label"] == murano_final.primary_label,
        "selected_k": source_final["selected_k"] == murano_final.selected_k,
        "span_selected_k": source_final["span_selected_k"]
        == murano_final.span_selected_k,
        "residual_selected_k": source_final["residual_selected_k"]
        == murano_final.residual_selected_k,
        "stability_decision": source_decision == murano_final.stability_status,
        "label_confidence": source_final["label_confidence"] == murano_final.confidence,
        "secondary_flags": tuple(source_final["secondary_flags"])
        == murano_final.secondary_flags,
        "global_flags": tuple(source_final["global_flags"])
        == murano_final.global_flags,
    }
    reporting_match = all(final_fields.values())
    projection = _projection_checks(
        api, residual_rows, residual_gram, source_identity["family"]
    )
    matched = (
        configuration["status"] == "match"
        and geometry["status"] == "match"
        and point_selection_match
        and valid_indices_match
        and counts_match
        and unit_rows["status"] == "match"
        and protocols["status"] == "match"
        and reporting_match
        and projection["status"] == "match"
    )
    return {
        "status": "match" if matched else "mismatch",
        "configuration": configuration,
        "geometry": geometry,
        "point_selection": {
            "status": "match" if point_selection_match else "mismatch",
            "family": source_identity["family"],
            "selected_k": source_identity["selected_k"],
            "selection_mode": source_identity["mode"],
        },
        "stability_input": {
            "status": (
                "match"
                if valid_indices_match
                and counts_match
                and unit_rows["status"] == "match"
                else "mismatch"
            ),
            "valid_indices_match": valid_indices_match,
            "counts_match": counts_match,
            "unit_rows": unit_rows,
        },
        "executed_stability": protocols,
        "final_reporting": {
            "status": "match" if reporting_match else "mismatch",
            "fields": final_fields,
            "primary_label": source_final["primary_label"],
            "stability_decision": source_decision,
            "label_confidence": source_final["label_confidence"],
        },
        "visualization": projection,
    }


def _compare(
    coordinates: np.ndarray,
    gram_kernel: np.ndarray,
    metadata: dict[str, Any],
    source_root: Path,
    *,
    residual_rows: np.ndarray | None = None,
    residual_gram: np.ndarray | None = None,
    magnitudes: np.ndarray | None = None,
    source_config_path: Path | None = None,
) -> dict[str, Any]:
    """Run the complete current-source same-coordinate comparison."""
    # Normalize and fit independent copies of one ordered float32 matrix.
    api = _source_api(source_root)
    config = FEGAConfig(seed=42)
    feature_id = int(metadata["feature_id"])
    source_rows, counts = api["normalize_logit_deltas"](
        torch.from_numpy(coordinates.copy()), eps=config.eps
    )
    source_provider = api["normalized_numpy_rows"](source_rows)
    murano_provider = _validate_directions(coordinates.copy(), warn_large=False)
    provider = _comparison(source_provider, murano_provider, rtol=0.0, atol=0.0)
    if counts["n_valid"] != len(coordinates):
        raise ValueError("Current source discarded a cached coordinate row")

    # Select k independently, then rerun the selected candidate to retain posteriors.
    source_config = api["DirectionalMixtureFitConfig"](
        backend="dense_cpu",
        k_values=list(config.vmf_k_values),
        bic_tolerance=config.vmf_bic_tolerance,
        resample_fraction=config.vmf_resample_fraction,
        resample_rounds=config.vmf_resample_rounds,
        n_init=config.vmf_n_init,
        max_iter=config.vmf_max_iter,
    )
    source_seed = api["feature_fit_seed"](config.seed, feature_id)
    source_score = api["score_vmf_feature"](
        torch.from_numpy(coordinates.copy()),
        source_config,
        seed=source_seed,
        assignment_stability_enabled=False,
    )
    source_k = int(source_score.metrics["selected_mode_count"])
    source_fit = api["fit_vmf_mixture"](
        source_rows,
        k=source_k,
        n_init=config.vmf_n_init,
        max_iter=config.vmf_max_iter,
        seed=api["derived_vmf_seed"](source_seed, source_k, -1, "candidate_fit"),
        backend="dense_cpu",
    )
    murano_seed = feature_seed(config.seed, feature_id)
    murano_selection = select_vmf(
        coordinates.copy(),
        config.vmf_k_values,
        murano_seed,
        config.vmf_n_init,
        config.vmf_max_iter,
        config.vmf_bic_tolerance,
        n_jobs=1,
        warn_large=False,
    )
    murano_k = murano_selection.selected.n_components
    murano_fit = fit_vmf(
        coordinates.copy(),
        murano_k,
        _derived_seed(murano_seed, murano_k, -1, "candidate_fit"),
        config.vmf_n_init,
        config.vmf_max_iter,
    )
    permutation = _component_permutation(
        np.asarray(source_fit.labels), murano_fit.labels, source_k
    )
    partition_match = source_k == murano_k and permutation is not None
    if permutation is None:
        responsibilities = {"status": "mismatch"}
        kappas = {"status": "mismatch"}
    else:
        responsibilities = _comparison(
            murano_fit.responsibilities[list(permutation), :],
            source_fit.responsibilities,
            rtol=2.0e-12,
            atol=0.0,
        )
        kappas = _comparison(
            murano_fit.concentrations[list(permutation)],
            source_fit.kappas,
            rtol=3.0e-15,
            atol=0.0,
        )

    # Resolve groups through current source and compare every explicit schedule.
    source_labels = _group_labels(api, metadata)
    group_labels_match = source_labels == metadata["murano_labels"]
    schedule_metadata = {**metadata, "source_labels": source_labels}
    schedules = _schedule_checks(api, schedule_metadata, config)
    explicit_kernel = coordinates.astype(np.float64) @ coordinates.astype(np.float64).T
    logit_gram = _comparison(
        explicit_kernel,
        gram_kernel,
        rtol=1.0e-5,
        atol=1.0e-5,
    )

    # Execute expensive downstream phases only when the live call supplies their tensors.
    live_inputs = (residual_rows, residual_gram, magnitudes, source_config_path)
    if any(value is None for value in live_inputs) and not all(
        value is None for value in live_inputs
    ):
        raise ValueError("live downstream inputs must be supplied together")
    assignment: dict[str, Any] = {"status": "not_evaluated"}
    live: dict[str, Any] | None = None
    if all(value is not None for value in live_inputs):
        source_assignment = api["assignment_stability"](
            source_rows,
            np.asarray(source_fit.labels),
            source_k,
            source_config,
            seed=source_seed,
            fit_fn=api["default_fit_fn"],
        )
        murano_assignment = murano_assignment_stability(
            coordinates.copy(),
            murano_selection.selected,
            murano_seed,
            fraction=config.vmf_resample_fraction,
            rounds=config.vmf_resample_rounds,
            n_jobs=1,
            n_init=config.vmf_n_init,
            max_iter=config.vmf_max_iter,
        )
        murano_assignment_status = (
            "not_applicable"
            if murano_k <= 1
            else "available"
            if murano_assignment is not None
            else "unavailable"
        )
        source_assignment_value = source_assignment.get("value")
        assignment_value = (
            {"status": "match"}
            if source_assignment_value is None and murano_assignment is None
            else _comparison(
                np.asarray([source_assignment_value]),
                np.asarray([murano_assignment]),
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )
        assignment_match = (
            source_assignment["status"] == murano_assignment_status
            and assignment_value["status"] == "match"
        )
        assignment = {
            "status": "match" if assignment_match else "mismatch",
            "availability_match": source_assignment["status"]
            == murano_assignment_status,
            "value": assignment_value,
            "replicates": int(source_assignment["requested_count"]),
        }
        assert residual_rows is not None
        assert residual_gram is not None
        assert magnitudes is not None
        assert source_config_path is not None
        source_pipeline = api["FEGAPipelineConfig"].from_file(source_config_path)
        live = _live_checks(
            api,
            config,
            source_pipeline,
            metadata,
            source_labels,
            residual_rows,
            residual_gram,
            magnitudes,
            source_provider,
            source_score,
            source_fit,
            source_assignment,
            murano_selection,
            murano_assignment,
        )
    matched = (
        provider["status"] == "match"
        and source_k == murano_k
        and partition_match
        and responsibilities["status"] == "match"
        and kappas["status"] == "match"
        and group_labels_match
        and schedules["status"] == "match"
        and logit_gram["status"] == "match"
        and assignment["status"] in {"match", "not_evaluated"}
        and (live is None or live["status"] == "match")
    )
    report = {
        "status": "match" if matched else "mismatch",
        "claim": "current FEGA and Murano on one identical ordered coordinate matrix",
        "counts": {
            "rows": len(coordinates),
            "vocabulary_dimensions": int(coordinates.shape[1]),
        },
        "normalized_provider": provider,
        "selected_k_match": source_k == murano_k,
        "partition_match": partition_match,
        "responsibilities": responsibilities,
        "kappas": kappas,
        "assignment_stability": assignment,
        "group_labels_match": group_labels_match,
        "schedules": schedules,
        "explicit_logit_vs_gram_kernel": logit_gram,
        "not_claimed": [
            "historical cache regeneration",
            "model-side effect equality",
            "paper-dataset reproduction",
        ],
    }
    if live is not None:
        report["live_geometry_stability_reporting_visualization"] = live
    return report


def main() -> None:
    """Read transient inputs, compare them, and write one compact JSON section."""
    # Keep the helper CLI private and explicit so it cannot discover hidden artifacts.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coordinates", type=Path)
    parser.add_argument("gram_kernel", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    coordinates = np.load(args.coordinates, allow_pickle=False)
    gram_kernel = np.load(args.gram_kernel, allow_pickle=False)
    metadata = json.loads(args.metadata.read_text())
    report = _compare(coordinates, gram_kernel, metadata, args.source_root)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["status"] != "match":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
