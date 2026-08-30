"""Pure FEGA geometry classification from precomputed point metrics."""

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping


TAU_R = {2: 1.6, 3: 2.3, 4: 3.0, 8: 5.0}
TAU_P = {2: 0.08, 3: 0.05, 4: 0.03, 8: 0.01}
TAU_R_CTR = {2: 1.5, 3: 2.2, 4: 2.9}
GLOBAL_FLAG_ORDER = (
    "long_tail_spectrum",
    "magnitude_unstable",
    "sample_size_unstable",
    "leave_out_unstable",
    "exploratory_low_n",
)
TERMINAL_LABELS = {
    "insufficient_effect_evidence",
    "geometry_metrics_unavailable",
    "undefined_geometry",
}


@dataclass(frozen=True)
class GeometryRecord:
    """Source-equivalent metrics needed to classify one FEGA point."""

    n_valid: int
    zero_filter_frac: float
    c_ray: float | None = None
    b_axis: float | None = None
    s_span: Mapping[int, float] = field(default_factory=dict)
    u_span: Mapping[int, float] = field(default_factory=dict)
    d_span: Mapping[int, float] = field(default_factory=dict)
    r_span_ent: float | None = None
    r_span_pr: float | None = None
    m_cv: float | None = None
    selected_mode_count: int | None = None
    delta_mix: float | None = None
    mode_mass_min: float | None = None
    min_mode_c_ray: float | None = None
    mode_kappa_min: float | None = None
    assignment_stability: float | None = None
    e_res: float | None = None
    s_res: Mapping[int, float] = field(default_factory=dict)
    r_ctr_pr: float | None = None


@dataclass(frozen=True)
class GeometryClassification:
    """Primary label, candidate audit data, and selected dimensions."""

    primary_label: str
    candidate_labels: tuple[str, ...]
    flags: Mapping[str, bool]
    selected_k: int | None
    span_selected_k: int | None
    residual_selected_k: int | None
    selection_mode: str = "strict"
    stability_status: str | None = None
    confidence: str | None = None
    secondary_flags: tuple[str, ...] = ()
    global_flags: tuple[str, ...] = ()


def classify_geometry(record: GeometryRecord) -> GeometryClassification:
    """Classify a point with the paper's inclusive and strict boundaries."""
    # Gate every scientific candidate on the shared data-quality requirements.
    eligible = record.n_valid >= 8 and record.zero_filter_frac <= 0.30
    span_one = record.s_span.get(1)
    directed = eligible and _at_least(record.c_ray, 0.80) and _at_least(span_one, 0.80)
    below_ray = record.c_ray is not None and record.c_ray < 0.80
    axis = (
        eligible
        and _at_least(span_one, 0.80)
        and below_ray
        and _at_least(record.b_axis, 0.15)
    )
    one_d = (
        eligible
        and _at_least(span_one, 0.80)
        and below_ray
        and not _at_least(record.b_axis, 0.15)
    )

    # Accept multimode geometry only when every mode-quality check passes.
    multimode = (
        eligible
        and record.selected_mode_count is not None
        and record.selected_mode_count > 1
        and _at_least(record.delta_mix, 0.10)
        and _at_least(record.mode_mass_min, 0.10)
        and _at_least(record.min_mode_c_ray, 0.70)
        and record.mode_kappa_min is not None
        and math.isfinite(record.mode_kappa_min)
        and _at_least(record.assignment_stability, 0.80)
    )

    # Select the smallest global span dimension satisfying every paper boundary.
    span_selected_k = (
        next(
            (
                k
                for k in TAU_R
                if _mapping_at_least(record.s_span, k, 0.90)
                and _at_least(record.r_span_pr, TAU_R[k])
                and _mapping_at_least(record.u_span, k, TAU_P[k])
                and _mapping_at_most(record.d_span, k, 0.60)
            ),
            None,
        )
        if eligible
        else None
    )
    span_label = (
        "global_2D_directional_subspace"
        if span_selected_k == 2
        else "global_kD_directional_subspace"
    )
    span = span_selected_k is not None

    # Select the smallest centered-residual dimension satisfying its boundaries.
    residual_selected_k = (
        next(
            (
                k
                for k in TAU_R_CTR
                if _at_least(record.e_res, 0.10)
                and _mapping_at_least(record.s_res, k, 0.80)
                and _at_least(record.r_ctr_pr, TAU_R_CTR[k])
            ),
            None,
        )
        if eligible
        else None
    )
    residual = residual_selected_k is not None

    # Preserve the source fallback anchors when a complete strict gate does not pass.
    span_anchor_k = (
        next(
            (k for k in (2, 3, 4, 8) if _mapping_at_least(record.s_span, k, 0.90)),
            None,
        )
        if eligible
        else None
    )
    residual_anchor_ks = (
        tuple(
            k
            for k in (1, 2, 3, 4)
            if _at_least(record.e_res, 0.10)
            and _mapping_at_least(record.s_res, k, 0.80)
        )
        if eligible
        else ()
    )
    residual_anchor_k = residual_anchor_ks[0] if residual_anchor_ks else None
    multimode_anchor = (
        eligible
        and record.selected_mode_count is not None
        and record.selected_mode_count > 1
        and _at_least(record.delta_mix, 0.10)
        and any(
            (
                _at_least(record.mode_mass_min, 0.10),
                _at_least(record.min_mode_c_ray, 0.70),
                _at_least(record.assignment_stability, 0.80),
            )
        )
    )

    # Distinguish positive diffuse evidence from a record with no supported geometry.
    family_anchor = any(
        (
            directed,
            axis,
            one_d,
            multimode_anchor,
            span_anchor_k is not None,
            residual_anchor_k is not None,
        )
    )
    finite_spans = [record.s_span[k] for k in (2, 3, 4, 8) if k in record.s_span]
    positive_high_dimensional = (
        eligible
        and not family_anchor
        and span_one is not None
        and span_one < 0.80
        and record.e_res is not None
        and record.e_res < 0.10
        and bool(finite_spans)
        and all(value < 0.90 for value in finite_spans)
        and record.selected_mode_count is not None
        and record.delta_mix is not None
        and (record.selected_mode_count <= 1 or record.delta_mix < 0.10)
    )
    long_tail = (
        eligible
        and not family_anchor
        and record.r_span_ent is not None
        and record.r_span_pr is not None
        and record.r_span_ent / (record.r_span_pr + 1.0e-12) >= 1.50
    )
    unresolved = positive_high_dimensional or long_tail

    # Keep candidate/fallback reporting stable while choosing by strict priority.
    flags = {
        "eligible": eligible,
        "directed_ray": directed,
        "axis_or_antipodal": axis,
        "oneD_diffuse": one_d,
        "multi_mode_directional_geometry": multimode or multimode_anchor,
        "global_2D_directional_subspace": (span_selected_k or span_anchor_k) == 2,
        "global_kD_directional_subspace": (span_selected_k or span_anchor_k)
        in (3, 4, 8),
        "residual_lowD_k": residual or residual_anchor_k is not None,
        "unresolved_high_dimensional_or_diffuse": unresolved,
    }
    candidate_labels = tuple(
        label for label, passed in flags.items() if label != "eligible" and passed
    )

    selected_k = None
    selection_mode = "strict"
    if directed:
        primary_label = "directed_ray"
    elif axis:
        primary_label = "axis_or_antipodal"
    elif multimode:
        primary_label = "multi_mode_directional_geometry"
    elif span:
        primary_label = span_label
        selected_k = span_selected_k
    elif residual:
        primary_label = "residual_lowD_k"
        selected_k = residual_selected_k
    elif one_d:
        primary_label = "oneD_diffuse"
        selection_mode = "fallback"
    elif span_anchor_k is not None:
        primary_label = (
            "global_2D_directional_subspace"
            if span_anchor_k == 2
            else "global_kD_directional_subspace"
        )
        selected_k = span_anchor_k
        selection_mode = "fallback"
    elif residual_anchor_k is not None:
        primary_label = "residual_lowD_k"
        selected_k = residual_anchor_k
        selection_mode = "fallback"
    elif unresolved:
        primary_label = "unresolved_high_dimensional_or_diffuse"
        selection_mode = "fallback"
    elif not eligible:
        primary_label = "insufficient_effect_evidence"
        selection_mode = "terminal"
    else:
        primary_label = "undefined_geometry"
        selection_mode = "terminal"

    # Publish the point-level diagnostics for which this reduced record has evidence.
    secondary_flags: list[str] = []
    if (
        directed
        and record.r_span_pr is not None
        and 1.45 <= record.r_span_pr < TAU_R[2]
    ):
        secondary_flags.append("ray_span_boundary")
    if one_d:
        secondary_flags.append("oneD_not_ray_not_axis")
        secondary_flags.append(
            "b_axis_missing"
            if record.b_axis is None or not math.isfinite(record.b_axis)
            else "b_axis_low"
        )
    if multimode_anchor:
        multimode_blocked = False
        for value, threshold, missing_flag, failed_flag in (
            (record.mode_mass_min, 0.10, "mode_mass_missing", "mode_mass_failed"),
            (record.min_mode_c_ray, 0.70, "mode_c_ray_missing", "mode_c_ray_failed"),
            (
                record.assignment_stability,
                0.80,
                "assignment_stability_missing",
                "assignment_stability_failed",
            ),
        ):
            if value is None or not math.isfinite(value):
                secondary_flags.append(missing_flag)
                multimode_blocked = True
            elif value < threshold:
                secondary_flags.append(failed_flag)
                multimode_blocked = True
        if multimode_blocked:
            secondary_flags.append("multimode_candidate_blocked")
    if span_anchor_k is not None:
        secondary_flags.append("span_selected_k")
        strict_span = (
            _at_least(record.s_span.get(span_anchor_k), 0.90)
            and _at_least(record.r_span_pr, TAU_R[span_anchor_k])
            and _at_least(record.u_span.get(span_anchor_k), TAU_P[span_anchor_k])
            and _mapping_at_most(record.d_span, span_anchor_k, 0.60)
        )
        if directed or not strict_span:
            secondary_flags.append("lowD_candidate_blocked")
    if residual_anchor_k is not None:
        secondary_flags.append("residual_selected_k")
    if directed and any(k >= 2 for k in residual_anchor_ks):
        secondary_flags.append("directed_ray_with_lowD_residual")
    if positive_high_dimensional:
        secondary_flags.append("positive_highD_evidence")

    # Global flags describe the same point independently of its selected family.
    global_flags: list[str] = []
    if (
        primary_label not in TERMINAL_LABELS
        and record.r_span_ent is not None
        and record.r_span_pr is not None
        and record.r_span_ent / (record.r_span_pr + 1.0e-12) >= 1.50
    ):
        global_flags.append("long_tail_spectrum")
    if _at_least(record.m_cv, 1.00):
        global_flags.append("magnitude_unstable")
    secondary_flags.extend(global_flags)

    return GeometryClassification(
        primary_label=primary_label,
        candidate_labels=candidate_labels,
        flags=flags,
        selected_k=selected_k,
        span_selected_k=selected_k if primary_label.startswith("global_") else None,
        residual_selected_k=(
            selected_k if primary_label == "residual_lowD_k" else None
        ),
        selection_mode=selection_mode,
        secondary_flags=tuple(sorted(set(secondary_flags))),
        global_flags=tuple(flag for flag in GLOBAL_FLAG_ORDER if flag in global_flags),
    )


def qualify_geometry(
    classification: GeometryClassification, evidence: Mapping[str, Any]
) -> GeometryClassification:
    """Attach selected-family stability confidence and flags to a point label.

    Args:
        classification: Point-selected geometry result whose family stays fixed.
        evidence: Selected-family stability result for the same feature.

    Returns:
        The final reporting result with stability status, confidence, and flags.
    """
    # Translate the stability decision into confidence and diagnostic flags.
    decision = str(evidence.get("decision", "unavailable"))
    if decision == "stable":
        confidence = (
            "accepted"
            if evidence.get("assignment_stability") == "reused"
            else (
                "exploratory"
                if isinstance(evidence.get("protocol"), Mapping)
                and evidence["protocol"].get("status") == "exploratory"
                else "accepted"
            )
        )
    elif decision == "not_evaluated":
        confidence = {
            "fallback": "candidate",
            "terminal": (
                "insufficient"
                if classification.primary_label == "insufficient_effect_evidence"
                else "undefined"
            ),
        }.get(classification.selection_mode)
    else:
        confidence = "unstable" if decision == "unstable" else None

    # Merge only reporting-owned qualifiers; stability never changes the family.
    secondary_flags, global_flags = _qualified_flags(
        classification.secondary_flags,
        classification.global_flags,
        evidence,
    )
    return replace(
        classification,
        stability_status=decision,
        confidence=confidence,
        secondary_flags=secondary_flags,
        global_flags=global_flags,
    )


def _qualified_flags(
    point_flags: tuple[str, ...],
    global_flags: tuple[str, ...],
    evidence: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Merge point diagnostics with selected-family and global flags."""
    # Keep selected-family protocol status separate from record-level global flags.
    secondary = set(point_flags)
    globals_set = set(global_flags)
    decision = str(evidence.get("decision", "unavailable"))
    protocol_records = [
        evidence.get(name)
        for name in ("scalar_ci", "leave_out", "sample_size", "principal_angle")
    ]
    groups = evidence.get("groups")
    unavailable = any(
        isinstance(record, Mapping)
        and (
            record.get("status") == "unavailable"
            or int(record.get("unavailable_count", 0) or 0) > 0
        )
        for record in protocol_records
    ) or (
        isinstance(groups, Mapping)
        and groups.get("status") == "group_sampling_unavailable"
    )
    if decision == "unstable":
        secondary.add("selected_family_unstable")
        if unavailable:
            secondary.add("selected_family_evidence_unavailable")
    elif decision == "unavailable":
        secondary.add("selected_family_evidence_unavailable")
    elif decision == "not_evaluated":
        secondary.add("selected_family_not_evaluated")

    secondary.update(globals_set)
    ordered_globals = tuple(flag for flag in GLOBAL_FLAG_ORDER if flag in globals_set)
    return tuple(sorted(secondary)), ordered_globals


def _at_least(value: float | None, threshold: float) -> bool:
    """Return whether an optional scalar meets an inclusive lower bound."""
    # Missing evidence cannot satisfy a classification boundary.
    return value is not None and value >= threshold


def _mapping_at_least(values: Mapping[int, float], key: int, threshold: float) -> bool:
    """Return whether a keyed metric meets an inclusive lower bound."""
    # Delegate optional-value handling to the scalar boundary helper.
    return _at_least(values.get(key), threshold)


def _mapping_at_most(values: Mapping[int, float], key: int, threshold: float) -> bool:
    """Return whether a keyed metric meets an inclusive upper bound."""
    # Missing evidence cannot satisfy a classification boundary.
    value = values.get(key)
    return value is not None and value <= threshold
