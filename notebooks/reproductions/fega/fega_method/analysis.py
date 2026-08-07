"""Model-free FEGA analysis and visualization pipeline steps."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from . import keys
from .artifacts import (
    FEGAEffectStore,
    FEGAFeatureEffects,
    FEGAGeometryResult,
    FEGARenderCapture,
    FEGAReportingResult,
    FEGAStabilityResult,
    FEGAVMFResult,
    FEGAVisualizationResult,
)
from .config import FEGAConfig
from .geometry import GeometryMetrics, compute_geometry_metrics
from .reporting import (
    GeometryRecord,
    classify_geometry,
    qualify_geometry,
)
from .stability import selected_family_stability
from .visual_detail import (
    render_residual_view,
    render_subspace_plane,
    residual_point_colors,
)
from .visualization import (
    CLASS_PALETTE,
    project_directions,
    rank_family_candidates,
    render_projection_2d,
    render_sphere_surface,
    surface_coordinates,
)
from murano.results import Results
from murano.steps.base import Step


def _copy_render_value(value: Any) -> Any:
    """Copy one renderer input value into an immutable comparison payload."""
    # Freeze mutable renderer inputs so later render-side mutation cannot leak back.
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, list | tuple):
        return tuple(_copy_render_value(item) for item in value)
    return value


def _capture_render(
    renderer: str,
    coordinates: np.ndarray,
    **kwargs: Any,
) -> FEGARenderCapture:
    """Capture one exact local renderer invocation for later comparison."""
    # Store only copied scientific inputs from the actual local render boundary.
    point_colors = kwargs.get("point_colors")
    assignments = kwargs.get("mode_assignments")
    selected_k = kwargs.get("selected_k")
    return FEGARenderCapture(
        renderer=renderer,
        coordinates=np.asarray(coordinates, dtype=np.float64).copy(),
        kwargs={key: _copy_render_value(value) for key, value in kwargs.items()},
        point_colors=(
            None
            if point_colors is None
            else tuple(str(color) for color in point_colors)
        ),
        assignments=(
            None
            if assignments is None
            else tuple(int(assignment) for assignment in assignments)
        ),
        selected_k=None if selected_k is None else int(selected_k),
    )


class FEGAGeometryMetrics(Step):
    """Compute deterministic dense point geometry for every retained feature."""

    reads = [keys.EFFECTS]
    writes = [keys.GEOMETRY]
    read_types = {keys.EFFECTS: FEGAEffectStore}
    write_types = {keys.GEOMETRY: FEGAGeometryResult}

    def __init__(self, config: FEGAConfig) -> None:
        """Configure source metric dimensions and numerical tolerance."""
        # Keep the immutable scientific configuration used by every feature.
        self.config = config

    def __call__(self, results: Results) -> Results:
        """Compute one float64 geometry record per feature in sorted order."""
        # Analyze compact residual rows through their canonical logit-space Gram.
        effects = results[keys.EFFECTS]
        metrics = {
            feature: compute_geometry_metrics(
                effects.features[feature].directions.numpy(),
                effects.gram.numpy(),
                k_values=self.config.span_k_values,
                residual_k_values=self.config.residual_k_values,
                eps=self.config.eps,
            )
            for feature in effects.ordered_feature_ids
        }
        results[keys.GEOMETRY] = FEGAGeometryResult(metrics)
        return results


def _record(
    geometry: GeometryMetrics,
    feature: int,
    magnitudes: np.ndarray,
    retained_mask: tuple[bool, ...],
    vmf: FEGAVMFResult,
) -> GeometryRecord:
    """Combine the values needed to classify one feature's geometry."""
    # Consume retained source-style vMF reporting scalars instead of recomputing them here.
    selection = vmf.features.get(feature)
    mode_count = None if selection is None else selection.selected.n_components
    attempted = len(retained_mask)
    retained = sum(retained_mask)
    mean_magnitude = float(np.mean(magnitudes)) if len(magnitudes) else 0.0
    return GeometryRecord(
        n_valid=geometry.n_valid,
        zero_filter_frac=(attempted - retained) / max(attempted, 1),
        c_ray=geometry.c_ray,
        b_axis=geometry.b_axis,
        s_span={
            key: value for key, value in geometry.s_span.items() if value is not None
        },
        u_span={
            key: value for key, value in geometry.u_span.items() if value is not None
        },
        d_span={
            key: value for key, value in geometry.d_span.items() if value is not None
        },
        r_span_ent=geometry.r_span_ent,
        r_span_pr=geometry.r_span_pr,
        m_cv=(
            float(np.std(magnitudes) / mean_magnitude) if mean_magnitude > 0.0 else None
        ),
        selected_mode_count=None if mode_count is None else int(mode_count),
        delta_mix=None if selection is None else getattr(selection, "delta_mix", None),
        mode_mass_min=(
            None if selection is None else getattr(selection, "mode_mass_min", None)
        ),
        min_mode_c_ray=(
            None if selection is None else getattr(selection, "min_mode_c_ray", None)
        ),
        mode_kappa_min=(
            None if selection is None else getattr(selection, "mode_kappa_min", None)
        ),
        assignment_stability=vmf.assignment_stability.get(feature),
        e_res=geometry.e_res,
        s_res={
            key: value for key, value in geometry.s_res.items() if value is not None
        },
        r_ctr_pr=geometry.r_ctr_pr,
    )


class FEGAStability(Step):
    """Qualify each point-selected family without changing its family label."""

    reads = [keys.EFFECTS, keys.GEOMETRY, keys.VMF]
    writes = [keys.STABILITY]
    read_types = {
        keys.EFFECTS: FEGAEffectStore,
        keys.GEOMETRY: FEGAGeometryResult,
        keys.VMF: FEGAVMFResult,
    }
    write_types = {keys.STABILITY: FEGAStabilityResult}

    def __init__(self, config: FEGAConfig) -> None:
        """Configure deterministic stability schedules."""
        # Run the notebook's full stability schedule directly.
        self.config = config

    def __call__(self, results: Results) -> Results:
        """Run the stability protocol appropriate for each selected family."""
        # Analyze every feature in deterministic order.
        effects = results[keys.EFFECTS]
        geometry = results[keys.GEOMETRY]
        vmf = results[keys.VMF]
        records: dict[int, dict[str, Any]] = {}
        for feature in effects.ordered_feature_ids:
            feature_effects = effects.features[feature]
            metrics = geometry.features[feature]
            classification = classify_geometry(
                _record(
                    metrics,
                    feature,
                    feature_effects.magnitudes.numpy(),
                    feature_effects.retained_mask,
                    vmf,
                )
            )
            records[feature] = self._feature_record(
                feature,
                classification.primary_label,
                classification.selection_mode,
                classification.selected_k,
                feature_effects,
                effects.gram,
                metrics,
                vmf.assignment_stability.get(feature),
            )
        results[keys.STABILITY] = FEGAStabilityResult(records)
        return results

    def _feature_record(
        self,
        feature: int,
        family: str,
        selection_mode: str,
        selected_k: int | None,
        effects: FEGAFeatureEffects,
        gram: torch.Tensor,
        metrics: GeometryMetrics,
        assignment_stability: float | None,
    ) -> dict[str, Any]:
        """Compute the selected family's bounded stability evidence."""
        # Compute stability for the selected family using the FEGA rules.
        return selected_family_stability(
            family,
            selection_mode,
            selected_k,
            effects.directions.numpy(),
            gram.numpy(),
            metrics,
            [context.group_label for context in effects.contexts],
            self.config,
            feature,
            assignment_stability=assignment_stability,
        )


class FEGAGeometryReporting(Step):
    """Publish deterministic FEGA labels with selected-family qualification."""

    reads = [keys.EFFECTS, keys.GEOMETRY, keys.VMF, keys.STABILITY]
    writes = [keys.REPORTING]
    read_types = {
        keys.EFFECTS: FEGAEffectStore,
        keys.GEOMETRY: FEGAGeometryResult,
        keys.VMF: FEGAVMFResult,
        keys.STABILITY: FEGAStabilityResult,
    }
    write_types = {keys.REPORTING: FEGAReportingResult}

    def __call__(self, results: Results) -> Results:
        """Classify sorted features and attach, without replacing, stability status."""
        # Point classification remains authoritative; stability only qualifies it.
        effects = results[keys.EFFECTS]
        geometry = results[keys.GEOMETRY]
        vmf = results[keys.VMF]
        stability = results[keys.STABILITY]
        feature_ids = effects.ordered_feature_ids
        records = {}
        classifications = {}
        for feature in feature_ids:
            record = _record(
                geometry.features[feature],
                feature,
                effects.features[feature].magnitudes.numpy(),
                effects.features[feature].retained_mask,
                vmf,
            )
            records[feature] = record
            point = classify_geometry(record)
            evidence = stability.features[feature]
            classifications[feature] = qualify_geometry(point, evidence)
        results[keys.REPORTING] = FEGAReportingResult(
            records, classifications, feature_ids
        )
        return results


class FEGAVisualize(Step):
    """Render the approved sphere-surface and 2D candidate figures."""

    reads = [keys.EFFECTS, keys.GEOMETRY, keys.VMF, keys.REPORTING]
    writes = [keys.VISUALIZATION]
    read_types = {
        keys.EFFECTS: FEGAEffectStore,
        keys.GEOMETRY: FEGAGeometryResult,
        keys.VMF: FEGAVMFResult,
        keys.REPORTING: FEGAReportingResult,
    }
    write_types = {keys.VISUALIZATION: FEGAVisualizationResult}

    def __init__(
        self,
        output_dir: str | Path,
        *,
        top_k_per_family: int = 3,
        figures: Sequence[str] = ("sphere_surface", "projection_2d"),
        dpi: int = 300,
    ) -> None:
        """Configure output location, figure inventory, count, and resolution."""
        # Validate the user-facing visualization boundary without adding plot variants.
        requested = tuple(dict.fromkeys(str(figure) for figure in figures))
        allowed = {"sphere_surface", "projection_2d"}
        if set(requested) - allowed:
            raise ValueError(f"FEGA figures must be chosen from {sorted(allowed)}")
        if top_k_per_family < 0 or dpi < 1:
            raise ValueError("top_k_per_family must be nonnegative and dpi positive")
        self.output_dir = Path(output_dir)
        self.top_k_per_family = int(top_k_per_family)
        self.figures = requested
        self.dpi = int(dpi)

    def __call__(self, results: Results) -> Results:
        """Rank cached analysis artifacts and render the selected minimal figures."""
        # Build renderer inputs from the existing analysis results.
        effects = results[keys.EFFECTS]
        geometry = results[keys.GEOMETRY]
        vmf = results[keys.VMF]
        reporting = results[keys.REPORTING]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        records = [
            self._record(
                feature,
                effects.features[feature],
                geometry.features[feature],
                vmf,
                reporting,
            )
            for feature in reporting.feature_ids
        ]
        files: list[Path] = []
        skipped: dict[str, str] = {}
        captures: dict[int, dict[str, FEGARenderCapture]] = {}

        # Render one bounded candidate set per family from the same cached directions.
        ranked = rank_family_candidates(records, top_k=self.top_k_per_family)
        for family in CLASS_PALETTE:
            candidates = ranked.get(family, [])
            usable = [candidate for candidate in candidates if candidate["n_valid"] > 0]
            if not usable:
                skipped[family] = (
                    "no usable retained effect directions"
                    if candidates
                    else "no candidate in this family"
                )
                continue
            for rank, candidate in enumerate(usable, start=1):
                feature = int(candidate["feature_id"])
                feature_dir = self.output_dir / family / f"rank_{rank:02d}_f{feature}"
                feature_dir.mkdir(parents=True, exist_ok=True)
                feature_files, feature_captures = self._render_candidate(
                    feature_dir,
                    family,
                    candidate.get("selected_k"),
                    effects.features[feature],
                    vmf,
                    effects.gram,
                )
                files.extend(feature_files)
                captures[feature] = feature_captures
        results[keys.VISUALIZATION] = FEGAVisualizationResult(
            files=tuple(files), skipped=skipped, captures=captures
        )
        return results

    def _record(
        self,
        feature: int,
        feature_effects: FEGAFeatureEffects,
        metrics: GeometryMetrics,
        vmf: FEGAVMFResult,
        reporting: FEGAReportingResult,
    ) -> dict[str, Any]:
        """Flatten one typed feature into candidate-ranking fields."""
        # Preserve the exact point metrics while deriving magnitude summaries locally.
        magnitudes = feature_effects.magnitudes.numpy()
        mean = float(np.mean(magnitudes)) if len(magnitudes) else 0.0
        selection = vmf.features.get(feature)
        record: dict[str, Any] = {
            "feature_id": feature,
            "primary_label": reporting.features[feature].primary_label,
            "selected_k": reporting.features[feature].selected_k,
            "n_valid": metrics.n_valid,
            "m_median": float(np.median(magnitudes)) if len(magnitudes) else None,
            "m_cv": float(np.std(magnitudes) / mean) if mean > 0.0 else None,
            "r2": metrics.r2,
            "c_ray": metrics.c_ray,
            "r_span_pr": metrics.r_span_pr,
            "b_axis": metrics.b_axis,
            "e_res": metrics.e_res,
            "r_ctr_pr": metrics.r_ctr_pr,
            "u_span_2": metrics.u_span.get(2),
            "d_span_2": metrics.d_span.get(2),
            "selected_mode_count": (
                None if selection is None else selection.selected.n_components
            ),
            "delta_mix": None
            if selection is None
            else getattr(selection, "delta_mix", None),
            "mode_mass_min": (
                None if selection is None else getattr(selection, "mode_mass_min", None)
            ),
            "min_mode_c_ray": (
                None
                if selection is None
                else getattr(selection, "min_mode_c_ray", None)
            ),
            "mode_kappa_min": (
                None
                if selection is None
                else getattr(selection, "mode_kappa_min", None)
            ),
        }
        record.update({f"s_span_{k}": metrics.s_span.get(k) for k in (1, 2, 3, 4, 8)})
        record["assignment_stability"] = vmf.assignment_stability.get(feature)
        return record

    def _render_candidate(
        self,
        feature_dir: Path,
        family: str,
        selected_k: int | None,
        effects: FEGAFeatureEffects,
        vmf: FEGAVMFResult,
        gram: torch.Tensor,
    ) -> tuple[list[Path], dict[str, FEGARenderCapture]]:
        """Render the two approved views for one ranked feature."""
        # Project once from the exact D G D.T kernel and reuse its signed coordinates.
        projection = project_directions(
            effects.directions.numpy(), gram.numpy(), dimensions=3, centered=False
        )
        sphere = projection.coordinates.copy()
        if family == "axis_or_antipodal" and np.count_nonzero(
            sphere[:, 0] > 0
        ) > np.count_nonzero(sphere[:, 0] < 0):
            sphere[:, 0] *= -1.0
        plane = self._plane_coordinates(family, effects, gram, sphere)
        surface, keep = surface_coordinates(sphere)
        point_colors, assignments = self._point_colors(
            family, sphere, plane, vmf, effects.feature_id
        )
        surface_colors = (
            [
                color
                for color, retained in zip(point_colors, keep, strict=True)
                if retained
            ]
            if point_colors is not None
            else None
        )
        outputs: list[Path] = []
        captures: dict[str, FEGARenderCapture] = {}
        if "sphere_surface" in self.figures:
            path = feature_dir / "sphere_surface.png"
            sphere_kwargs = {
                "color": CLASS_PALETTE[family],
                "dpi": self.dpi,
                "point_colors": surface_colors,
            }
            captures["sphere_surface"] = _capture_render(
                "render_sphere_surface",
                surface,
                **sphere_kwargs,
            )
            render_sphere_surface(path, surface, **sphere_kwargs)
            outputs.append(path)
        if "projection_2d" in self.figures:
            path = feature_dir / "projection_2d.png"
            if family == "residual_lowD_k" and selected_k is not None:
                projection_kwargs = {
                    "selected_k": selected_k,
                    "color": CLASS_PALETTE[family],
                    "dpi": self.dpi,
                    "point_colors": point_colors
                    or [CLASS_PALETTE[family]] * len(plane),
                }
                captures["projection_2d"] = _capture_render(
                    "render_residual_view",
                    plane,
                    **projection_kwargs,
                )
                render_residual_view(path, plane, **projection_kwargs)
            elif family == "global_2D_directional_subspace":
                projection_kwargs = {
                    "color": CLASS_PALETTE[family],
                    "dpi": self.dpi,
                    "point_colors": point_colors,
                }
                captures["projection_2d"] = _capture_render(
                    "render_subspace_plane",
                    sphere,
                    **projection_kwargs,
                )
                render_subspace_plane(path, sphere, **projection_kwargs)
            else:
                projection_kwargs = {
                    "color": CLASS_PALETTE[family],
                    "dpi": self.dpi,
                    "point_colors": point_colors,
                    "axis_guide": family == "axis_or_antipodal",
                    "view_padding": 2.4 if family == "directed_ray" else 1.35,
                    "primary_axis_guide": family
                    not in {
                        "multi_mode_directional_geometry",
                        "oneD_diffuse",
                        "residual_lowD_k",
                        "unresolved_high_dimensional_or_diffuse",
                    },
                    "view_limit": (
                        1.35 if family == "multi_mode_directional_geometry" else None
                    ),
                    "mode_assignments": assignments,
                    "fitted_line": family == "oneD_diffuse",
                }
                captures["projection_2d"] = _capture_render(
                    "render_projection_2d",
                    plane,
                    **projection_kwargs,
                )
                render_projection_2d(path, plane, **projection_kwargs)
            outputs.append(path)
        return outputs, captures

    @staticmethod
    def _plane_coordinates(
        family: str,
        effects: FEGAFeatureEffects,
        gram: torch.Tensor,
        sphere: np.ndarray,
    ) -> np.ndarray:
        """Return the source family-specific two-dimensional display coordinates."""
        # Center residual/unresolved views and rigidly orient directed-ray variation.
        if family == "residual_lowD_k":
            return project_directions(
                effects.directions.numpy(), gram.numpy(), dimensions=4, centered=True
            ).coordinates
        plane = sphere[:, :2].copy()
        if family in {"directed_ray", "unresolved_high_dimensional_or_diffuse"}:
            plane -= plane.mean(axis=0)
        if family == "directed_ray" and len(plane):
            _, _, right = np.linalg.svd(plane, full_matrices=False)
            plane = plane @ right.T
            for column in range(plane.shape[1]):
                anchor = int(np.argmax(np.abs(plane[:, column])))
                if plane[anchor, column] < 0.0:
                    plane[:, column] *= -1.0
        return plane

    @staticmethod
    def _point_colors(
        family: str,
        coordinates: np.ndarray,
        display_coordinates: np.ndarray,
        vmf: FEGAVMFResult,
        feature: int,
    ) -> tuple[list[str] | None, list[int] | None]:
        """Return deterministic axis or mixture colors aligned to effect rows."""
        # Color only families whose membership is part of the visual explanation.
        if family == "axis_or_antipodal":
            colors = [
                "#4F6D8A" if value < 0 else "#C7444E" if value > 0 else "#333333"
                for value in coordinates[:, 0]
            ]
            return colors, None
        if family == "residual_lowD_k":
            return residual_point_colors(display_coordinates[:, 0].tolist()), None
        selection = vmf.features.get(feature)
        if family != "multi_mode_directional_geometry" or selection is None:
            return None, None
        assignments = [int(value) for value in selection.selected.labels.tolist()]
        mode_palette = ("#E83E8C", "#E67E22", "#2A9D8F", "#6C5CE7")
        modes = sorted(
            set(assignments),
            key=lambda mode: (
                float(coordinates[np.asarray(assignments) == mode, 0].mean()),
                float(coordinates[np.asarray(assignments) == mode, 1].mean()),
                mode,
            ),
        )
        color_by_mode = {mode: mode_palette[index] for index, mode in enumerate(modes)}
        return [color_by_mode[mode] for mode in assignments], assignments
