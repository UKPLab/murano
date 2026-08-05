"""Model-free FEGA analysis and visualization pipeline steps."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from murano import keys
from murano.fega.artifacts import (
    FEGAEffectStore,
    FEGAFeatureEffects,
    FEGAGeometryResult,
    FEGAReportingResult,
    FEGAStabilityResult,
    FEGAVMFResult,
    FEGAVisualizationResult,
)
from murano.fega.checkpoints import (
    effect_resume_metadata,
    load_phase_checkpoint,
    phase_checkpoint_path,
    save_phase_checkpoint,
)
from murano.fega.config import FEGAConfig
from murano.fega.geometry import GeometryMetrics, compute_geometry_metrics
from murano.fega.reporting import (
    GeometryRecord,
    classify_geometry,
    qualify_geometry,
)
from murano.fega.stability import selected_family_stability
from murano.fega.visual_detail import (
    render_residual_view,
    render_subspace_plane,
    residual_point_colors,
)
from murano.fega.visualization import (
    CLASS_PALETTE,
    atlas_coordinates,
    project_directions,
    rank_family_candidates,
    render_atlas,
    render_projection_2d,
    render_sphere_surface,
    surface_coordinates,
)
from murano.results import Results
from murano.steps.base import Step


def _matching_analysis_id(*artifacts: Any) -> str:
    """Return the shared nonempty effect-analysis identity for phase inputs."""
    # Reject checkpoint mixtures before any downstream result is computed or rendered.
    identities = tuple(getattr(artifact, "analysis_id", "") for artifact in artifacts)
    if (
        not identities
        or not identities[0]
        or any(identity != identities[0] for identity in identities[1:])
    ):
        raise ValueError("FEGA phase inputs come from different effect analyses")
    return identities[0]


class FEGAGeometryMetrics(Step):
    """Compute deterministic dense point geometry for every retained feature."""

    reads = [keys.FEGA_EFFECTS]
    writes = [keys.FEGA_GEOMETRY]
    read_types = {keys.FEGA_EFFECTS: FEGAEffectStore}
    write_types = {keys.FEGA_GEOMETRY: FEGAGeometryResult}

    def __init__(self, config: FEGAConfig) -> None:
        """Configure source metric dimensions and numerical tolerance."""
        # Keep the immutable scientific config for exact checkpoint reproduction.
        self.config = config

    def __call__(self, results: Results) -> Results:
        """Compute one float64 geometry record per feature in sorted order."""
        # Analyze compact residual rows through their canonical logit-space Gram.
        effects = results[keys.FEGA_EFFECTS]
        analysis_id = _matching_analysis_id(effects)
        metrics = {
            feature: compute_geometry_metrics(
                effects.features[feature].directions.numpy(),
                effects.gram.numpy(),
                k_values=self.config.span_k_values,
                residual_k_values=self.config.residual_k_values,
                eps=self.config.eps,
            )
            for feature in sorted(effects.features)
        }
        results[keys.FEGA_GEOMETRY] = FEGAGeometryResult(metrics, analysis_id)
        return results


def _vmf_metrics(
    geometry: GeometryMetrics,
    effects: FEGAFeatureEffects,
    vmf: FEGAVMFResult,
    gram: torch.Tensor,
) -> dict[str, float | int | None]:
    """Derive source mixture-reporting scalars from the selected immutable fit."""
    # Use the Gram-unit kernel so mode rays equal explicit vocabulary-space rays.
    selection = vmf.features.get(effects.feature_id)
    if selection is None:
        return {}
    fit = selection.selected
    rows = effects.directions.to(torch.float64)
    labels = np.asarray(fit.labels, dtype=np.int64)
    kernel = (rows @ gram.to(torch.float64) @ rows.T).numpy()
    count = len(rows)
    global_c_ray = float((kernel.sum() - np.trace(kernel)) / (count * (count - 1)))
    rays: list[float] = []
    for mode in range(fit.n_components):
        indices = np.flatnonzero(labels == mode)
        if len(indices) < 2:
            return {
                "selected_mode_count": fit.n_components,
                "mode_mass_min": float(np.min(fit.weights)),
                "mode_kappa_min": float(np.min(fit.concentrations)),
            }
        block = kernel[np.ix_(indices, indices)]
        rays.append(
            float((block.sum() - np.trace(block)) / (len(indices) * (len(indices) - 1)))
        )
    return {
        "selected_mode_count": fit.n_components,
        "delta_mix": float(np.dot(fit.weights, rays) - global_c_ray),
        "mode_mass_min": float(np.min(fit.weights)),
        "min_mode_c_ray": min(rays),
        "mode_kappa_min": float(np.min(fit.concentrations)),
    }


def _record(
    geometry: GeometryMetrics,
    effects: FEGAFeatureEffects,
    vmf: FEGAVMFResult,
    gram: torch.Tensor,
) -> GeometryRecord:
    """Combine point geometry and mixture evidence into the reporting contract."""
    # Preserve missing mixture evidence rather than manufacturing fallback values.
    mix = _vmf_metrics(geometry, effects, vmf, gram)
    mode_count = mix.get("selected_mode_count")
    attempted = len(effects.retained_mask)
    retained = sum(effects.retained_mask)
    magnitudes = effects.magnitudes.numpy()
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
        selected_mode_count=(
            int(mode_count) if isinstance(mode_count, int | np.integer) else None
        ),
        delta_mix=mix.get("delta_mix"),
        mode_mass_min=mix.get("mode_mass_min"),
        min_mode_c_ray=mix.get("min_mode_c_ray"),
        mode_kappa_min=mix.get("mode_kappa_min"),
        assignment_stability=vmf.assignment_stability.get(effects.feature_id),
        e_res=geometry.e_res,
        s_res={
            key: value for key, value in geometry.s_res.items() if value is not None
        },
        r_ctr_pr=geometry.r_ctr_pr,
    )


class FEGAStability(Step):
    """Qualify each point-selected family without changing its family label."""

    reads = [keys.FEGA_EFFECTS, keys.FEGA_GEOMETRY, keys.FEGA_VMF]
    writes = [keys.FEGA_STABILITY]
    read_types = {
        keys.FEGA_EFFECTS: FEGAEffectStore,
        keys.FEGA_GEOMETRY: FEGAGeometryResult,
        keys.FEGA_VMF: FEGAVMFResult,
    }
    write_types = {keys.FEGA_STABILITY: FEGAStabilityResult}

    def __init__(
        self, config: FEGAConfig, *, checkpoint_dir: str | Path | None = None
    ) -> None:
        """Configure deterministic stability schedules and partial resume."""
        # Keep optional feature-level resume beside its expensive resampling phase.
        self.config = config
        self.checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)

    def __call__(self, results: Results) -> Results:
        """Run the stability protocol appropriate for each selected family."""
        # Restore only a checkpoint produced with the identical scientific config.
        effects = results[keys.FEGA_EFFECTS]
        geometry = results[keys.FEGA_GEOMETRY]
        vmf = results[keys.FEGA_VMF]
        analysis_id = _matching_analysis_id(effects, geometry, vmf)
        records: dict[int, dict[str, Any]] = {}
        metadata = effect_resume_metadata(effects, self.config)
        if (
            self.checkpoint_dir is not None
            and phase_checkpoint_path(self.checkpoint_dir, "stability").exists()
        ):
            prior, prior_metadata = load_phase_checkpoint(
                self.checkpoint_dir, "stability"
            )
            if isinstance(prior, FEGAStabilityResult) and prior_metadata == metadata:
                records.update(prior.features)
        for feature in sorted(effects.features):
            if feature in records:
                continue
            feature_effects = effects.features[feature]
            metrics = geometry.features[feature]
            classification = classify_geometry(
                _record(metrics, feature_effects, vmf, effects.gram)
            )
            records[feature] = self._feature_record(
                feature,
                classification.primary_label,
                classification.selection_mode,
                classification.selected_k,
                feature_effects,
                effects.gram,
                metrics,
            )
            if self.checkpoint_dir is not None:
                save_phase_checkpoint(
                    self.checkpoint_dir,
                    "stability",
                    FEGAStabilityResult(dict(records), analysis_id),
                    metadata,
                )
        results[keys.FEGA_STABILITY] = FEGAStabilityResult(records, analysis_id)
        if self.checkpoint_dir is not None:
            save_phase_checkpoint(
                self.checkpoint_dir,
                "stability",
                results[keys.FEGA_STABILITY],
                metadata,
            )
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
    ) -> dict[str, Any]:
        """Compute the selected family's bounded stability evidence."""
        # Delegate the frozen family to the exact source protocol inventory.
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
        )


class FEGAGeometryReporting(Step):
    """Publish deterministic FEGA labels with selected-family qualification."""

    reads = [keys.FEGA_EFFECTS, keys.FEGA_GEOMETRY, keys.FEGA_VMF, keys.FEGA_STABILITY]
    writes = [keys.FEGA_REPORTING]
    read_types = {
        keys.FEGA_EFFECTS: FEGAEffectStore,
        keys.FEGA_GEOMETRY: FEGAGeometryResult,
        keys.FEGA_VMF: FEGAVMFResult,
        keys.FEGA_STABILITY: FEGAStabilityResult,
    }
    write_types = {keys.FEGA_REPORTING: FEGAReportingResult}

    def __call__(self, results: Results) -> Results:
        """Classify sorted features and attach, without replacing, stability status."""
        # Point classification remains authoritative; stability only qualifies it.
        effects = results[keys.FEGA_EFFECTS]
        geometry = results[keys.FEGA_GEOMETRY]
        vmf = results[keys.FEGA_VMF]
        stability = results[keys.FEGA_STABILITY]
        analysis_id = _matching_analysis_id(effects, geometry, vmf, stability)
        if vmf.unembedding_fingerprint != effects.unembedding_fingerprint:
            raise ValueError("FEGA reporting inputs use different unembeddings")
        feature_ids = tuple(sorted(effects.features))
        classifications = {}
        for feature in feature_ids:
            point = classify_geometry(
                _record(
                    geometry.features[feature],
                    effects.features[feature],
                    vmf,
                    effects.gram,
                )
            )
            evidence = stability.features[feature]
            classifications[feature] = qualify_geometry(point, evidence)
        results[keys.FEGA_REPORTING] = FEGAReportingResult(
            classifications, feature_ids, analysis_id
        )
        return results


class FEGAVisualize(Step):
    """Render the approved atlas, sphere-surface, and 2D candidate figures."""

    reads = [keys.FEGA_EFFECTS, keys.FEGA_GEOMETRY, keys.FEGA_VMF, keys.FEGA_REPORTING]
    writes = [keys.FEGA_VISUALIZATION]
    read_types = {
        keys.FEGA_EFFECTS: FEGAEffectStore,
        keys.FEGA_GEOMETRY: FEGAGeometryResult,
        keys.FEGA_VMF: FEGAVMFResult,
        keys.FEGA_REPORTING: FEGAReportingResult,
    }
    write_types = {keys.FEGA_VISUALIZATION: FEGAVisualizationResult}

    def __init__(
        self,
        output_dir: str | Path,
        *,
        top_k_per_family: int = 3,
        figures: Sequence[str] = ("atlas", "sphere_surface", "projection_2d"),
        dpi: int = 300,
    ) -> None:
        """Configure output location, figure inventory, count, and resolution."""
        # Validate the user-facing visualization boundary without adding plot variants.
        requested = tuple(dict.fromkeys(str(figure) for figure in figures))
        allowed = {"atlas", "sphere_surface", "projection_2d"}
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
        # Build flat source-shaped records without loading a model or recomputing labels.
        effects = results[keys.FEGA_EFFECTS]
        geometry = results[keys.FEGA_GEOMETRY]
        vmf = results[keys.FEGA_VMF]
        reporting = results[keys.FEGA_REPORTING]
        analysis_id = _matching_analysis_id(effects, geometry, vmf, reporting)
        if vmf.unembedding_fingerprint != effects.unembedding_fingerprint:
            raise ValueError("FEGA visualization inputs use different unembeddings")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        records = [
            self._record(
                feature,
                effects.features[feature],
                geometry.features[feature],
                vmf,
                reporting,
                effects.gram,
            )
            for feature in reporting.feature_ids
        ]
        files: list[Path] = []
        skipped: dict[str, str] = {}

        # Render the deterministic PCA option of the source atlas preprocessing.
        if "atlas" in self.figures:
            atlas_path = self.output_dir / "geometry_atlas.png"
            render_atlas(atlas_path, records, atlas_coordinates(records), dpi=self.dpi)
            files.append(atlas_path)

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
                files.extend(
                    self._render_candidate(
                        feature_dir,
                        family,
                        candidate.get("selected_k"),
                        effects.features[feature],
                        vmf,
                        effects.gram,
                    )
                )
        results[keys.FEGA_VISUALIZATION] = FEGAVisualizationResult(
            files=tuple(files), skipped=skipped, analysis_id=analysis_id
        )
        return results

    def _record(
        self,
        feature: int,
        feature_effects: FEGAFeatureEffects,
        metrics: GeometryMetrics,
        vmf: FEGAVMFResult,
        reporting: FEGAReportingResult,
        gram: torch.Tensor,
    ) -> dict[str, Any]:
        """Flatten one typed feature into source atlas and ranking fields."""
        # Preserve the exact point metrics while deriving magnitude summaries locally.
        magnitudes = feature_effects.magnitudes.numpy()
        mean = float(np.mean(magnitudes)) if len(magnitudes) else 0.0
        mix = _vmf_metrics(metrics, feature_effects, vmf, gram)
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
        }
        record.update({f"s_span_{k}": metrics.s_span.get(k) for k in (1, 2, 3, 4, 8)})
        record.update(mix)
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
    ) -> list[Path]:
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
        if "sphere_surface" in self.figures:
            path = feature_dir / "sphere_surface.png"
            render_sphere_surface(
                path,
                surface,
                color=CLASS_PALETTE[family],
                dpi=self.dpi,
                point_colors=surface_colors,
            )
            outputs.append(path)
        if "projection_2d" in self.figures:
            path = feature_dir / "projection_2d.png"
            if family == "residual_lowD_k" and selected_k is not None:
                render_residual_view(
                    path,
                    plane,
                    selected_k=selected_k,
                    color=CLASS_PALETTE[family],
                    dpi=self.dpi,
                    point_colors=point_colors or [CLASS_PALETTE[family]] * len(plane),
                )
            elif family == "global_2D_directional_subspace":
                render_subspace_plane(
                    path,
                    sphere,
                    color=CLASS_PALETTE[family],
                    dpi=self.dpi,
                    point_colors=point_colors,
                )
            else:
                render_projection_2d(
                    path,
                    plane,
                    color=CLASS_PALETTE[family],
                    dpi=self.dpi,
                    point_colors=point_colors,
                    axis_guide=family == "axis_or_antipodal",
                    view_padding=2.4 if family == "directed_ray" else 1.35,
                    primary_axis_guide=family
                    not in {
                        "multi_mode_directional_geometry",
                        "oneD_diffuse",
                        "residual_lowD_k",
                        "unresolved_high_dimensional_or_diffuse",
                    },
                    view_limit=(
                        1.35 if family == "multi_mode_directional_geometry" else None
                    ),
                    mode_assignments=assignments,
                    fitted_line=family == "oneD_diffuse",
                )
            outputs.append(path)
        return outputs

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
