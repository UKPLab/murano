"""Projection, ranking, and minimal plotting utilities for FEGA outputs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from murano.fega.visual_detail import load_pyplot


CLASS_PALETTE: dict[str, str] = {
    "directed_ray": "#1f77b4",
    "axis_or_antipodal": "#ff7f0e",
    "oneD_diffuse": "#bcbd22",
    "multi_mode_directional_geometry": "#2ca02c",
    "global_2D_directional_subspace": "#d62728",
    "global_kD_directional_subspace": "#9467bd",
    "residual_lowD_k": "#17becf",
    "unresolved_high_dimensional_or_diffuse": "#8c564b",
    "insufficient_effect_evidence": "#e377c2",
    "geometry_metrics_unavailable": "#7f7f7f",
    "undefined_geometry": "#aec7e8",
}
ATLAS_VECTOR_KEYS = (
    "r2",
    "c_ray",
    "s_span_1",
    "s_span_2",
    "s_span_3",
    "s_span_4",
    "s_span_8",
    "r_span_pr",
    "u_span_2",
    "d_span_2",
    "b_axis",
    "e_res",
    "r_ctr_pr",
    "delta_mix",
    "selected_mode_count",
    "min_mode_c_ray",
    "assignment_stability",
    "n_valid",
    "m_cv",
)


@dataclass(frozen=True)
class SpectralProjection:
    """Result of projecting directions through a feature-space kernel.

    Attributes:
        coordinates: Spectral coordinates with the requested column count.
        kernel: The finite, symmetric kernel used for eigendecomposition.
        eigenvalues: Eigenvalues in descending order, before zero padding.
        explained_ratios: Each eigenvalue divided by their nonnegative sum.
        numerical_rank: Count of eigenvalues above the scale-aware tolerance.
        centered: Whether the returned kernel was double-centered.
    """

    coordinates: NDArray[np.float64]
    kernel: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    explained_ratios: NDArray[np.float64]
    numerical_rank: int
    centered: bool


def project_directions(
    directions: ArrayLike,
    gram: ArrayLike,
    *,
    dimensions: int,
    centered: bool = False,
) -> SpectralProjection:
    """Project directions using the kernel ``D @ G @ D.T`` in float64.

    The kernel is optionally double-centered before a symmetric
    eigendecomposition. Eigenvectors receive a deterministic sign based on the
    largest-magnitude coordinate in each column, and missing requested
    dimensions are padded with zeros.

    Args:
        directions: Two-dimensional direction matrix ``D``.
        gram: Square feature-space kernel ``G`` matching ``D``.
        dimensions: Number of coordinate columns to return.
        centered: Whether to double-center the direction kernel.

    Returns:
        A deterministic spectral projection and its numerical diagnostics.

    Raises:
        ValueError: If inputs have invalid shapes, non-finite values, or the
            kernel has a materially negative eigenvalue.
    """

    # Validate the public numerical boundary before constructing the kernel.
    direction_array = np.asarray(directions, dtype=np.float64)
    gram_array = np.asarray(gram, dtype=np.float64)
    if direction_array.ndim != 2 or not direction_array.size:
        raise ValueError("directions must be a non-empty rank-2 array")
    if direction_array.shape[1] == 0 or dimensions <= 0:
        raise ValueError("directions and requested dimensions must be positive")
    feature_count = direction_array.shape[1]
    if gram_array.shape != (feature_count, feature_count):
        raise ValueError("gram must be square and match directions")

    # Form the exact float64 kernel, then share the spectral path with cached data.
    kernel = direction_array @ gram_array @ direction_array.T
    return project_kernel(kernel, dimensions=dimensions, centered=centered)


def project_kernel(
    kernel: ArrayLike,
    *,
    dimensions: int,
    centered: bool = False,
) -> SpectralProjection:
    """Project a compact FEGA direction kernel into signed coordinates.

    Args:
        kernel: Square matrix equivalent to ``D @ G @ D.T``.
        dimensions: Number of coordinate columns to return.
        centered: Whether to double-center the kernel first.

    Returns:
        A deterministic spectral projection and its numerical diagnostics.

    Raises:
        ValueError: If the kernel is malformed, non-finite, or materially indefinite.
    """
    # Validate compact cached geometry before symmetrizing numerical roundoff.
    kernel_array: NDArray[np.float64] = np.asarray(kernel, dtype=np.float64)
    if (
        kernel_array.ndim != 2
        or kernel_array.shape[0] == 0
        or kernel_array.shape[0] != kernel_array.shape[1]
        or dimensions <= 0
    ):
        raise ValueError("kernel must be non-empty and square and dimensions positive")
    kernel_array = (kernel_array + kernel_array.T) / 2.0
    if centered:
        row_mean = kernel_array.mean(axis=1, keepdims=True)
        kernel_array = kernel_array - row_mean - row_mean.T + kernel_array.mean()
        kernel_array = (kernel_array + kernel_array.T) / 2.0
    if not np.isfinite(kernel_array).all():
        raise ValueError("projected kernel must contain only finite values")

    # Decompose in descending order and reject kernels that are not PSD.
    eigenvalues, eigenvectors = np.linalg.eigh(kernel_array)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if eigenvalues[-1] < -1e-5:
        raise ValueError("projected kernel is not positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Build deterministic coordinates and pad unavailable dimensions with zero.
    take = min(dimensions, eigenvalues.size)
    coordinates = eigenvectors[:, :take] * np.sqrt(eigenvalues[:take])
    for column in range(take):
        anchor = int(np.argmax(np.abs(coordinates[:, column])))
        if coordinates[anchor, column] < 0.0:
            coordinates[:, column] *= -1.0
    if take < dimensions:
        coordinates = np.pad(coordinates, ((0, 0), (0, dimensions - take)))

    # Report scale-aware rank and explained ratios without dividing by zero.
    largest = float(eigenvalues[0]) if eigenvalues.size else 0.0
    tolerance = max(1e-12, largest * 1e-12)
    numerical_rank = int(np.count_nonzero(eigenvalues > tolerance))
    total = float(eigenvalues.sum())
    explained_ratios = (
        eigenvalues / total if total > 0.0 else np.zeros_like(eigenvalues)
    )
    return SpectralProjection(
        coordinates=coordinates,
        kernel=kernel_array,
        eigenvalues=eigenvalues,
        explained_ratios=explained_ratios,
        numerical_rank=numerical_rank,
        centered=centered,
    )


def surface_coordinates(
    coordinates: ArrayLike, *, tolerance: float = 1.0e-12
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Normalize nonzero coordinate rows onto the unit sphere surface.

    Rows with norm at most ``1e-12`` are omitted because they have no defined
    direction on the sphere.

    Args:
        coordinates: Rank-2 coordinate matrix.

    Returns:
        Unit-normalized rows and the mask selecting them from ``coordinates``.

    Raises:
        ValueError: If ``coordinates`` is not rank 2.
    """

    # Keep only rows with a well-defined direction, then normalize them.
    coordinate_array = np.asarray(coordinates, dtype=np.float64)
    if coordinate_array.ndim != 2:
        raise ValueError("coordinates must be a rank-2 array")
    norms = np.linalg.norm(coordinate_array, axis=1)
    valid = norms > tolerance
    return coordinate_array[valid] / norms[valid, None], valid


def rank_family_candidates(
    candidates: Sequence[Any] | Mapping[str, Sequence[Any]], *, top_k: int
) -> dict[str, list[Any]]:
    """Select deterministic top candidates independently for each family.

    Normal families sort by valid-count descending, median magnitude
    descending, then feature id ascending. The unresolved family uses its
    source diagnostics: valid-count descending, projected radial span
    descending, ray concentration ascending, projected center radius
    descending, then feature id ascending. Records may be typed objects or
    mappings but must expose the named fields.

    Args:
        candidates: Candidate records carrying ``primary_label``, or records
            already grouped by family name.
        top_k: Maximum records retained from each family.

    Returns:
        A new mapping containing deterministically ordered candidate lists.

    Raises:
        ValueError: If ``top_k`` is negative.
    """

    # Apply the source ranking tuple per family without copying artifact data.
    if top_k < 0:
        raise ValueError("top_k must be nonnegative")
    if isinstance(candidates, Mapping):
        candidates_by_family = candidates
    else:
        grouped: dict[str, list[Any]] = {}
        for candidate in candidates:
            family = str(_record_field(candidate, "primary_label"))
            grouped.setdefault(family, []).append(candidate)
        candidates_by_family = grouped

    ranked: dict[str, list[Any]] = {}
    for family, family_candidates in candidates_by_family.items():
        if family == "unresolved_high_dimensional_or_diffuse":
            ordered = sorted(
                family_candidates,
                key=lambda candidate: (
                    _descending_rank(_record_field(candidate, "n_valid")),
                    _descending_rank(_record_field(candidate, "r_span_pr")),
                    _ascending_rank(_record_field(candidate, "c_ray")),
                    _descending_rank(_record_field(candidate, "r_ctr_pr")),
                    int(_record_field(candidate, "feature_id")),
                ),
            )
        else:
            ordered = sorted(
                family_candidates,
                key=lambda candidate: (
                    _descending_rank(_record_field(candidate, "n_valid")),
                    _descending_rank(_record_field(candidate, "m_median")),
                    int(_record_field(candidate, "feature_id")),
                ),
            )
        ranked[family] = ordered[:top_k]
    return ranked


def atlas_coordinates(records: Sequence[Mapping[str, Any]]) -> NDArray[np.float64]:
    """Embed FEGA metric records with the source robust-preprocessing PCA path.

    Args:
        records: Flat records containing the fields in :data:`ATLAS_VECTOR_KEYS`.

    Returns:
        Two deterministic PCA coordinates per input record.
    """
    # Impute missing values by field medians and append their explicit mask.
    if not records:
        return np.empty((0, 2), dtype=np.float64)
    raw = np.asarray(
        [
            [
                _finite_float(record.get(key))
                if _finite_float(record.get(key)) is not None
                else np.nan
                for key in ATLAS_VECTOR_KEYS
            ]
            for record in records
        ],
        dtype=np.float64,
    )
    missing = ~np.isfinite(raw)
    imputed = raw.copy()
    for column in range(imputed.shape[1]):
        observed = raw[~missing[:, column], column]
        imputed[missing[:, column], column] = (
            float(np.median(observed)) if observed.size else 0.0
        )

    # Use the source scaler and PCA, including sklearn's near-constant scale policy.
    if len(records) == 1:
        return np.zeros((1, 2), dtype=np.float64)
    scaled = RobustScaler().fit_transform(imputed)
    matrix = np.column_stack([scaled, missing.astype(np.float64)])
    dimensions = min(2, matrix.shape[0], matrix.shape[1])
    coordinates = PCA(n_components=dimensions).fit_transform(matrix)
    if dimensions == 1:
        coordinates = np.column_stack([coordinates[:, 0], np.zeros(len(records))])
    return coordinates.astype(np.float64, copy=False)


def render_atlas(
    output_path: str | Path,
    records: Sequence[Mapping[str, Any]],
    coordinates: ArrayLike,
    *,
    dpi: int = 220,
) -> None:
    """Render a minimal FEGA family atlas from precomputed coordinates.

    Args:
        output_path: Destination PNG path.
        records: Records with ``primary_label`` and ``m_median`` fields.
        coordinates: Two-dimensional atlas coordinates aligned to ``records``.
        dpi: Output resolution.
    """
    # Draw label-colored points with the source effect-magnitude size rule.
    plt = load_pyplot()
    points = np.asarray(coordinates, dtype=np.float64)
    if points.shape != (len(records), 2):
        raise ValueError("atlas coordinates must have shape (records, 2)")
    figure, axis = plt.subplots(figsize=(10.5, 7.2), dpi=dpi)
    labels = [str(record["primary_label"]) for record in records]
    for family in CLASS_PALETTE:
        indices = [index for index, label in enumerate(labels) if label == family]
        if not indices:
            continue
        sizes = [
            16.0
            + 22.0
            * math.log1p(max(_finite_float(records[index].get("m_median")) or 0.0, 0.0))
            for index in indices
        ]
        axis.scatter(
            points[indices, 0],
            points[indices, 1],
            c=CLASS_PALETTE[family],
            s=sizes,
            alpha=0.76,
            edgecolors="white",
            linewidths=0.4,
            label=family,
        )
    axis.set_xlabel("Feature-map coordinate 1")
    axis.set_ylabel("Feature-map coordinate 2")
    axis.grid(True, alpha=0.25)
    if records:
        axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def render_sphere_surface(
    output_path: str | Path,
    coordinates: ArrayLike,
    *,
    color: str,
    dpi: int = 300,
    point_colors: Sequence[str] | None = None,
) -> None:
    """Render unit-sphere coordinates with the FEGA source camera and styling.

    Args:
        output_path: Image path passed to Matplotlib.
        coordinates: Three-dimensional unit-sphere coordinates.
        color: Default family color.
        dpi: Output resolution.
        point_colors: Optional colors matching the coordinate row count.

    Raises:
        ValueError: If coordinates do not have exactly three columns.
        ImportError: If Matplotlib is not installed.
    """

    # Import the optional plotting dependency only for an actual render call.
    plt = load_pyplot()
    points = np.asarray(coordinates, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("sphere coordinates must have shape (n, 3)")

    # Draw the source wire sphere, direction rays, and surface markers.
    figure = plt.figure(figsize=(6.2, 5.8))
    axis = figure.add_subplot(111, projection="3d")
    azimuth = np.linspace(0.0, 2.0 * np.pi, 32)
    polar = np.linspace(0.0, np.pi, 16)
    x = np.outer(np.cos(azimuth), np.sin(polar))
    y = np.outer(np.sin(azimuth), np.sin(polar))
    z = np.outer(np.ones_like(azimuth), np.cos(polar))
    axis.plot_wireframe(x, y, z, color="#B8C0CC", linewidth=0.35, alpha=0.42)
    colors = [color] * len(points) if point_colors is None else point_colors
    for point, point_color in zip(points, colors, strict=True):
        axis.quiver(
            0,
            0,
            0,
            *point,
            color=point_color,
            linewidth=0.8,
            alpha=0.55,
            arrow_length_ratio=0.18,
            normalize=False,
        )
    axis.scatter(*points.T, c=colors, edgecolors=colors, s=90, depthshade=False)
    axis.scatter([0], [0], [0], c="black", s=16)
    axis.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), zlim=(-1.05, 1.05))
    axis.set_box_aspect((1, 1, 1))
    axis.set_proj_type("ortho")
    axis.view_init(elev=18, azim=-55)
    axis.grid(False)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_zticks([])
    for pane_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
        pane_axis.pane.set_alpha(0.0)
        pane_axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        pane_axis.line.set_color("#202020")
        pane_axis.line.set_linewidth(0.8)
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(figure)


def render_projection_2d(
    output_path: str | Path,
    coordinates: ArrayLike,
    *,
    color: str,
    dpi: int = 300,
    point_colors: Sequence[str] | None = None,
    axis_guide: bool = False,
    view_padding: float = 1.35,
    primary_axis_guide: bool = True,
    view_limit: float | None = None,
    mode_assignments: Sequence[int] | None = None,
    fitted_line: bool = False,
) -> None:
    """Render the first two projection dimensions with source FEGA styling.

    Args:
        output_path: Image path passed to Matplotlib.
        coordinates: Rank-2 coordinates with at least two columns.
        color: Default family color.
        dpi: Output resolution.
        point_colors: Optional colors matching the coordinate row count.
        axis_guide: Whether to draw the unsigned-axis band.
        view_padding: Multiplicative symmetric axis padding.
        primary_axis_guide: Whether to emphasize the horizontal spectral axis.
        view_limit: Optional fixed symmetric display limit.
        mode_assignments: Optional fitted component per row.
        fitted_line: Whether to display the orthogonal best-fit line.

    Raises:
        ValueError: If fewer than two coordinate columns are provided.
        ImportError: If Matplotlib is not installed.
    """

    # Import lazily and validate the two-dimensional render boundary.
    plt = load_pyplot()
    points = np.asarray(coordinates, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("projection coordinates must have at least two columns")
    points = points[:, :2]
    maximum = float(np.max(np.abs(points), initial=0.0))
    limit = (
        max(maximum * view_padding, 1e-6) if view_limit is None else float(view_limit)
    )

    # Draw symmetric guides and points without decorative axes.
    figure, axis = plt.subplots(figsize=(6.2, 5.8))
    if axis_guide:
        axis.axhspan(-0.05 * limit, 0.05 * limit, color="#7A68A6", alpha=0.12)
    if primary_axis_guide:
        horizontal = axis.axhline(0, color="#7A68A6", linewidth=2.0)
        horizontal.set_dashes((14, 8))
    else:
        axis.axhline(0, color="#C4CDDC", linewidth=0.8, linestyle="--")
    axis.axvline(0, color="#C4CDDC", linewidth=0.8, linestyle="--")
    colors = [color] * len(points) if point_colors is None else point_colors

    # Add only the requested analytical guides derived from the shown points.
    if fitted_line and points.shape[0] >= 2:
        centroid = points.mean(axis=0)
        centered = points - centroid
        _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
        direction = right_vectors[0]
        positions = centered @ direction
        span = positions.max() - positions.min()
        line_positions = np.array(
            [positions.min() - 0.1 * span, positions.max() + 0.1 * span]
        )
        endpoints = centroid + np.outer(line_positions, direction)
        axis.plot(
            endpoints[:, 0],
            endpoints[:, 1],
            color=color,
            linewidth=4.0,
            alpha=0.52,
            solid_capstyle="round",
        )
    if mode_assignments is not None:
        if len(mode_assignments) != len(points):
            raise ValueError("mode_assignments must match the coordinate row count")
        assignments = np.asarray(mode_assignments)
        for mode in np.unique(assignments):
            members = np.flatnonzero(assignments == mode)
            centroid = points[members].mean(axis=0)
            mode_color = colors[int(members[0])]
            for member in members:
                axis.plot(
                    [centroid[0], points[member, 0]],
                    [centroid[1], points[member, 1]],
                    color=mode_color,
                    linewidth=0.9,
                    alpha=0.45,
                )
    axis.scatter(
        points[:, 0],
        points[:, 1],
        c=colors,
        s=108,
        edgecolors="white",
        linewidths=0.45,
    )
    axis.scatter([0], [0], c="#172B4D", s=18)
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(figure)


def _record_field(record: Any, name: str) -> Any:
    """Read one ranking field from either a mapping or typed artifact."""

    # Preserve typed artifact instances while also supporting serialized records.
    return record[name] if isinstance(record, Mapping) else getattr(record, name)


def _finite_float(value: Any) -> float | None:
    """Return one finite ranking or atlas value, otherwise ``None``."""
    # Keep missing and nonfinite data explicit for source-style imputation.
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None


def _descending_rank(value: Any) -> tuple[bool, float]:
    """Sort finite values descending while placing missing values last."""
    # Encode source missing-last ordering directly in one tuple.
    parsed = _finite_float(value)
    return parsed is None, 0.0 if parsed is None else -parsed


def _ascending_rank(value: Any) -> tuple[bool, float]:
    """Sort finite values ascending while placing missing values last."""
    # Encode source missing-last ordering directly in one tuple.
    parsed = _finite_float(value)
    return parsed is None, 0.0 if parsed is None else parsed
