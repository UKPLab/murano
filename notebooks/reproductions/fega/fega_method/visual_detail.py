"""Detailed FEGA views for subspace and residual geometry families."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


RESIDUAL_NEGATIVE_COLOR = "#3B4CC0"
RESIDUAL_NEUTRAL_COLOR = "#F7F7F7"
RESIDUAL_POSITIVE_COLOR = "#B40426"


def residual_point_colors(values: Sequence[float]) -> list[str]:
    """Map centered residual PC1 continuously to the source diverging scale."""
    # Normalize only color intensity; analytical coordinates remain unchanged.
    scale = max((abs(float(value)) for value in values), default=0.0)
    if scale == 0.0:
        return [RESIDUAL_NEUTRAL_COLOR] * len(values)
    colors = []
    for value in values:
        normalized = float(value) / scale
        colors.append(
            _interpolate_hex(
                RESIDUAL_NEUTRAL_COLOR,
                RESIDUAL_NEGATIVE_COLOR
                if normalized < 0.0
                else RESIDUAL_POSITIVE_COLOR,
                abs(normalized),
            )
        )
    return colors


def render_subspace_plane(
    output_path: str | Path,
    coordinates: np.ndarray,
    *,
    color: str,
    dpi: int,
    point_colors: Sequence[str] | None = None,
) -> None:
    """Render the global-2D family around its leading PC1-PC2 plane."""
    # Preserve PC3 height and show orthogonal feet instead of flattening the cloud.
    plt = load_pyplot()
    figure = plt.figure(figsize=(6.2, 5.8))
    axis = figure.add_subplot(111, projection="3d")
    _draw_subspace_plane(
        axis,
        np.asarray(coordinates, dtype=np.float64)[:, :3],
        color=color,
        point_colors=point_colors,
        show_feet=True,
    )
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(figure)


def render_residual_view(
    output_path: str | Path,
    coordinates: np.ndarray,
    *,
    selected_k: int,
    color: str,
    dpi: int,
    point_colors: Sequence[str],
) -> None:
    """Render the source view that exposes the selected residual dimension."""
    # Choose the smallest view that honestly shows k=2, k=3, or k=4 evidence.
    plt = load_pyplot()
    points = np.asarray(coordinates, dtype=np.float64)
    if selected_k == 2:
        figure = plt.figure(figsize=(6.2, 5.8))
        axis = figure.add_subplot(111, projection="3d")
        _draw_subspace_plane(
            axis, points[:, :3], color=color, point_colors=point_colors
        )
    elif selected_k == 3:
        figure = plt.figure(figsize=(6.2, 5.8))
        axis = figure.add_subplot(111, projection="3d")
        _draw_volume(axis, points[:, :3], point_colors)
    elif selected_k == 4:
        figure, axes = plt.subplots(2, 1, figsize=(6.2, 7.0))
        for axis, pair in zip(axes, (points[:, :2], points[:, 2:4]), strict=True):
            _draw_neutral_projection(axis, pair, point_colors, padding=1.18)
    else:
        figure, axis = plt.subplots(figsize=(6.2, 5.8))
        _draw_neutral_projection(axis, points[:, :2], point_colors)
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(figure)


def _draw_subspace_plane(
    axis: Any,
    coordinates: np.ndarray,
    *,
    color: str,
    point_colors: Sequence[str] | None,
    show_feet: bool = False,
) -> None:
    """Draw true leading-three coordinates around the origin PC1-PC2 plane."""
    # Use one scale on all axes so off-plane displacement is not exaggerated.
    maximum = float(np.max(np.abs(coordinates))) if coordinates.size else 1.0
    limit = max(maximum * 1.08, 1.0e-6)
    plane_x, plane_y = np.meshgrid([-limit, limit], [-limit, limit])
    axis.plot_surface(
        plane_x,
        plane_y,
        np.zeros_like(plane_x),
        color=color,
        alpha=0.14,
        shade=False,
        linewidth=0.0,
    )
    axis.plot(
        [-limit, limit],
        [0.0, 0.0],
        [0.0, 0.0],
        color=color,
        alpha=0.42,
        linewidth=0.9,
        linestyle="--",
    )
    axis.plot(
        [0.0, 0.0],
        [-limit, limit],
        [0.0, 0.0],
        color=color,
        alpha=0.42,
        linewidth=0.9,
        linestyle="--",
    )
    colors = [color] * len(coordinates) if point_colors is None else point_colors
    if show_feet:
        for point, point_color in zip(coordinates, colors, strict=True):
            axis.plot(
                [point[0], point[0]],
                [point[1], point[1]],
                [0.0, point[2]],
                color=point_color,
                linewidth=0.9,
                alpha=0.34,
            )
        axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            np.zeros(len(coordinates)),
            c=colors,
            edgecolors="white",
            linewidths=0.35,
            s=75.6,
            alpha=0.32,
            depthshade=False,
        )
    axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 2],
        c=colors,
        edgecolors="white",
        linewidths=0.45,
        s=108,
        depthshade=False,
    )
    _finish_3d(axis, limit, elev=24.0)


def _draw_volume(
    axis: Any, coordinates: np.ndarray, point_colors: Sequence[str]
) -> None:
    """Draw three neutral centered-residual axes and the true PC1-PC2-PC3 cloud."""
    # Keep all dimensions metrically equal and visually unprivileged.
    maximum = float(np.max(np.abs(coordinates))) if coordinates.size else 1.0
    limit = max(maximum * 1.12, 1.0e-6)
    for start, stop in (
        ((-limit, 0.0, 0.0), (limit, 0.0, 0.0)),
        ((0.0, -limit, 0.0), (0.0, limit, 0.0)),
        ((0.0, 0.0, -limit), (0.0, 0.0, limit)),
    ):
        axis.plot(
            [start[0], stop[0]],
            [start[1], stop[1]],
            [start[2], stop[2]],
            color="#C4CDDC",
            linewidth=0.8,
            linestyle="--",
        )
    axis.scatter(
        *coordinates.T,
        c=point_colors,
        edgecolors="white",
        linewidths=0.45,
        s=108,
        depthshade=False,
    )
    _finish_3d(axis, limit, elev=24.0)


def _finish_3d(axis: Any, limit: float, *, elev: float) -> None:
    """Apply the shared source camera and text-free three-dimensional frame."""
    # Keep the same origin, orthographic camera, and transparent panes.
    axis.scatter([0.0], [0.0], [0.0], c="#172B4D", s=18, depthshade=False)
    axis.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.set_proj_type("ortho")
    axis.view_init(elev=elev, azim=-55.0)
    axis.grid(False)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_zticks([])
    for dimension_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
        dimension_axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        dimension_axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        dimension_axis.line.set_color("#202020")
        dimension_axis.line.set_linewidth(0.8)


def _draw_neutral_projection(
    axis: Any,
    coordinates: np.ndarray,
    point_colors: Sequence[str],
    *,
    padding: float = 1.35,
) -> None:
    """Draw one source-neutral centered residual coordinate pair."""
    # Use symmetric equal limits with only neutral zero guides.
    maximum = float(np.max(np.abs(coordinates))) if coordinates.size else 1.0
    limit = max(maximum * padding, 1.0e-6)
    axis.axhline(0.0, color="#C4CDDC", linewidth=0.8, linestyle="--")
    axis.axvline(0.0, color="#C4CDDC", linewidth=0.8, linestyle="--")
    axis.scatter(
        *coordinates.T,
        c=point_colors,
        edgecolors="white",
        linewidths=0.45,
        s=108,
    )
    axis.scatter([0.0], [0.0], c="#172B4D", s=18)
    axis.set(xlim=(-limit, limit), ylim=(-limit, limit))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _interpolate_hex(start: str, end: str, weight: float) -> str:
    """Interpolate two RGB colors deterministically."""
    # Convert each channel directly to avoid another plotting dependency.
    start_rgb = tuple(int(start[index : index + 2], 16) for index in (1, 3, 5))
    end_rgb = tuple(int(end[index : index + 2], 16) for index in (1, 3, 5))
    channels = (
        round(left + weight * (right - left))
        for left, right in zip(start_rgb, end_rgb, strict=True)
    )
    return "#" + "".join(f"{channel:02X}" for channel in channels)


def load_pyplot() -> Any:
    """Load headless Matplotlib only for an actual render call."""
    # Keep FEGA analysis usable without importing a display backend.
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "Install notebooks/reproductions/fega/requirements.txt to render FEGA figures"
        ) from error

    return plt
