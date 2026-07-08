"""Stateless Plotly visualization utilities.

Replaces the legacy ``BaseVisualizationLens`` classes with pure functions
that map data to ``plotly.graph_objects.Figure`` instances.

These functions do **not** accept ``Results`` objects; they operate on raw
Python data or tensors so they can be used independently of the pipeline.

Requires the ``plot`` extra (install with ``pip install murano-interp[plot]``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from murano._optional import require_optional

if TYPE_CHECKING:
    import plotly.graph_objects as go


def plot_heatmap(
    z_data: list[list[float]] | list[list[int]],
    x_labels: list[str] | None = None,
    y_labels: list[str] | None = None,
    title: str = "",
    color_scale: str = "Viridis",
    hover_data: list[list[str]] | None = None,
    colorbar_title: str = "",
    square: bool = False,
) -> go.Figure:
    """Create a heatmap figure.

    Args:
        z_data: 2-D matrix of values (rows × columns).
        x_labels: Labels for the x-axis (columns).
        y_labels: Labels for the y-axis (rows). Defaults to
            ``["Layer 0", "Layer 1", …]`` when ``None``.
        title: Plot title.
        color_scale: Plotly colorscale name (e.g. ``"Viridis"``, ``"RdBu_r"``).
        hover_data: Optional 2-D list of custom hover text, same shape as
            ``z_data``.
        colorbar_title: Label for the colorbar (the value legend); blank hides it.
        square: If True, force square cells (equal x and y scale), for a matrix
            whose axes share a unit such as an attention pattern.

    Returns:
        A ``plotly.graph_objects.Figure`` with a single heatmap trace.
    """
    require_optional("plot", "plotly")
    import plotly.graph_objects as go

    if y_labels is None:
        y_labels = [f"Layer {i}" for i in range(len(z_data))]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_data,
            x=x_labels,
            y=y_labels,
            colorscale=color_scale,
            colorbar=dict(title=dict(text=colorbar_title, side="right")),
            customdata=hover_data,
            hovertemplate=(
                "x: %{x}<br>y: %{y}<br>z: %{z}<br>%{customdata}"
                if hover_data is not None
                else None
            ),
        )
    )

    fig.update_layout(title=title, template="plotly_white")
    if square:
        fig.update_yaxes(scaleanchor="x", constrain="domain")
        fig.update_xaxes(constrain="domain")
    return fig


def plot_line_chart(
    x_data: list[float] | list[int],
    y_series: dict[str, list[float] | list[int]],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
) -> go.Figure:
    """Create a multi-line line chart figure.

    Args:
        x_data: Values for the x-axis.
        y_series: Mapping from trace name to y-values. Each entry produces
            one scatter trace.
        title: Plot title.
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.

    Returns:
        A ``plotly.graph_objects.Figure`` with one scatter trace per entry
        in ``y_series``.
    """
    require_optional("plot", "plotly")
    import plotly.graph_objects as go

    fig = go.Figure()

    for name, y_values in y_series.items():
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_values,
                mode="lines",
                name=name,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
    )
    return fig


def save_figure(
    fig: go.Figure,
    path: str | Path,
    width: int | None = None,
    height: int | None = None,
    scale: float = 2,
) -> Path:
    """Save a Plotly figure to disk, preferring a static image with an HTML fallback.

    ``fig.write_image`` needs a Chrome/kaleido backend that is not always present
    (headless GPU compute nodes are a common case); when it is unavailable the
    figure is instead written as a self-contained HTML file (``.html`` next to the
    requested path) so it is never lost.

    Args:
        fig: The figure to save.
        path: Destination path. A static-image suffix (``.png``, ``.svg``, …) is
            honored when image export works; otherwise ``.html`` is used.
        width: Image width in pixels (image export only).
        height: Image height in pixels (image export only).
        scale: Image resolution multiplier (image export only).

    Returns:
        The path actually written (the requested image path, or the ``.html``
        fallback).
    """
    require_optional("plot", "plotly")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(str(path), width=width, height=height, scale=scale)
        return path
    except Exception as exc:
        # No image backend (e.g. Chrome missing on a compute node): keep the
        # figure as interactive, self-contained HTML rather than failing. Log the
        # cause so a genuinely fixable export error is not silently swallowed.
        from murano.logging import logger

        html_path = path.with_suffix(".html")
        logger.warning(
            "save_figure: image export failed (%s); wrote HTML fallback %s",
            exc,
            html_path,
        )
        fig.write_html(str(html_path), include_plotlyjs=True)
        return html_path
