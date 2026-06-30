"""Stateless Plotly visualization utilities.

Replaces the legacy ``BaseVisualizationLens`` classes with pure functions
that map data to ``plotly.graph_objects.Figure`` instances.

These functions do **not** accept ``Results`` objects; they operate on raw
Python data or tensors so they can be used independently of the pipeline.

Requires the ``plot`` extra (install with ``pip install murano-interp[plot]``).
"""

from __future__ import annotations

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
            customdata=hover_data,
            hovertemplate=(
                "x: %{x}<br>y: %{y}<br>z: %{z}<br>%{customdata}"
                if hover_data is not None
                else None
            ),
        )
    )

    fig.update_layout(title=title)
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
