"""Plotting utilities for steering directions.

Requires the ``plot`` extra (install with: pip install murano-interp[plot]).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from murano._optional import require_optional
from murano.plotting.plotly_utils import BAR_COLOR, HIGHLIGHT_COLOR

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from murano.steps.train import SteeringResult


def plot_separation_scores(steering: SteeringResult) -> go.Figure:
    """Plot separation scores across layers as a bar chart.

    Highlights the best-scoring layer in a distinct color.

    Args:
        steering: SteeringResult containing per-layer separation scores.

    Returns:
        An interactive Plotly bar chart, one bar per layer.
    """
    require_optional("plot", "plotly")
    import plotly.graph_objects as go

    layers = sorted(steering.separation_scores.keys())
    scores = [steering.separation_scores[layer] for layer in layers]
    colors = [
        HIGHLIGHT_COLOR if layer == steering.best_layer else BAR_COLOR
        for layer in layers
    ]

    fig = go.Figure(
        go.Bar(x=[str(layer) for layer in layers], y=scores, marker_color=colors)
    )
    fig.update_layout(
        title="Separation Score by Layer",
        xaxis_title="Layer",
        yaxis_title="Separation Score",
    )
    return fig


def plot_direction_cosine_similarity(steering: SteeringResult) -> go.Figure:
    """Plot a heatmap of cosine similarities between per-layer directions.

    Args:
        steering: SteeringResult with one direction vector per layer.

    Returns:
        An interactive Plotly heatmap of the layer-by-layer cosine similarity.
    """
    require_optional("plot", "plotly")
    import plotly.graph_objects as go
    import torch

    layers = sorted(steering.direction_per_layer.keys())
    dirs = torch.stack(
        [steering.direction_per_layer[layer].float() for layer in layers]
    )
    dirs_norm = dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    sim_matrix = (dirs_norm @ dirs_norm.T).cpu().numpy()
    labels = [str(layer) for layer in layers]

    fig = go.Figure(
        go.Heatmap(
            z=sim_matrix.tolist(),
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale=[[0.0, "#3b4cc0"], [0.5, "#f7f7f7"], [1.0, "#b40426"]],
            colorbar={"title": "Cosine Similarity"},
        )
    )
    fig.update_layout(
        title="Direction Cosine Similarity Across Layers",
        xaxis_title="Layer",
        yaxis_title="Layer",
    )
    return fig
