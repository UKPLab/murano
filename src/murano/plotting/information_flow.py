"""Plotly visualization for InformationFlow results.

Renders each head's activation frequency (the share of prompts whose routes go
through it) as a layer-by-head heatmap, the view of the paper's Figs. 6 and 7.

Requires the ``plot`` extra (install with ``pip install murano-interp[plot]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from murano.nodes import SELF_ATTN
from murano.plotting.attention import plot_head_matrix

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from murano.steps.information_flow import InformationFlowResult


def plot_information_flow(
    result: InformationFlowResult,
    title: str = "Head activation frequency",
) -> go.Figure:
    """Render the per-head activation frequency as a layer-by-head heatmap.

    Args:
        result: InformationFlowResult produced by the InformationFlow step.
        title: Plot title; the threshold is appended.

    Returns:
        A ``plotly.graph_objects.Figure`` heatmap, layers on the y-axis and
        heads on the x-axis.
    """
    heads = [node for node in result.contributions if node.module == SELF_ATTN]
    n_layers = max(node.layer for node in heads) + 1
    n_heads = max(node.head or 0 for node in heads) + 1
    matrix = [[0.0] * n_heads for _ in range(n_layers)]
    for node in heads:
        matrix[node.layer][node.head or 0] = result.contributions[node]
    return plot_head_matrix(
        matrix,
        title=f"{title} (tau={result.threshold})",
        value_label="Frequency",
    )
