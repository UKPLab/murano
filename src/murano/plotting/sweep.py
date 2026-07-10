"""Plot the scores a Sweep collected, choosing the shape from the swept items.

A sweep over attention heads has a canonical picture, the layer-by-head heatmap.
Anything else (layer indices, ablation method names) has no such geometry, so it
falls back to one bar per swept item.

Requires the ``plot`` extra (install with ``pip install murano-interp[plot]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from murano._optional import require_optional
from murano.nodes import SELF_ATTN, Node
from murano.plotting.attention import plot_head_matrix
from murano.plotting.plotly_utils import BAR_COLOR, HIGHLIGHT_COLOR

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from murano.artifacts import SweepResult


def _is_head_sweep(scores: dict) -> bool:
    """Whether every swept item addresses one attention head."""
    return bool(scores) and all(
        isinstance(item, Node) and item.module == SELF_ATTN and item.head is not None
        for item in scores
    )


def plot_sweep(
    result: SweepResult,
    column: str | None = None,
    title: str = "",
    diverging: bool | None = None,
) -> go.Figure:
    """Render a sweep's scores: a layer-by-head heatmap, or a bar per item.

    A sweep over attention heads becomes the canonical heatmap; anything else
    (layer indices, ablation method names, sender/receiver pairs) becomes one bar
    per item, in swept order.

    Args:
        result: The SweepResult to render.
        column: Harvested key to render. None selects the sweep's primary column.
        title: Plot title. Defaults to the column name.
        diverging: Center the color scale (heatmap) or color by sign (bars) at
            zero. None decides from the data: a column whose values straddle zero
            is diverging, one that does not is sequential. A signed statistic
            drawn on a sequential scale reads as if zero had a sign.

    Returns:
        A ``plotly.graph_objects.Figure``.
    """
    require_optional("plot", "plotly")
    import plotly.graph_objects as go

    scores = result.column(column)
    name = column or result.primary
    label = title or name
    values = list(scores.values())
    if diverging is None:
        diverging = min(values) < 0.0 < max(values)

    if _is_head_sweep(scores):
        return plot_head_matrix(
            result.head_matrix(column=column),
            title=label,
            value_label=name,
            color_scale="RdBu" if diverging else "Viridis",
            zmid=0.0 if diverging else None,
        )

    labels = [str(item) for item in scores]
    colors = (
        [HIGHLIGHT_COLOR if value < 0 else BAR_COLOR for value in values]
        if diverging
        else BAR_COLOR
    )
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors))
    fig.update_layout(
        title=label,
        xaxis_title="item",
        yaxis_title=name,
        template="plotly_white",
        showlegend=False,
    )
    return fig
