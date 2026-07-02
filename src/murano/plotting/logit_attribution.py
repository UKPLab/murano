"""Plotly visualization for LogitAttribution results.

Renders the top contributors to the target logit (difference) as a horizontal
bar chart, signed so positive bars push toward the answer and negative bars push
away. The embedding and the catch-all ``other`` term are included alongside the
named attention-head and MLP components.

Requires the ``plot`` extra (install with ``pip install murano-interp[plot]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from murano._optional import require_optional

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from murano.steps.logit_attribution import LogitAttributionResult


def plot_logit_attribution(
    result: LogitAttributionResult,
    top_k: int = 20,
    title: str = "Direct Logit Attribution",
) -> go.Figure:
    """Render the top-contributing components as a signed horizontal bar chart.

    Args:
        result: LogitAttributionResult produced by the LogitAttribution step.
        top_k: Number of components to show, ranked by absolute contribution.
        title: Plot title.

    Returns:
        A ``plotly.graph_objects.Figure`` with one horizontal bar per component,
        most influential at the top, colored by sign.
    """
    require_optional("plot", "plotly")
    import plotly.graph_objects as go

    items: list[tuple[str, float]] = [
        (str(node), value) for node, value in result.contributions.items()
    ]
    items.append(("embed", result.embed_contribution))
    items.append(("other", result.other_contribution))

    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    items = items[:top_k]
    # Largest at the top: plotly draws the first bar at the bottom, so reverse.
    items.reverse()

    labels = [name for name, _ in items]
    values = [value for _, value in items]
    colors = ["#d62728" if v < 0 else "#1f77b4" for v in values]

    fig = go.Figure(
        data=go.Bar(x=values, y=labels, orientation="h", marker_color=colors)
    )
    fig.update_layout(
        title=f"{title} ({result.target}, total={result.total:.2f})",
        xaxis_title="Contribution (logits)",
        yaxis_title="Component",
    )
    return fig
