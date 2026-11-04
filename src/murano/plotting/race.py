"""Plotly visualization for RACE results.

Draws the distribution of positive CAM scores in every layer, one panel per
module, as half-violins over ``log10`` of the score (the layout of Fig. 2 in the RACE paper).
It shows where in depth the consistent neurons sit, and why the selection
ranks within each layer instead of across the model.

Requires the ``plot`` extra (install with ``pip install murano-interp[plot]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from murano._optional import require_optional

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from murano.steps.race import RACEResult


def plot_race_layers(
    result: RACEResult,
    gamma: float | None = None,
    max_points: int = 100,
    title: str = "RACE: CAM scores by layer",
) -> go.Figure:
    """Render each layer's positive CAM scores as a half-violin, per module.

    Neurons with a zero score (no confident positive alignment) are left out,
    since they have no logarithm; the trace name says how many remain.

    Args:
        result: RACEResult produced by the RACE step.
        gamma: Credible level for the CAM bound; defaults to ``result.gamma``.
        max_points: Each layer's scores are reduced to this many evenly spaced
            quantiles, which keeps the figure small on wide models.
        title: Plot title.

    Returns:
        A ``plotly.graph_objects.Figure`` with one column per scored module.
    """
    require_optional("plot", "plotly")
    import plotly.graph_objects as go
    from torch import linspace  # pyright: ignore[reportPrivateImportUsage]
    from plotly.subplots import make_subplots

    cam = result.cam(gamma)
    modules = sorted({node.module for node in cam})
    fig = make_subplots(
        rows=1, cols=len(modules), shared_yaxes=True, subplot_titles=modules
    )
    for col, module in enumerate(modules, start=1):
        for node in sorted(n for n in cam if n.module == module):
            positive = cam[node][cam[node] > 0].sort().values
            if not positive.numel():
                continue
            picks = linspace(0, positive.numel() - 1, max_points).round()
            values = positive[picks.long().unique()].log10().tolist()
            fig.add_trace(
                go.Violin(
                    x=values,
                    y=[f"L{node.layer}"] * len(values),
                    name=f"L{node.layer} ({positive.numel()} > 0)",
                    orientation="h",
                    side="positive",
                    width=1.8,
                    points=False,
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
        # Plotly's violin KDE ignores a log axis, so take the log here instead.
        fig.update_xaxes(title_text="log10 CAM score", row=1, col=col)
    fig.update_yaxes(title_text="Layer", row=1, col=1)
    fig.update_layout(title=title)
    return fig
