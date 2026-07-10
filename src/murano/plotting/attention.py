"""Plotly visualization for attention-pattern results.

Two views: a single head's query-by-key attention matrix
(:func:`plot_attention_pattern`), and a layer-by-head matrix of any per-head
statistic (:func:`plot_head_matrix`) such as entropy, sink mass, distance, or an
offset readout.

Requires the ``plot`` extra (install with ``pip install murano-interp[plot]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor  # pyright: ignore[reportPrivateImportUsage]

from murano._optional import require_optional
from murano.plotting.plotly_utils import plot_heatmap

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from murano.steps.attention import AttentionResult


def plot_attention_pattern(
    result: AttentionResult,
    layer: int,
    head: int,
    input_index: int = 0,
    title: str | None = None,
    color_scale: str = "Blues",
) -> go.Figure:
    """Render one head's attention matrix for a single input.

    Each row is a query token attending over the key tokens (the row sums to 1).
    Padding positions are dropped so only real tokens label the axes, and the
    query axis reads top to bottom in sequence order.

    Args:
        result: AttentionResult produced by the RecordAttention step.
        layer: Layer of the head to plot.
        head: Head index to plot.
        input_index: Which prompt to plot (default: first).
        title: Plot title; defaults to ``"Attention  L{layer} H{head}"``.
        color_scale: Plotly colorscale name.

    Returns:
        A ``plotly.graph_objects.Figure`` heatmap with query tokens on the
        y-axis and key tokens on the x-axis.
    """
    require_optional("plot", "plotly")

    mask = result.attention_mask[input_index].bool()
    keep = mask.tolist()
    pattern = result.head_pattern(layer, head, input_index)
    tokens = [tok for tok, k in zip(result.str_tokens[input_index], keep) if k]
    positioned = [f"{i}:{tok}" for i, tok in enumerate(tokens)]

    z = pattern[mask][:, mask].tolist()
    fig = plot_heatmap(
        z_data=z,
        x_labels=positioned,
        y_labels=positioned,
        title=title or f"Attention  L{layer} H{head}",
        color_scale=color_scale,
        colorbar_title="Attention weight",
        square=True,
    )
    fig.update_layout(xaxis_title="Key token", yaxis_title="Query token")
    # Sequence order top to bottom on the query axis (the natural reading order).
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top")
    return fig


def plot_head_matrix(
    matrix: Tensor | list[list[float]],
    title: str = "",
    layers: list[int] | None = None,
    value_label: str = "",
    color_scale: str = "Viridis",
    zmid: float | None = None,
) -> go.Figure:
    """Render a layer-by-head heatmap of a per-head statistic.

    Args:
        matrix: A ``[n_layers, n_heads]`` tensor or nested list, e.g. the output
            of ``AttentionResult.entropy()``.
        title: Plot title.
        layers: Layer indices for the y-axis labels; defaults to ``0..n-1``.
        value_label: Label for the colorbar (the statistic being shown).
        color_scale: Plotly colorscale name.
        zmid: Value anchored to the middle of the colorscale. Pass ``0`` for a
            signed statistic on a diverging scale, so a head with no effect is
            drawn in the neutral color.

    Returns:
        A ``plotly.graph_objects.Figure`` heatmap with layers on the y-axis
        (layer 0 at the top) and heads on the x-axis.
    """
    require_optional("plot", "plotly")

    z = matrix.tolist() if isinstance(matrix, Tensor) else matrix
    n_heads = len(z[0]) if z else 0
    label_layers = layers if layers is not None else list(range(len(z)))
    fig = plot_heatmap(
        z_data=z,
        x_labels=[f"H{head}" for head in range(n_heads)],
        y_labels=[f"L{layer}" for layer in label_layers],
        title=title,
        color_scale=color_scale,
        colorbar_title=value_label,
        zmid=zmid,
    )
    fig.update_layout(xaxis_title="Head", yaxis_title="Layer")
    fig.update_yaxes(autorange="reversed")
    return fig
