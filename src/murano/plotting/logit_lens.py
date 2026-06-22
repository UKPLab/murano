"""Plotly visualization for LogitLens results.

Renders the standard logit-lens heatmap: rows are layers, columns are input
token positions, cell values are the argmax probability at that
(layer, position), and cell text is the predicted token.

Requires ``plotly`` (install with ``pip install murano[plot]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from murano.steps.logit_lens import LogitLensResult


def plot_logit_lens(
    result: LogitLensResult,
    input_index: int = 0,
    title: str = "Logit Lens",
    color_scale: str = "RdYlBu_r",
) -> go.Figure:
    """Render a logit-lens heatmap for a single input.

    Args:
        result: LogitLensResult produced by the LogitLens step.
        input_index: Which prompt to plot (default: first).
        title: Plot title.
        color_scale: Plotly colorscale name. Defaults to ``"RdYlBu_r"`` to
            match the conventional logit-lens visualization.

    Returns:
        A ``plotly.graph_objects.Figure`` with a heatmap trace; cell text
        shows the predicted token at each (layer, position).
    """
    import plotly.graph_objects as go

    mask = result.attention_mask[input_index].bool()
    keep = mask.tolist()

    max_probs = result.max_probs[:, input_index, :].numpy()
    predicted = [
        [tok for tok, k in zip(layer[input_index], keep) if k]
        for layer in result.predicted_words
    ]
    input_words = [tok for tok, k in zip(result.input_words[input_index], keep) if k]

    z = max_probs[:, mask].tolist()
    layer_labels = [f"Layer {address.layer}" for address in result.addresses]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=input_words,
            y=layer_labels,
            text=predicted,
            texttemplate="%{text}",
            colorscale=color_scale,
            zmid=0.5,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Input tokens",
        yaxis_title="Layer",
    )
    return fig
