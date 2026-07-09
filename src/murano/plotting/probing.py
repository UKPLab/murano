"""Plotting utilities for probing results.

Requires the ``plot`` extra (plotly); the confusion matrix also needs the
``probe`` extra (scikit-learn).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from murano._optional import require_optional
from murano.plotting.plotly_utils import BAR_COLOR, HIGHLIGHT_COLOR

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from murano.steps.probe import ProbeResult
    from murano.steps.record import LabeledActivationStore


def plot_probe_accuracy(probe: ProbeResult) -> go.Figure:
    """Plot per-layer probe accuracy with cross-validation error bars.

    Highlights the best-scoring layer in a distinct color.

    Args:
        probe: ProbeResult containing per-layer accuracy and CV fold scores.

    Returns:
        An interactive Plotly bar chart, one bar per layer.
    """
    require_optional("plot", "plotly")
    import plotly.graph_objects as go

    layers = sorted(probe.accuracy_per_layer.keys())
    means = [probe.accuracy_per_layer[layer] for layer in layers]
    stds = [float(probe.cv_scores[layer].std()) for layer in layers]
    colors = [
        HIGHLIGHT_COLOR if layer == probe.best_layer else BAR_COLOR for layer in layers
    ]

    fig = go.Figure(
        go.Bar(
            x=[str(layer) for layer in layers],
            y=means,
            marker_color=colors,
            error_y={"type": "data", "array": stds, "visible": True},
        )
    )
    fig.update_layout(
        title="Probe Accuracy by Layer",
        xaxis_title="Layer",
        yaxis_title="Accuracy",
        yaxis={"range": [0, 1.05]},
    )
    return fig


def plot_confusion_matrix(
    probe: ProbeResult, store: LabeledActivationStore
) -> go.Figure | None:
    """Plot a confusion matrix at the best layer using the refitted classifier.

    Args:
        probe: ProbeResult with refitted classifiers.
        store: LabeledActivationStore containing activations and labels.

    Returns:
        An interactive Plotly heatmap, or None when the best layer has no
        refitted classifier (e.g. the Probe step ran without ``refit=True``).
    """
    require_optional("plot", "plotly")
    require_optional("probe")
    import plotly.graph_objects as go
    from sklearn.metrics import confusion_matrix as cm_func

    best = probe.best_layer
    if best not in probe.classifiers:
        return None

    clf = probe.classifiers[best]
    X = store.activations[best].float().numpy()
    y_true = store.labels.numpy()
    y_pred = clf.predict(X)

    labels = probe.label_names or [str(i) for i in sorted(set(y_true))]
    matrix = cm_func(y_true, y_pred)

    fig = go.Figure(
        go.Heatmap(
            z=matrix.tolist(),
            x=labels,
            y=labels,
            text=matrix.tolist(),
            texttemplate="%{text}",
            colorscale="Blues",
        )
    )
    fig.update_layout(
        title=f"Confusion Matrix (Layer {best})",
        xaxis_title="Predicted",
        yaxis_title="True",
        yaxis={"autorange": "reversed"},
    )
    return fig
