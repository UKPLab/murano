"""Plotting utilities for probing results.

Requires matplotlib, seaborn, and scikit-learn.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from murano.steps.probe import ProbeResult
    from murano.steps.record import LabeledActivationStore


def _setup():
    """Consistent seaborn style for all plots."""
    import seaborn as sns

    sns.set_theme(
        style="whitegrid", context="notebook", palette="muted", font_scale=1.1
    )
    return sns


def _save(fig, save_path):
    """Save figure with consistent settings."""
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")


def plot_probe_accuracy(
    probe: ProbeResult,
    save_path: str | Path | None = None,
) -> None:
    """Plot per-layer probe accuracy with cross-validation error bars.

    Highlights the best-scoring layer in a distinct colour.

    Args:
        probe: ProbeResult containing per-layer accuracy and CV fold scores.
        save_path: If provided, write the figure to this path.
    """
    import matplotlib.pyplot as plt

    sns = _setup()

    layers = sorted(probe.accuracy_per_layer.keys())
    means = [probe.accuracy_per_layer[layer] for layer in layers]
    stds = [probe.cv_scores[layer].std() for layer in layers]
    palette = [
        sns.color_palette("muted")[3]
        if layer == probe.best_layer
        else sns.color_palette("muted")[0]
        for layer in layers
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(
        [str(layer) for layer in layers],
        means,
        yerr=stds,
        capsize=3,
        color=palette,
        edgecolor="white",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title("Probe Accuracy by Layer")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    _save(fig, save_path)
    plt.close(fig)


def plot_confusion_matrix(
    probe: ProbeResult,
    store: LabeledActivationStore,
    save_path: str | Path | None = None,
) -> None:
    """Plot a confusion matrix at the best layer using the refitted classifier.

    No-op if ``probe.classifiers`` does not contain the best layer (e.g.
    when the Probe step ran without ``refit=True``).

    Args:
        probe: ProbeResult with refitted classifiers.
        store: LabeledActivationStore containing activations and labels.
        save_path: If provided, write the figure to this path.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix as cm_func

    sns = _setup()

    best = probe.best_layer
    if best not in probe.classifiers:
        return

    clf = probe.classifiers[best]
    X = store.activations[best].float().numpy()
    y_true = store.labels.numpy()
    y_pred = clf.predict(X)

    labels = probe.label_names or [str(i) for i in sorted(set(y_true))]
    matrix = cm_func(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        linewidths=0,
        linecolor="none",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (Layer {best})")

    plt.tight_layout()
    _save(fig, save_path)
    plt.close(fig)
