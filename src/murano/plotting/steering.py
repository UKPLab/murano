"""Plotting utilities for steering directions.

Requires the ``plot`` extra (install with: pip install murano-interp[plot]).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from murano.plotting._common import _save, _setup

if TYPE_CHECKING:
    from murano.steps.train import SteeringResult


def plot_separation_scores(
    steering: SteeringResult,
    save_path: str | Path | None = None,
) -> None:
    """Plot separation scores across layers as a bar chart.

    Highlights the best-scoring layer in a distinct colour.

    Args:
        steering: SteeringResult containing per-layer separation scores.
        save_path: If provided, write the figure to this path; the parent
            directory is created if missing.
    """
    import matplotlib.pyplot as plt

    sns = _setup()

    layers = sorted(steering.separation_scores.keys())
    scores = [steering.separation_scores[layer] for layer in layers]
    palette = [
        sns.color_palette("muted")[3]
        if layer == steering.best_layer
        else sns.color_palette("muted")[0]
        for layer in layers
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=[str(layer) for layer in layers], y=scores, palette=palette, ax=ax)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Separation Score")
    ax.set_title("Separation Score by Layer")

    plt.tight_layout()
    _save(fig, save_path)
    plt.close(fig)


def plot_direction_cosine_similarity(
    steering: SteeringResult,
    save_path: str | Path | None = None,
) -> None:
    """Plot a heatmap of cosine similarities between per-layer directions.

    Args:
        steering: SteeringResult with one direction vector per layer.
        save_path: If provided, write the figure to this path.
    """
    import matplotlib.pyplot as plt
    import torch

    sns = _setup()

    layers = sorted(steering.direction_per_layer.keys())

    dirs = torch.stack(
        [steering.direction_per_layer[layer].float() for layer in layers]
    )
    dirs_norm = dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    sim_matrix = (dirs_norm @ dirs_norm.T).cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        sim_matrix,
        ax=ax,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0,
        linecolor="none",
        xticklabels=[str(layer) for layer in layers],
        yticklabels=[str(layer) for layer in layers],
        cbar_kws={"label": "Cosine Similarity", "shrink": 0.8},
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ax.set_title("Direction Cosine Similarity Across Layers")

    plt.tight_layout()
    _save(fig, save_path)
    plt.close(fig)
