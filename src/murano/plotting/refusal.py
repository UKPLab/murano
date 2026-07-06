"""Plotting utilities for refusal results.

Requires the ``plot`` extra (install with: pip install murano-interp[plot]).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from murano.plotting._common import _save, _setup

if TYPE_CHECKING:
    from murano.steps.train import SteeringResult
    from murano.steps.refusal.evaluate import EvalResult


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


def plot_compliance_comparison(
    eval_result: EvalResult,
    save_path: str | Path | None = None,
) -> None:
    """Plot clean vs ablated compliance rates as side-by-side bars.

    Args:
        eval_result: EvalResult with ``clean_compliance`` and
            ``ablated_compliance`` attributes.
        save_path: If provided, write the figure to this path.
    """
    import matplotlib.pyplot as plt

    sns = _setup()

    labels = ["Clean", "Ablated"]
    values = [eval_result.clean_compliance, eval_result.ablated_compliance]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        labels,
        values,
        color=[sns.color_palette("muted")[0], sns.color_palette("muted")[3]],
        edgecolor="white",
    )
    ax.bar_label(
        bars,
        labels=[f"{val:.0%}" for val in values],
        label_type="edge",
        padding=3,
        fontsize=13,
        fontweight="bold",
    )

    ax.set_ylim(0, 1.15)
    ax.set_title("Compliance Rate: Clean vs Ablated")

    plt.tight_layout()
    _save(fig, save_path)
    plt.close(fig)


def plot_refusal_heatmap(
    prompts: list[str],
    clean_generations: list[str],
    modified_generations: list[str],
    refusal_phrases: list[str] | None = None,
    save_path: str | Path | None = None,
    max_prompts: int = 30,
    context_window: int = 300,
) -> None:
    """Plot a per-prompt refusal heatmap (compliant vs refusal).

    Two columns per row: clean and modified generations. Refusal is detected
    by phrase match in the leading window of each generation.

    Args:
        prompts: Original prompts used for generation.
        clean_generations: Baseline generations, paired with ``prompts``.
        modified_generations: Post-intervention generations, paired with
            ``prompts``.
        refusal_phrases: Phrases that signal a refusal. Defaults to
            ``murano.evaluation.REFUSAL_PHRASES`` when None.
        save_path: If provided, write the figure to this path.
        max_prompts: Truncate the plot to the first ``max_prompts`` rows.
        context_window: Leading characters of each generation to scan for a
            refusal phrase.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import numpy as np

    sns = _setup()

    from murano.evaluation import REFUSAL_PHRASES, is_refusal

    if refusal_phrases is None:
        refusal_phrases = REFUSAL_PHRASES

    n = min(len(prompts), max_prompts)
    matrix = np.zeros((n, 2))

    for i in range(n):
        matrix[i, 0] = float(
            is_refusal(clean_generations[i], refusal_phrases, context_window)
        )
        matrix[i, 1] = float(
            is_refusal(modified_generations[i], refusal_phrases, context_window)
        )

    prompt_labels = [p[:60] + "..." if len(p) > 60 else p for p in prompts[:n]]

    cmap = mcolors.ListedColormap(
        [sns.color_palette("muted")[2], sns.color_palette("muted")[3]]
    )

    fig, ax = plt.subplots(figsize=(8, max(5, n * 0.35)))
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Clean", "Ablated"])
    ax.set_yticks(range(n))
    ax.set_yticklabels(prompt_labels, fontsize=8)
    ax.set_title("Per-Prompt Refusal Detection")
    ax.grid(False)

    legend_patches = [
        mpatches.Patch(color=sns.color_palette("muted")[3], label="Refusal"),
        mpatches.Patch(color=sns.color_palette("muted")[2], label="Compliant"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9, framealpha=0.9)

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
