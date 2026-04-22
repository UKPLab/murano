"""Plotting utilities for jailbreaking results.

Requires matplotlib and seaborn (install with: pip install murano[plot]).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from murano.steps.train import SteeringResult
    from murano.steps.jailbreaking.evaluate import EvalResult


def _setup():
    """Consistent seaborn style for all plots."""
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="notebook", palette="muted", font_scale=1.1)
    return sns


def _save(fig, save_path):
    """Save figure with consistent settings."""
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")


def plot_separation_scores(
    steering: SteeringResult,
    save_path: str | Path | None = None,
) -> None:
    """Bar chart of separation scores across layers."""
    import matplotlib.pyplot as plt
    sns = _setup()

    layers = sorted(steering.separation_scores.keys())
    scores = [steering.separation_scores[l] for l in layers]
    palette = [sns.color_palette("muted")[3] if l == steering.best_layer
               else sns.color_palette("muted")[0] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=[str(l) for l in layers], y=scores, palette=palette, ax=ax)
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
    """Side-by-side bar chart comparing clean and ablated compliance rates."""
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
) -> None:
    """Heatmap showing refusal per prompt (green=compliant, red=refusal)."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import numpy as np
    sns = _setup()

    from murano.evaluation import REFUSAL_PHRASES

    if refusal_phrases is None:
        refusal_phrases = REFUSAL_PHRASES

    def is_refusal(text: str) -> bool:
        return any(p in text.lower()[:300] for p in refusal_phrases)

    n = min(len(prompts), max_prompts)
    matrix = np.zeros((n, 2))

    for i in range(n):
        matrix[i, 0] = 1.0 if is_refusal(clean_generations[i]) else 0.0
        matrix[i, 1] = 1.0 if is_refusal(modified_generations[i]) else 0.0

    prompt_labels = [p[:60] + "..." if len(p) > 60 else p for p in prompts[:n]]

    cmap = mcolors.ListedColormap([sns.color_palette("muted")[2], sns.color_palette("muted")[3]])

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
    """Heatmap of cosine similarities between direction vectors at different layers."""
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    sns = _setup()

    layers = sorted(steering.direction_per_layer.keys())
    n = len(layers)

    dirs = torch.stack([steering.direction_per_layer[l].float() for l in layers])
    dirs_norm = dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    sim_matrix = (dirs_norm @ dirs_norm.T).cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(sim_matrix, ax=ax, cmap="RdBu_r", vmin=-1, vmax=1, square=True,
                linewidths=0, linecolor="none",
                xticklabels=[str(l) for l in layers],
                yticklabels=[str(l) for l in layers],
                cbar_kws={"label": "Cosine Similarity", "shrink": 0.8})
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ax.set_title("Direction Cosine Similarity Across Layers")

    plt.tight_layout()
    _save(fig, save_path)
    plt.close(fig)
