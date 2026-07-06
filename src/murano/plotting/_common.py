"""Shared style and dependency helpers for the matplotlib/seaborn plots."""

from __future__ import annotations

from pathlib import Path

from murano._optional import require_optional


def _require_plotting():
    """Ensure the plotting dependencies are installed and return them.

    Returns:
        The imported ``matplotlib.pyplot`` and ``seaborn`` modules.
    """
    require_optional("plot", "matplotlib", "seaborn")
    import matplotlib.pyplot as plt
    import seaborn as sns

    return plt, sns


def _setup():
    """Consistent seaborn style for all plots."""
    _, sns = _require_plotting()

    sns.set_theme(
        style="whitegrid", context="notebook", palette="muted", font_scale=1.1
    )
    return sns


def _save(fig, save_path):
    """Save figure with consistent settings."""
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
