"""Plot step: generates and saves standard steering visualizations."""

from __future__ import annotations

from pathlib import Path

from murano import keys
from murano.logging import logger
from murano.results import Results
from murano.steps.base import Step


class Plot(Step):
    """Generate and save steering plots into a ``plots/`` subdirectory.

    Reads from results (uses whatever is available):
        results['steering']: SteeringResult

    Args:
        output_dir: Root output directory. If None, uses results['output_dir'].
    """

    reads = []  # all reads are optional / conditional
    writes = []

    def __init__(self, output_dir: str | None = None):
        self.output_dir = output_dir

    def __call__(self, results: Results) -> Results:
        from murano.plotting import (
            plot_separation_scores,
            plot_direction_cosine_similarity,
        )

        root = (
            Path(self.output_dir)
            if self.output_dir
            else Path(results.get(keys.OUTPUT_DIR, "."))
        )
        plots_dir = root / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        if keys.STEERING in results:
            plot_separation_scores(
                results[keys.STEERING], save_path=plots_dir / "separation_scores.png"
            )
            plot_direction_cosine_similarity(
                results[keys.STEERING], save_path=plots_dir / "cosine_similarity.png"
            )

        logger.info("Plots saved to %s", plots_dir)
        return results
