"""Plot step for probing results."""

from __future__ import annotations

from pathlib import Path

from murano.logging import logger
from murano.results import Results
from murano.steps.base import Step


class ProbePlot(Step):
    """Generates and saves probing visualizations.

    Reads from results (uses whatever is available):
        results['probe']: ProbeResult
        results['record']: LabeledActivationStore (for confusion matrix)
        results['output_dir']: Path (optional)

    Args:
        output_dir: Root output directory. If None, uses results['output_dir'].
    """

    reads = []  # all reads are optional / conditional
    writes = []

    def __init__(self, output_dir: str | None = None):
        self.output_dir = output_dir

    def __call__(self, results: Results) -> Results:
        from murano.plotting.probing import (
            plot_probe_accuracy,
            plot_confusion_matrix,
        )

        root = (
            Path(self.output_dir)
            if self.output_dir
            else Path(results.get("output_dir", "."))
        )
        plots_dir = root / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        if "probe" in results:
            plot_probe_accuracy(
                results["probe"],
                save_path=plots_dir / "probe_accuracy.png",
            )
            if results["probe"].classifiers and "record" in results:
                plot_confusion_matrix(
                    results["probe"],
                    results["record"],
                    save_path=plots_dir / "confusion_matrix.png",
                )

        logger.info("Probing plots saved to %s", plots_dir)
        return results
