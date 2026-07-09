"""Plot step: saves and displays visualizations for whichever results are present."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from murano import keys
from murano.logging import logger
from murano.results import Results
from murano.steps.base import Step


def _display(fig: Any) -> None:
    """Render a figure inline when running inside an IPython/Jupyter session.

    A no-op outside IPython (e.g. a plain script or the test runner), so the
    step never emits stray output when there is no notebook to render into.
    """
    try:
        from IPython.core.getipython import get_ipython

        if get_ipython() is None:
            return
        from IPython.display import display
    except ImportError:
        return
    display(fig)


class Plot(Step):
    """Generate, save, and display plots for whichever results are present.

    A single generic plotting step. It inspects ``results`` and, for every
    domain it recognizes, builds an interactive Plotly figure, writes it under a
    ``plots/`` subdirectory, and displays it inline when run in a notebook, so
    the same step drops onto the end of any pipeline. Each domain is optional
    and handled only when its result is present:

        results['steering']: SteeringResult
            separation_scores, cosine_similarity
        results['probe']: ProbeResult
            probe_accuracy, plus confusion_matrix when the probe kept its
            classifiers and results['record'] is available
        results['logit_lens']: LogitLensResult
            logit_lens
        results['logit_attribution']: LogitAttributionResult
            logit_attribution

    Attention and SAE plots are not auto-dispatched: they need an explicit head
    or feature selection with no sensible default, so call the
    ``murano.plotting`` functions (``plot_attention_pattern``,
    ``plot_sae_token_activations``, ...) directly instead.

    Args:
        output_dir: Root output directory. If None, uses results['output_dir']
            when a preceding Save set it, otherwise ``murano_outputs``.
        show: Display each figure inline when run in a notebook. Defaults to True.
        save_format: File format for saved figures. ``"png"`` (default) writes a
            static image via kaleido and falls back to a self-contained
            interactive HTML page (with a warning) when kaleido is unavailable,
            as on many headless Slurm nodes. Pass ``"html"`` to always write the
            interactive page, or another kaleido format (e.g. ``"svg"``,
            ``"pdf"``). Inline display in a notebook is interactive regardless.
    """

    reads = []  # all reads are optional / conditional
    writes = []

    def __init__(
        self,
        output_dir: str | None = None,
        show: bool = True,
        save_format: str = "png",
    ):
        self.output_dir = output_dir
        self.show = show
        self.save_format = save_format

    def _emit(self, fig: Any, plots_dir: Path, name: str) -> None:
        """Save a figure and, when interactive, display it. Skips ``None``."""
        if fig is None:
            return
        path = plots_dir / f"{name}.{self.save_format}"
        if self.save_format == "html":
            fig.write_html(str(path), include_plotlyjs=True)
        else:
            from murano.plotting import save_figure

            save_figure(fig, path)
        if self.show:
            _display(fig)

    def __call__(self, results: Results) -> Results:
        root = (
            Path(self.output_dir)
            if self.output_dir
            else Path(results.get(keys.OUTPUT_DIR, keys.DEFAULT_OUTPUT_DIR))
        )
        plots_dir = root / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        if keys.STEERING in results:
            from murano.plotting import (
                plot_direction_cosine_similarity,
                plot_separation_scores,
            )

            steering = results[keys.STEERING]
            self._emit(plot_separation_scores(steering), plots_dir, "separation_scores")
            self._emit(
                plot_direction_cosine_similarity(steering),
                plots_dir,
                "cosine_similarity",
            )

        if keys.PROBE in results:
            from murano.plotting.probing import (
                plot_confusion_matrix,
                plot_probe_accuracy,
            )

            probe = results[keys.PROBE]
            self._emit(plot_probe_accuracy(probe), plots_dir, "probe_accuracy")
            if probe.classifiers and keys.RECORD in results:
                self._emit(
                    plot_confusion_matrix(probe, results[keys.RECORD]),
                    plots_dir,
                    "confusion_matrix",
                )

        if keys.LOGIT_LENS in results:
            from murano.plotting import plot_logit_lens

            self._emit(
                plot_logit_lens(results[keys.LOGIT_LENS]), plots_dir, "logit_lens"
            )

        if keys.LOGIT_ATTRIBUTION in results:
            from murano.plotting import plot_logit_attribution

            self._emit(
                plot_logit_attribution(results[keys.LOGIT_ATTRIBUTION]),
                plots_dir,
                "logit_attribution",
            )

        logger.info("Plots saved to %s", plots_dir)
        return results
