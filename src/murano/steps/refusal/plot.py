"""Plot step: generates and saves standard visualizations.

This step is refusal-specific: it generates refusal-related plots.
"""

from __future__ import annotations

from pathlib import Path

from murano import keys
from murano.logging import logger
from murano.results import Results
from murano.steps.base import Step


class Plot(Step):
    """Generate and save all plots into a ``plots/`` subdirectory.

    Reads from results (uses whatever is available):
        results['steering']: SteeringResult
        results['intervene']: InterveneResult
        results['eval']: MetricComparison
        results['prompts']: PromptBatch
        results['output_dir']: Path (from Save step, optional)

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
            plot_compliance_comparison,
            plot_refusal_heatmap,
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

        if keys.EVAL in results:
            plot_compliance_comparison(
                results[keys.EVAL], save_path=plots_dir / "compliance_comparison.png"
            )

        prompts = None
        if keys.INTERVENE in results:
            if keys.PROMPTS in results:
                prompt_batch = results[keys.PROMPTS]
                prompts = (
                    prompt_batch.raw_prompts
                    if prompt_batch.raw_prompts is not None
                    else prompt_batch.prompts
                )
            elif keys.DATASET in results and hasattr(
                results[keys.DATASET], "positive_texts"
            ):
                prompts = results[keys.DATASET].positive_texts

        if keys.INTERVENE in results and prompts is not None:
            plot_refusal_heatmap(
                prompts=prompts,
                clean_generations=results[keys.INTERVENE].clean_generations,
                modified_generations=results[keys.INTERVENE].modified_generations,
                save_path=plots_dir / "refusal_heatmap.png",
            )

        logger.info("Plots saved to %s", plots_dir)
        return results
