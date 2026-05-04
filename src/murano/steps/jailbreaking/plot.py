"""Plot step: generates and saves standard visualizations.

This step is jailbreaking-specific: it generates refusal-related plots.
"""

from __future__ import annotations

from pathlib import Path

from murano.logging import logger
from murano.results import Results
from murano.steps.base import Step


class Plot(Step):
    """Generate and save all plots into a ``plots/`` subdirectory.

    Reads from results (uses whatever is available):
        results['steering']: SteeringResult
        results['intervene']: InterveneResult
        results['eval']: EvalResult
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
            else Path(results.get("output_dir", "."))
        )
        plots_dir = root / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        if "steering" in results:
            plot_separation_scores(
                results["steering"], save_path=plots_dir / "separation_scores.png"
            )
            plot_direction_cosine_similarity(
                results["steering"], save_path=plots_dir / "cosine_similarity.png"
            )

        if "eval" in results:
            plot_compliance_comparison(
                results["eval"], save_path=plots_dir / "compliance_comparison.png"
            )

        prompts = None
        if "intervene" in results:
            if "prompts" in results:
                prompt_batch = results["prompts"]
                prompts = (
                    prompt_batch.raw_prompts
                    if prompt_batch.raw_prompts is not None
                    else prompt_batch.prompts
                )
            elif "dataset" in results and hasattr(results["dataset"], "positive_texts"):
                prompts = results["dataset"].positive_texts

        if "intervene" in results and prompts is not None:
            plot_refusal_heatmap(
                prompts=prompts,
                clean_generations=results["intervene"].clean_generations,
                modified_generations=results["intervene"].modified_generations,
                save_path=plots_dir / "refusal_heatmap.png",
            )

        logger.info("Plots saved to %s", plots_dir)
        return results
