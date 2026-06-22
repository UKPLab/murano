"""Generic evaluation steps for generation comparisons."""

from __future__ import annotations

from typing import Callable

from murano import keys
from murano.artifacts import GenerationComparison, MetricResult
from murano.logging import logger
from murano.results import Results
from murano.steps.base import Step


class GenerationMetric(Step):
    """Score baseline vs modified generations with a user-provided metric.

    Reads from results:
        results['intervene']: GenerationComparison

    Writes to results:
        results['metric']: MetricResult

    Args:
        metric_name: Identifier written to ``MetricResult.metric_name``.
        score_fn: Aggregate scoring function applied to a list of generations.
        item_score_fn: Optional per-item scorer; results populate
            ``MetricResult.baseline_scores`` / ``modified_scores``.
        input_key: Results key to read the GenerationComparison from.
        output_key: Results key under which to write the MetricResult.
    """

    reads = [keys.INTERVENE]
    writes = [keys.METRIC]

    def __init__(
        self,
        metric_name: str,
        score_fn: Callable[[list[str]], float],
        item_score_fn: Callable[[str], float] | None = None,
        input_key: str = keys.INTERVENE,
        output_key: str = keys.METRIC,
    ):
        self.metric_name = metric_name
        self.score_fn = score_fn
        self.item_score_fn = item_score_fn
        self.input_key = input_key
        self.output_key = output_key
        self.reads = [input_key]
        self.writes = [output_key]

    def expected_read_types(self, results=None, available_types=None):
        """Return ``{input_key: GenerationComparison}``."""
        return {self.input_key: GenerationComparison}

    def expected_write_types(self, results=None, available_types=None):
        """Return ``{output_key: MetricResult}``."""
        return {self.output_key: MetricResult}

    def __call__(self, results: Results) -> Results:
        comparison = results[self.input_key]
        baseline = comparison.baseline_generations
        modified = comparison.modified_generations
        baseline_scores = None
        modified_scores = None

        if self.item_score_fn is not None:
            baseline_scores = [self.item_score_fn(text) for text in baseline]
            modified_scores = [self.item_score_fn(text) for text in modified]

        metric = MetricResult(
            metric_name=self.metric_name,
            baseline_score=self.score_fn(baseline),
            modified_score=self.score_fn(modified),
            baseline_scores=baseline_scores,
            modified_scores=modified_scores,
            baseline_label=comparison.baseline_label,
            modified_label=comparison.modified_label,
            metadata=dict(comparison.metadata),
        )
        results[self.output_key] = metric
        logger.info(
            "%s: %s=%.4f, %s=%.4f",
            self.metric_name,
            metric.baseline_label,
            metric.baseline_score,
            metric.modified_label,
            metric.modified_score,
        )
        return results
