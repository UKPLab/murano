"""Evaluate step: measures compliance rate after intervention.

This step is refusal-specific: it detects refusal phrases in model
generations. Generic evaluation steps (accuracy, KL div, etc.) belong
in murano/steps/ directly.
"""

from __future__ import annotations

from murano import keys
from murano.artifacts import GenerationComparison, MetricComparison
from murano.evaluation import REFUSAL_PHRASES, is_refusal, score_generations
from murano.logging import logger
from murano.results import Results
from murano.steps.base import Step


class ComplianceRate(Step):
    """Measure compliance rate by detecting refusal phrases.

    Reads from results:
        results['intervene']: GenerationComparison-compatible artifact

    Writes to results:
        results['eval']: MetricComparison (metric_name "compliance_rate")

    Args:
        method: Detection method. Currently only ``"keyword"`` is supported.
        context_window: Number of leading characters of each generation to
            scan for a refusal phrase.

    Raises:
        ValueError: If ``method`` is not a supported value.
    """

    reads = [keys.INTERVENE]
    writes = [keys.EVAL]
    read_types = {keys.INTERVENE: GenerationComparison}
    write_types = {keys.EVAL: MetricComparison}

    def __init__(self, method: str = "keyword", context_window: int = 300):
        if method != "keyword":
            raise ValueError("ComplianceRate currently only supports method='keyword'.")
        self.method = method
        self.context_window = context_window

    def __call__(self, results: Results) -> Results:
        intervene = results[keys.INTERVENE]
        clean_scores = [self._score_one(text) for text in intervene.clean_generations]
        ablated_scores = [
            self._score_one(text) for text in intervene.modified_generations
        ]
        clean_compliance = self._score(intervene.clean_generations)
        ablated_compliance = self._score(intervene.modified_generations)
        results[keys.EVAL] = MetricComparison(
            metric_name="compliance_rate",
            baseline_score=clean_compliance,
            modified_score=ablated_compliance,
            baseline_scores=clean_scores,
            modified_scores=ablated_scores,
            baseline_label="clean",
            modified_label="ablated",
            metadata={"method": "keyword", "context_window": self.context_window},
        )
        logger.info(
            "Compliance: clean=%.1f%%, ablated=%.1f%%",
            clean_compliance * 100,
            ablated_compliance * 100,
        )
        return results

    def _score(self, generations: list[str]) -> float:
        return score_generations(
            generations,
            refusal_phrases=REFUSAL_PHRASES,
            context_window=self.context_window,
        )

    def _score_one(self, generation: str) -> float:
        return float(
            not is_refusal(
                generation,
                refusal_phrases=REFUSAL_PHRASES,
                context_window=self.context_window,
            )
        )
