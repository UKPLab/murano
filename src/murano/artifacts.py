"""Shared artifact types used across Murano pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PromptBatch:
    """Prompt inputs for generation-style experiments.

    Attributes:
        prompts: Prompts actually fed to the model.
        raw_prompts: Original prompts before templating, if available.
        source: Human-readable description of where the prompts came from.
        metadata: Arbitrary prompt-level metadata.
    """

    prompts: list[str]
    raw_prompts: list[str] | None = None
    source: str = "manual"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.prompts)


@dataclass
class GenerationComparison:
    """Paired baseline vs modified generations for the same prompts.

    Attributes:
        baseline_generations: Generations from the unmodified pipeline.
        modified_generations: Generations from the post-intervention pipeline,
            paired by index with ``baseline_generations``.
        prompts: Prompts used for generation, paired by index. May be None
            when the upstream step did not record them.
        baseline_label: Display label for the baseline column.
        modified_label: Display label for the modified column.
        metadata: Arbitrary comparison-level metadata.
    """

    baseline_generations: list[str]
    modified_generations: list[str]
    prompts: list[str] | None = None
    baseline_label: str = "clean"
    modified_label: str = "modified"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def clean_generations(self) -> list[str]:
        """Backward-compatible alias for baseline generations."""
        return self.baseline_generations

    @property
    def ablated_generations(self) -> list[str]:
        """Backward-compatible alias used by some evaluation code."""
        return self.modified_generations

    def __len__(self) -> int:
        return len(self.baseline_generations)


@dataclass
class MetricComparison:
    """Aggregate metric comparing baseline vs modified generations.

    Attributes:
        metric_name: Identifier of the metric (e.g., ``"compliance_rate"``).
        baseline_score: Aggregate score on the baseline generations.
        modified_score: Aggregate score on the modified generations.
        baseline_scores: Per-item scores on the baseline generations, when
            the metric exposes them.
        modified_scores: Per-item scores on the modified generations, when
            the metric exposes them.
        baseline_label: Display label for the baseline column.
        modified_label: Display label for the modified column.
        metadata: Arbitrary metric-level metadata (method name, parameters).
    """

    metric_name: str
    baseline_score: float
    modified_score: float
    baseline_scores: list[float] | None = None
    modified_scores: list[float] | None = None
    baseline_label: str = "clean"
    modified_label: str = "modified"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricScore:
    """Scalar result of a forward-pass evaluation metric.

    Holds one comparable number, plus an optional per-example breakdown, so a
    causal experiment ends in a value that can be compared across runs. Unlike
    :class:`MetricComparison`, which is shaped for baseline-vs-modified generation
    comparisons, this carries a single forward-pass score (logit difference,
    KL divergence, answer log-probability, recovered effect).

    Attributes:
        metric_name: Identifier of the metric (e.g. ``"logit_diff"``).
        value: Aggregate scalar score, typically the mean over examples.
        per_example: Per-example scores, when the metric exposes them.
        metadata: Arbitrary metric-level metadata (input keys, answer position,
            direction, recovered-metric endpoints).
    """

    metric_name: str
    value: float
    per_example: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
