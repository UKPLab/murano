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
    """Paired baseline vs modified generations for the same prompts."""

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
class MetricResult:
    """Aggregate metric comparing baseline vs modified generations."""

    metric_name: str
    baseline_score: float
    modified_score: float
    baseline_scores: list[float] | None = None
    modified_scores: list[float] | None = None
    baseline_label: str = "clean"
    modified_label: str = "modified"
    metadata: dict[str, Any] = field(default_factory=dict)
