"""Shared artifact types used across Murano pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from murano.nodes import SELF_ATTN, Node, NodeDict


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
class ComponentSelection:
    """An ordered, scored set of component addresses chosen by a discovery step.

    A discovery step (for example ranking heads by direct logit attribution)
    writes this so a downstream :class:`~murano.steps.patch.Patch` or
    :class:`~murano.steps.path_patch.PathPatch` can read its targets at run time,
    letting attribute-then-patch compose in a single pipeline. It iterates over
    its :attr:`nodes` (best first), so it drops straight into the same target
    coercion the patch steps use for a hand-written address list.

    Attributes:
        nodes: Selected component addresses, best first.
        scores: ``{Node: float}`` score behind each selected node (the ranking
            value that put it in the selection).
        metadata: Provenance (source key, ranking criterion, cutoff).
    """

    nodes: list[Node]
    scores: dict[Node, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.nodes = [Node.coerce(node) for node in self.nodes]
        self.scores = NodeDict(self.scores)

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, node: object) -> bool:
        try:
            return Node.coerce(node) in self.nodes  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False


@dataclass
class SweepResult:
    """Per-item scores from running one step chain once per swept item.

    A :class:`~murano.steps.sweep.Sweep` forks the pipeline's ``Results`` once per
    item, applies the steps built for that item, and harvests one or more metric
    keys. Each harvested key becomes a *column*: an ordered ``{item: float}`` map,
    in the order the items were swept.

    When every swept item is a :class:`~murano.nodes.Node`, the sweep is a
    *component* sweep and :attr:`contributions` exposes the primary column as a
    :class:`~murano.nodes.NodeDict`. That is the same shape
    :class:`~murano.steps.logit_attribution.LogitAttributionResult` publishes, so a
    component sweep drops straight into
    :class:`~murano.steps.select.SelectComponents` without an adapter.

    Attributes:
        columns: ``{read_key: {item: float}}``, one entry per harvested key.
        primary: Read key whose column :attr:`scores` returns; the first key
            passed to ``Sweep(read=...)``.
        contributions: The primary column as a ``{Node: float}`` NodeDict when
            every swept item is a Node, otherwise None.
        metadata: Provenance (harvested keys, item labels, swept step names).
    """

    columns: dict[str, dict[Any, float]]
    primary: str
    metadata: dict[str, Any] = field(default_factory=dict)
    contributions: NodeDict | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.primary not in self.columns:
            raise ValueError(
                f"primary={self.primary!r} is not one of the harvested columns "
                f"{sorted(self.columns)}."
            )
        items = self.columns[self.primary]
        if items and all(isinstance(item, Node) for item in items):
            self.columns = {
                key: NodeDict(column) for key, column in self.columns.items()
            }
            self.contributions = self.columns[self.primary]  # type: ignore[assignment]

    @property
    def scores(self) -> dict[Any, float]:
        """The primary column: ``{item: value}`` in swept order."""
        return self.columns[self.primary]

    @property
    def swept(self) -> list[Any]:
        """The swept items, in the order they were swept.

        Deliberately not named ``items``: a mapping-shaped ``items`` attribute
        would make this artifact duck-type as a dict for callers that probe with
        ``hasattr(x, "items")``, and they would then call a list.
        """
        return list(self.scores)

    def column(self, key: str | None = None) -> dict[Any, float]:
        """Return one harvested column, defaulting to the primary one.

        Args:
            key: Harvested read key. None selects :attr:`primary`.

        Returns:
            The ``{item: value}`` map for ``key``.

        Raises:
            KeyError: If ``key`` was not harvested by the sweep.
        """
        name = self.primary if key is None else key
        if name not in self.columns:
            raise KeyError(
                f"'{name}' was not harvested by this sweep; it read "
                f"{sorted(self.columns)}."
            )
        return self.columns[name]

    def head_matrix(
        self,
        n_layers: int | None = None,
        n_heads: int | None = None,
        column: str | None = None,
        fill: float = float("nan"),
    ) -> list[list[float]]:
        """Return the scores as a dense ``[n_layers, n_heads]`` grid.

        The grid is what :func:`~murano.plotting.plot_head_matrix` consumes.
        Positions the sweep never visited hold ``fill``; the default NaN renders
        as a blank cell, so an unswept head never reads as a head with no effect.

        Args:
            n_layers: Grid height. Defaults to one past the deepest swept layer.
            n_heads: Grid width. Defaults to one past the highest swept head.
            column: Harvested key to render. None selects :attr:`primary`.
            fill: Value for positions the sweep did not visit.

        Returns:
            A nested list of floats, indexed ``[layer][head]``.

        Raises:
            TypeError: If the sweep's items are not all attention-head Nodes.
        """
        scores = self.column(column)
        bad = [
            item
            for item in scores
            if not isinstance(item, Node)
            or item.module != SELF_ATTN
            or item.head is None
        ]
        if bad or not scores:
            raise TypeError(
                "head_matrix() needs a sweep over attention heads "
                f"(Node(layer, 'self_attn', head=...)); got {bad[:3] or 'no items'}."
            )
        deepest = max(item.layer for item in scores)
        widest = max(item.head for item in scores)  # type: ignore[type-var]
        rows = deepest + 1 if n_layers is None else n_layers
        cols = widest + 1 if n_heads is None else n_heads
        if rows <= deepest or cols <= widest:
            raise ValueError(
                f"the sweep reaches L{deepest}H{widest}, which does not fit a "
                f"{rows}x{cols} grid."
            )
        grid = [[fill] * cols for _ in range(rows)]
        for item, value in scores.items():
            grid[item.layer][item.head] = value  # type: ignore[index]
        return grid


@dataclass
class MetricComparison:
    """Aggregate metric comparing baseline vs modified generations.

    Attributes:
        metric_name: Identifier of the metric (e.g., ``"logit_diff"``).
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
