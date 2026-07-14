"""SelectComponents step: rank a per-component readout and pick the top components.

Turns a per-component importance readout into a concrete target set. It reads a
:class:`~murano.steps.logit_attribution.LogitAttributionResult` or a component
:class:`~murano.artifacts.SweepResult` (both map each head and MLP to a signed
score through ``contributions``), ranks the components, keeps the strongest, and
writes a :class:`~murano.artifacts.ComponentSelection`. A downstream
:class:`~murano.steps.patch.Patch` or :class:`~murano.steps.path_patch.PathPatch`
reads that selection at run time, so "score the important heads, then patch them"
runs as one pipeline instead of two with a hand-copied node list in between.
"""

from __future__ import annotations

from murano import keys
from murano.artifacts import ComponentSelection, SweepResult
from murano.logging import logger
from murano.nodes import Node, NodeDict, canonical_module
from murano.results import Results
from murano.steps.base import Step
from murano.steps.logit_attribution import LogitAttributionResult

_BY = ("abs", "signed", "negative")


def _extract_scores(source: object, source_key: str) -> NodeDict:
    """Return the ``{Node: float}`` scores of a per-component readout.

    Accepts anything exposing a ``contributions`` mapping (a
    :class:`LogitAttributionResult`, or a :class:`~murano.artifacts.SweepResult`
    whose items are all Nodes), or a bare ``{Node: float}`` mapping so a caller
    can rank scores it computed itself.

    Args:
        source: The readout to rank.
        source_key: Results key it came from, named in the error messages.

    Raises:
        TypeError: If ``source`` is a sweep over something other than Node
            addresses, or exposes neither a ``contributions`` map nor a mapping
            interface.
    """
    contributions = getattr(source, "contributions", None)
    if contributions is not None:
        return NodeDict(contributions)
    if isinstance(source, SweepResult):
        raise TypeError(
            f"SelectComponents ranks components, but the sweep under "
            f"'{source_key}' swept {source.metadata.get('items', [])[:3]}, which "
            f"are not Node addresses. Sweep over a NodeSet to select from the "
            f"result."
        )
    if hasattr(source, "items"):
        return NodeDict(source)
    raise TypeError(
        f"SelectComponents needs a result with per-component scores (a "
        f"LogitAttributionResult, a component SweepResult, or a {{Node: float}} "
        f"mapping); got {type(source).__name__} under '{source_key}'."
    )


class SelectComponents(Step):
    """Rank per-component scores and write the strongest as a ComponentSelection.

    Reads an attribution result's per-component scores, ranks them by magnitude
    (``by="abs"``, the default), signed value, or most-negative, and keeps either
    the top ``top_k`` or every component past ``threshold``. An optional
    ``modules`` filter restricts the pool before ranking, which is how you pick
    heads only for a :class:`~murano.steps.patch.Patch` (whose targets must be a
    single mode, all heads or all whole-components).

    Reads from results:
        results[source_key]: LogitAttributionResult or component SweepResult
            (default ``logit_attribution``).

    Writes to results:
        results[output_key]: ComponentSelection (default ``selection``).

    Args:
        source_key: Results key of the per-component readout to rank (default
            ``logit_attribution``); pass ``keys.SWEEP`` to rank a component sweep.
        top_k: Keep the ``top_k`` highest-ranked components. Pass this or
            ``threshold``, not both.
        threshold: Keep every component whose score passes the cutoff: ``by="abs"``
            keeps ``abs(score) >= threshold``, ``"signed"`` keeps
            ``score >= threshold``, ``"negative"`` keeps ``score <= threshold``.
        by: Ranking criterion: ``"abs"`` (largest magnitude, either sign),
            ``"signed"`` (most positive), or ``"negative"`` (most negative).
        modules: Optional module name or list of names to keep before ranking
            (e.g. ``"self_attn"`` for heads only); ``None`` ranks every component.
        output_key: Results key to write the ComponentSelection under.

    Raises:
        ValueError: If neither or both of ``top_k`` and ``threshold`` are given,
            ``top_k`` is not positive, or ``by`` is not one of the allowed values.
    """

    def __init__(
        self,
        source_key: str = keys.LOGIT_ATTRIBUTION,
        *,
        top_k: int | None = None,
        threshold: float | None = None,
        by: str = "abs",
        modules: str | list[str] | None = None,
        output_key: str = keys.SELECTION,
    ):
        if (top_k is None) == (threshold is None):
            raise ValueError(
                "Pass exactly one of top_k= (keep the k strongest) or threshold= "
                "(keep everything past a cutoff)."
            )
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}.")
        if by not in _BY:
            raise ValueError(f"by must be one of {_BY}, got {by!r}.")

        self.source_key = source_key
        self.top_k = top_k
        self.threshold = threshold
        self.by = by
        if modules is None:
            self.modules: set[str] | None = None
        else:
            names = [modules] if isinstance(modules, str) else list(modules)
            self.modules = {canonical_module(name) for name in names}
        self.output_key = output_key
        self.reads = [source_key]
        # _extract_scores also accepts a bare {Node: float} mapping, so include
        # dict here or the pipeline's pre-flight type check would reject the very
        # input the step supports.
        self.read_types = {source_key: (LogitAttributionResult, SweepResult, dict)}
        self.writes = [output_key]
        self.write_types = {output_key: ComponentSelection}

    def __call__(self, results: Results) -> Results:
        scores = _extract_scores(results[self.source_key], self.source_key)
        pool = {
            node: value
            for node, value in scores.items()
            if self.modules is None or node.module in self.modules
        }
        ranked = sorted(
            pool.items(), key=lambda item: self._rank(item[1]), reverse=True
        )

        if self.top_k is not None:
            chosen = ranked[: self.top_k]
        else:
            chosen = [(node, value) for node, value in ranked if self._passes(value)]

        if not chosen:
            raise ValueError(
                f"SelectComponents chose no component from '{self.source_key}' "
                f"(by={self.by!r}, top_k={self.top_k}, threshold={self.threshold}, "
                f"modules={sorted(self.modules) if self.modules else None}); "
                f"loosen the cutoff or widen the module filter."
            )

        nodes: list[Node] = [node for node, _ in chosen]
        logger.info(
            "SelectComponents: kept %d of %d component(s) from '%s' by %s",
            len(nodes),
            len(pool),
            self.source_key,
            self.by,
        )
        results[self.output_key] = ComponentSelection(
            nodes=nodes,
            scores={node: float(value) for node, value in chosen},
            metadata={
                "source_key": self.source_key,
                "by": self.by,
                "top_k": self.top_k,
                "threshold": self.threshold,
                "modules": sorted(self.modules) if self.modules else None,
                "n_available": len(pool),
            },
        )
        return results

    def _rank(self, value: float) -> float:
        """Return the sort key for a score under the active criterion (higher wins)."""
        if self.by == "abs":
            return abs(value)
        if self.by == "signed":
            return value
        return -value

    def _passes(self, value: float) -> bool:
        """Return whether a score clears ``threshold`` under the active criterion."""
        assert self.threshold is not None
        if self.by == "abs":
            return abs(value) >= self.threshold
        if self.by == "signed":
            return value >= self.threshold
        return value <= self.threshold
