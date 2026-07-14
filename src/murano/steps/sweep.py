"""Sweep step: run one step chain once per item and harvest a metric.

The shape every component study has in common. "Patch each head and measure how
much behavior it restores", "zero each head and measure the logit difference",
"steer at each layer and measure the effect": all of them fork the pipeline's
state, apply a couple of steps built for one component, read a number back out,
and repeat. Written by hand that is a closure over the model, the task, and a
baseline ``Results``, rebuilt in every notebook that needs it.

``Sweep`` is that loop as a step. It runs the shared prefix once (whatever the
pipeline built before it), then forks the ``Results`` per item so each item sees
the same starting state, and collects the harvested keys into a
:class:`~murano.artifacts.SweepResult`. Because the forks are shallow copies, the
swept steps' writes are scratch: they never reach the pipeline that follows. The
isolation is key-level only, though: a swept step that mutates a shared value in
place (rather than writing a new key) would leak that change across items, so
swept steps must treat the incoming ``Results`` values as read-only.

The swept steps are built by a callback, so what varies is not restricted to a
component. Sweeping over :class:`~murano.nodes.Node` addresses yields a component
sweep, whose scores feed :class:`~murano.steps.select.SelectComponents` and
:func:`~murano.plotting.plot_head_matrix` directly; sweeping over layer indices,
ablation method names, or ``(sender, receiver)`` pairs works the same way and
yields a plain ``{item: value}`` map.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

from murano import keys
from murano.artifacts import SweepResult
from murano.logging import logger
from murano.results import Results
from murano.steps.base import ExpectedType, Step
from murano.steps.metrics import _metric_score

StepFactory = Callable[[Any], "Step | Sequence[Step]"]


def _chain_io(steps: Sequence[Step]) -> tuple[list[str], set[str], dict[str, Any]]:
    """Return the net reads, the writes, and the read types of a step chain.

    A key a later step reads is only an external requirement when no earlier step
    in the chain wrote it, which is the same rule
    :meth:`~murano.pipeline.Pipeline.validate` applies across a pipeline.

    Args:
        steps: The chain, in execution order.

    Returns:
        Tuple of (keys the chain needs from outside, keys it writes, the expected
        types for the needed keys).
    """
    needed: list[str] = []
    produced: set[str] = set()
    read_types: dict[str, Any] = {}
    for step in steps:
        expected = step.expected_read_types()
        for key in step.reads:
            if key not in produced and key not in needed:
                needed.append(key)
                if key in expected:
                    read_types[key] = expected[key]
        produced.update(step.writes)
    return needed, produced, read_types


class Sweep(Step):
    """Run ``steps(item)`` once per item and harvest one or more metric keys.

    Each item gets a shallow fork of the incoming ``Results``, so every item is
    measured against the same baseline and the swept steps' intermediate writes
    are discarded. The harvested values are coerced to floats the same way the
    metric steps coerce their inputs, so reading a ``MetricScore`` key yields its
    ``value``.

    Reads from results:
        Whatever the swept chain needs and does not write itself, computed from
        the chain built for the first item. A ``Sweep`` whose chain reads
        ``logit_diff`` from a prior step therefore fails pre-flight validation
        when nothing upstream produced it, exactly like any other step.

    Writes to results:
        results[output_key]: SweepResult.

    Args:
        over: The items to sweep. Each must be hashable, since it keys the
            resulting scores. A ``NodeSet`` of heads gives a component sweep.
        steps: Callback building the step (or steps) to run for one item. Called
            once per item, and once up front to derive this step's read contract.
        read: Results key, or keys, to harvest after the chain runs. The first is
            the ``SweepResult``'s primary column. Every key must be written by
            the swept chain: harvesting a key the chain never writes would record
            the same baseline value for every item.
        output_key: Results key to write the SweepResult under.

    Raises:
        ValueError: If ``over`` is empty, ``read`` is empty, ``steps`` builds an
            empty chain, or a harvested key is never written by that chain.
        TypeError: If ``steps`` builds something that is not a Step.

    Example:
        Patch every head of a corrupted run and keep the four that restore the
        most behavior::

            Pipeline([
                LoadPaired(task),
                Logits(model),
                LogitDiffStep(correct, incorrect, output_key="ld_clean"),
                Sweep(
                    over=NodeSet.product(range(model.n_layers), ["self_attn"],
                                         heads=range(model.n_heads)),
                    steps=lambda head: [
                        Patch(model, targets=[head]),
                        LogitDiffStep(correct, incorrect,
                                      logits_key=keys.PATCHED_LOGITS,
                                      mask_key=keys.PATCHED_MASK,
                                      output_key="ld_patched"),
                    ],
                    read="ld_patched",
                ),
                SelectComponents(source_key=keys.SWEEP, top_k=4),
            ]).run()
    """

    def __init__(
        self,
        over: Iterable[Any],
        steps: StepFactory,
        read: str | Sequence[str],
        output_key: str = keys.SWEEP,
    ):
        self.over = list(over)
        if not self.over:
            raise ValueError("Sweep needs at least one item to sweep over.")

        self.read = [read] if isinstance(read, str) else list(read)
        if not self.read:
            raise ValueError("Sweep needs at least one key to harvest.")

        self._factory = steps
        self.output_key = output_key

        probe = self._steps_for(self.over[0])
        needed, produced, read_types = _chain_io(probe)
        unwritten = [key for key in self.read if key not in produced]
        if unwritten:
            chain = ", ".join(type(step).__name__ for step in probe)
            raise ValueError(
                f"Sweep harvests {unwritten}, which the swept chain ({chain}) "
                f"never writes; every item would record the same value. The "
                f"chain writes {sorted(produced)}."
            )

        self.reads = needed
        self.read_types: dict[str, ExpectedType] = read_types
        self.writes = [output_key]
        self.write_types = {output_key: SweepResult}
        # Named here rather than rebuilt at write time: the factory may be
        # expensive, and the chain's shape is fixed by the first item anyway.
        self._chain_names = [type(step).__name__ for step in probe]

    def _steps_for(self, item: Any) -> list[Step]:
        """Build and validate the step chain for one swept item."""
        built = self._factory(item)
        chain = [built] if isinstance(built, Step) else list(built)
        if not chain:
            raise ValueError(f"Sweep's steps callback built no step for {item!r}.")
        for step in chain:
            if not isinstance(step, Step):
                raise TypeError(
                    f"Sweep's steps callback must build Step objects, got "
                    f"{type(step).__name__} for item {item!r}."
                )
        return chain

    def _harvest(self, forked: Results, item: Any) -> dict[str, float]:
        """Read the harvested keys off one finished fork as floats."""
        values = {}
        for key in self.read:
            try:
                values[key] = _metric_score(forked[key])
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"Sweep harvests '{key}' as a number, but item {item!r} "
                    f"produced {type(forked[key]).__name__}. Append a metric step "
                    f"to the swept chain, or harvest a key that holds a "
                    f"MetricScore or a scalar."
                ) from exc
        return values

    def __call__(self, results: Results) -> Results:
        total = len(self.over)
        heartbeat = max(1, total // 10)
        columns: dict[str, dict[Any, float]] = {key: {} for key in self.read}
        logger.info("Sweep: %d item(s), harvesting %s", total, self.read)

        for index, item in enumerate(self.over, start=1):
            forked = results.copy()
            for step in self._steps_for(item):
                forked = step(forked)
            for key, value in self._harvest(forked, item).items():
                columns[key][item] = value
            if index % heartbeat == 0 or index == total:
                logger.info("Sweep: %d/%d", index, total)

        results[self.output_key] = SweepResult(
            columns=columns,
            primary=self.read[0],
            metadata={
                "n_items": total,
                "read": list(self.read),
                "items": [str(item) for item in self.over],
                "steps": self._chain_names,
            },
        )
        return results
