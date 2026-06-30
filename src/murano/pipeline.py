"""Pipeline: declarative composition of steps."""

from __future__ import annotations

from murano.logging import logger
from murano.results import Results
from murano.steps.base import (
    Step,
    _format_expected_types,
    _types_compatible,
)


def _warn_on_write_collision(
    step: Step, name: str, produced_by: dict[str, str]
) -> None:
    """Warn when a step writes a key an earlier step already produced.

    A later write silently overwrites the earlier value, which is the quiet
    failure mode of running the same step twice with default output keys (for
    example a clean and a corrupt ``Logits`` on one pipeline). The contract is
    not an error: overwriting can be intentional, so this only warns and names
    both producers.
    """
    for key in step.writes:
        prior = produced_by.get(key)
        if prior is not None:
            logger.warning(
                "Step %s writes %r, which %s already produced; the earlier value "
                "will be overwritten. Pass a distinct output key to keep both.",
                name,
                key,
                prior,
            )
        produced_by[key] = name


class Pipeline:
    """Sequential composition of steps.

    Tier 1 API: Pipeline([Load(...), Record(...), Train(...)]).run()

    Each step is a callable: step(results) -> results.
    Steps that inherit from Step get automatic pre-flight validation.
    """

    def __init__(self, steps: list):
        self.steps = steps

    def run(self, results: Results | None = None) -> Results:
        """Execute every step in order, returning the final Results.

        Each step that inherits from Step is validated against the running
        Results before it runs.

        Args:
            results: Starting Results object. A fresh empty Results is
                created when None.

        Returns:
            The Results object after the last step has run.
        """
        results = results or Results()
        produced_by: dict[str, str] = {}
        for step in self.steps:
            name = type(step).__name__
            if isinstance(step, Step):
                step.validate(results)
                _warn_on_write_collision(step, name, produced_by)
            logger.info("Running step: %s", name)
            results = step(results)
        return results

    def validate(self) -> list[str]:
        """Dry-run validation: check that step reads/writes chain correctly.

        Returns:
            List of all keys that will be available after the pipeline runs.

        Raises:
            KeyError: If a step reads a key that no prior step writes.
        """
        available: set[str] = set()
        available_types: dict[str, type | tuple[type, ...]] = {}
        produced_by: dict[str, str] = {}
        for step in self.steps:
            if isinstance(step, Step):
                read_types = step.expected_read_types(available_types=available_types)
                for key in step.reads:
                    if key not in available:
                        raise KeyError(
                            f"Step {type(step).__name__} requires '{key}', "
                            f"but no prior step writes it. "
                            f"Available at this point: {sorted(available)}"
                        )
                    if key in read_types and key in available_types:
                        if not _types_compatible(available_types[key], read_types[key]):
                            raise TypeError(
                                f"Step {type(step).__name__} requires '{key}' to be "
                                f"{_format_expected_types(read_types[key])}, but the "
                                f"pipeline provides {_format_expected_types(available_types[key])}."
                            )
                _warn_on_write_collision(step, type(step).__name__, produced_by)
                available.update(step.writes)
                available_types.update(
                    step.expected_write_types(available_types=available_types)
                )
        return sorted(available)

    def __repr__(self) -> str:
        step_names = [type(s).__name__ for s in self.steps]
        return f"Pipeline([{', '.join(step_names)}])"
