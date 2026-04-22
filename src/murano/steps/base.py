"""Step protocol — the contract every pipeline step must follow."""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import UnionType
from typing import Mapping, TypeAlias

from murano.results import Results

ExpectedType: TypeAlias = type | tuple[type, ...]


def _normalize_expected_types(expected: ExpectedType | None) -> tuple[type, ...]:
    if expected is None:
        return ()
    if isinstance(expected, tuple):
        return expected
    if isinstance(expected, UnionType):
        return tuple(t for t in expected.__args__ if isinstance(t, type))
    return (expected,)


def _format_expected_types(expected: ExpectedType | None) -> str:
    normalized = _normalize_expected_types(expected)
    if not normalized:
        return "unknown"
    return " | ".join(t.__name__ for t in normalized)


def _types_compatible(
    available: ExpectedType | None,
    expected: ExpectedType | None,
) -> bool:
    available_types = _normalize_expected_types(available)
    expected_types = _normalize_expected_types(expected)
    if not available_types or not expected_types:
        return True
    return all(issubclass(t, expected_types) for t in available_types)


class Step(ABC):
    """Base class for all pipeline steps.

    Every step declares what it reads from and writes to the Results object.
    Pipeline uses these declarations for validation before running.

    Subclasses must:
        1. Set ``reads`` and ``writes`` class attributes.
        2. Implement ``__call__(results) -> results``.

    Example:
        class MyStep(Step):
            reads = ["dataset"]
            writes = ["record"]

            def __call__(self, results: Results) -> Results:
                ...
                return results
    """

    reads: list[str] = []
    writes: list[str] = []
    read_types: dict[str, ExpectedType] = {}
    write_types: dict[str, ExpectedType] = {}

    @abstractmethod
    def __call__(self, results: Results) -> Results: ...

    def expected_read_types(
        self,
        results: Results | None = None,
        available_types: Mapping[str, ExpectedType] | None = None,
    ) -> Mapping[str, ExpectedType]:
        return self.read_types

    def expected_write_types(
        self,
        results: Results | None = None,
        available_types: Mapping[str, ExpectedType] | None = None,
    ) -> Mapping[str, ExpectedType]:
        return self.write_types

    def validate(self, results: Results) -> None:
        """Check that all required keys are present in results.

        Raises:
            KeyError: If a required key is missing.
        """
        expected_types = self.expected_read_types(results=results)
        for key in self.reads:
            if key not in results:
                available = list(results.keys())
                raise KeyError(
                    f"Step {type(self).__name__} requires '{key}' in results, "
                    f"but it is missing. Available keys: {available}. "
                    f"Did you forget a step that produces '{key}'?"
                )
            expected_type = expected_types.get(key)
            if expected_type is not None and not isinstance(
                results[key],
                _normalize_expected_types(expected_type),
            ):
                raise TypeError(
                    f"Step {type(self).__name__} requires '{key}' to be "
                    f"{_format_expected_types(expected_type)}, but got "
                    f"{type(results[key]).__name__}."
                )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(reads={self.reads}, writes={self.writes})"
