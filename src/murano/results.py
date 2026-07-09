"""Results container: the shared state passed between pipeline steps."""

from __future__ import annotations

from typing import Any

from murano.keys import DEFAULT_OUTPUT_DIR


class Results:
    """Dict-like container for pipeline step outputs.

    Every step reads from and writes to this object.
    The core contract: step(results) -> results.
    """

    def __init__(self):
        self._data: dict[str, Any] = {}

    def __getitem__(self, key: str) -> Any:
        if key not in self._data:
            available = list(self._data.keys())
            raise KeyError(
                f"'{key}' not found in results. "
                f"Available keys: {available}. "
                f"Did you forget a step that produces '{key}'?"
            )
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self):
        """Return a view of the stored result keys."""
        return self._data.keys()

    def get(self, key: str, default: Any = None) -> Any:
        """Safe key access with a default value.

        Args:
            key: Result key to look up.
            default: Value returned when ``key`` is not present.

        Returns:
            The stored value, or ``default`` if no value is stored under ``key``.
        """
        return self._data.get(key, default)

    def copy(self) -> Results:
        """Return a shallow copy for fan-out pipelines.

        Returns:
            A new Results object whose internal mapping is a shallow copy of
            the original; values themselves are not duplicated.
        """
        r = Results()
        r._data = self._data.copy()
        return r

    def save(
        self,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        run_name: str | None = None,
        model_id: str = "",
    ):
        """Save all results to disk.

        Args:
            output_dir: Base directory for outputs.
            run_name: Optional subdirectory name inside ``output_dir``.
            model_id: Model identifier for metadata.

        Returns:
            Path to the run directory.
        """
        from murano.io import save_results

        return save_results(
            self,
            output_dir=output_dir,
            model_id=model_id,
            run_name=run_name,
        )

    def __repr__(self) -> str:
        return f"Results({list(self._data.keys())})"
