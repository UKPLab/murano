"""Results container — the shared state passed between pipeline steps."""

from __future__ import annotations

from typing import Any


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
        return self._data.keys()

    def get(self, key: str, default: Any = None) -> Any:
        """Safe key access with a default value."""
        return self._data.get(key, default)

    def copy(self) -> Results:
        """Shallow copy for fan-out pipelines."""
        r = Results()
        r._data = self._data.copy()
        return r

    def save(
        self,
        output_dir: str = "murano_outputs",
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
