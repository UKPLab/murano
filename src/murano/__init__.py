"""Murano: mechanistic interpretability pipeline."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from murano.logging import setup_logging

try:
    __version__ = version("murano")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

if TYPE_CHECKING:
    from murano.artifacts import GenerationComparison, MetricResult, PromptBatch
    from murano.dataset import LabeledDataset, MuranoDataset
    from murano.evaluation import compliance_rate
    from murano.io import load_steering, save_ablated_model, save_results
    from murano.model import MuranoModel
    from murano.model import MuranoModel as Model
    from murano.pipeline import Pipeline
    from murano.results import Results
    from murano.steps.base import Step


_LAZY_ATTRS = {
    "MuranoModel": ("murano.model", "MuranoModel"),
    "Model": ("murano.model", "MuranoModel"),
    "PromptBatch": ("murano.artifacts", "PromptBatch"),
    "GenerationComparison": ("murano.artifacts", "GenerationComparison"),
    "MetricResult": ("murano.artifacts", "MetricResult"),
    "MuranoDataset": ("murano.dataset", "MuranoDataset"),
    "LabeledDataset": ("murano.dataset", "LabeledDataset"),
    "Pipeline": ("murano.pipeline", "Pipeline"),
    "Results": ("murano.results", "Results"),
    "Step": ("murano.steps.base", "Step"),
    "compliance_rate": ("murano.evaluation", "compliance_rate"),
    "save_results": ("murano.io", "save_results"),
    "load_steering": ("murano.io", "load_steering"),
    "save_ablated_model": ("murano.io", "save_ablated_model"),
}


__all__ = [
    "MuranoModel",
    "Model",
    "PromptBatch",
    "GenerationComparison",
    "MetricResult",
    "MuranoDataset",
    "LabeledDataset",
    "Pipeline",
    "Results",
    "Step",
    "__version__",
    "compliance_rate",
    "setup_logging",
    "save_results",
    "load_steering",
    "save_ablated_model",
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_ATTRS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'murano' has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    if name in {"MuranoModel", "Model"}:
        globals()["MuranoModel"] = value
        globals()["Model"] = value
    else:
        globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
