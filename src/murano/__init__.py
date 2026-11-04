"""Murano: mechanistic interpretability pipeline."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from murano.logging import setup_logging

try:
    __version__ = version("murano-interp")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

if TYPE_CHECKING:
    from murano.artifacts import (
        ComponentSelection,
        MetricScore,
        GenerationComparison,
        MetricComparison,
        NeuronSelection,
        PromptBatch,
        SweepResult,
    )
    from murano.dataset import CleanCorruptDataset, LabeledDataset, MuranoDataset
    from murano.io import (
        load_activation_store,
        load_attention,
        load_metric_score,
        load_labeled_activation_store,
        load_logit_lens,
        load_race,
        load_sae_activations,
        load_sae_examples,
        load_sae_labels,
        load_steering,
        save_ablated_model,
        save_results,
    )
    from murano.model import MuranoModel
    from murano.model import MuranoModel as Model
    from murano.nodes import (
        AddressLike,
        Edge,
        Neuron,
        Node,
        NodeDict,
        NodeSet,
        Side,
    )
    from murano.pipeline import Pipeline
    from murano.results import Results
    from murano.steps.attention import AttentionResult
    from murano.steps.base import Step
    from murano.steps.logit_attribution import (
        LogitAttribution,
        LogitAttributionResult,
    )
    from murano.steps.logit_lens import LogitLens, LogitLensResult
    from murano.steps.logits import Logits
    from murano.steps.race import RACE, RACEResult
    from murano.steps.select import SelectComponents
    from murano.steps.sae import (
        SAEActivationStore,
        SAEEncode,
        SAEFeatureExamples,
        SAEFeatureLabel,
        SAEFeatureLabels,
        SAEModel,
        SAETopActivations,
        sae_steer,
        top_sae_features_for_tokens,
        top_sae_features_per_prompt,
    )


_LAZY_ATTRS = {
    "MuranoModel": ("murano.model", "MuranoModel"),
    "Model": ("murano.model", "MuranoModel"),
    "Node": ("murano.nodes", "Node"),
    "Side": ("murano.nodes", "Side"),
    "Edge": ("murano.nodes", "Edge"),
    "Neuron": ("murano.nodes", "Neuron"),
    "NodeSet": ("murano.nodes", "NodeSet"),
    "NodeDict": ("murano.nodes", "NodeDict"),
    "AddressLike": ("murano.nodes", "AddressLike"),
    "PromptBatch": ("murano.artifacts", "PromptBatch"),
    "GenerationComparison": ("murano.artifacts", "GenerationComparison"),
    "MetricComparison": ("murano.artifacts", "MetricComparison"),
    "MetricScore": ("murano.artifacts", "MetricScore"),
    "ComponentSelection": ("murano.artifacts", "ComponentSelection"),
    "SweepResult": ("murano.artifacts", "SweepResult"),
    "NeuronSelection": ("murano.artifacts", "NeuronSelection"),
    "SelectComponents": ("murano.steps.select", "SelectComponents"),
    "MuranoDataset": ("murano.dataset", "MuranoDataset"),
    "LabeledDataset": ("murano.dataset", "LabeledDataset"),
    "CleanCorruptDataset": ("murano.dataset", "CleanCorruptDataset"),
    "Pipeline": ("murano.pipeline", "Pipeline"),
    "Results": ("murano.results", "Results"),
    "Step": ("murano.steps.base", "Step"),
    "LogitAttribution": ("murano.steps.logit_attribution", "LogitAttribution"),
    "LogitAttributionResult": (
        "murano.steps.logit_attribution",
        "LogitAttributionResult",
    ),
    "LogitLens": ("murano.steps.logit_lens", "LogitLens"),
    "LogitLensResult": ("murano.steps.logit_lens", "LogitLensResult"),
    "AttentionResult": ("murano.steps.attention", "AttentionResult"),
    "Logits": ("murano.steps.logits", "Logits"),
    "RACE": ("murano.steps.race", "RACE"),
    "RACEResult": ("murano.steps.race", "RACEResult"),
    "SAEEncode": ("murano.steps.sae", "SAEEncode"),
    "SAEActivationStore": ("murano.steps.sae", "SAEActivationStore"),
    "SAEModel": ("murano.steps.sae", "SAEModel"),
    "SAETopActivations": ("murano.steps.sae", "SAETopActivations"),
    "sae_steer": ("murano.steps.sae", "sae_steer"),
    "top_sae_features_for_tokens": ("murano.steps.sae", "top_sae_features_for_tokens"),
    "top_sae_features_per_prompt": ("murano.steps.sae", "top_sae_features_per_prompt"),
    "SAEFeatureExamples": ("murano.steps.sae", "SAEFeatureExamples"),
    "SAEFeatureLabel": ("murano.steps.sae", "SAEFeatureLabel"),
    "SAEFeatureLabels": ("murano.steps.sae", "SAEFeatureLabels"),
    "save_results": ("murano.io", "save_results"),
    "load_steering": ("murano.io", "load_steering"),
    "load_logit_lens": ("murano.io", "load_logit_lens"),
    "load_race": ("murano.io", "load_race"),
    "load_attention": ("murano.io", "load_attention"),
    "load_activation_store": ("murano.io", "load_activation_store"),
    "load_labeled_activation_store": ("murano.io", "load_labeled_activation_store"),
    "load_sae_activations": ("murano.io", "load_sae_activations"),
    "load_sae_examples": ("murano.io", "load_sae_examples"),
    "load_sae_labels": ("murano.io", "load_sae_labels"),
    "load_metric_score": ("murano.io", "load_metric_score"),
    "save_ablated_model": ("murano.io", "save_ablated_model"),
}


__all__ = [
    "MuranoModel",
    "Model",
    "Node",
    "Side",
    "Edge",
    "Neuron",
    "NodeSet",
    "NodeDict",
    "AddressLike",
    "PromptBatch",
    "GenerationComparison",
    "MetricComparison",
    "MetricScore",
    "ComponentSelection",
    "SweepResult",
    "NeuronSelection",
    "SelectComponents",
    "MuranoDataset",
    "LabeledDataset",
    "CleanCorruptDataset",
    "Pipeline",
    "Results",
    "Step",
    "LogitAttribution",
    "LogitAttributionResult",
    "LogitLens",
    "LogitLensResult",
    "AttentionResult",
    "Logits",
    "RACE",
    "RACEResult",
    "SAEActivationStore",
    "SAEEncode",
    "SAEFeatureExamples",
    "SAEFeatureLabel",
    "SAEFeatureLabels",
    "SAEModel",
    "SAETopActivations",
    "sae_steer",
    "top_sae_features_for_tokens",
    "top_sae_features_per_prompt",
    "__version__",
    "setup_logging",
    "save_results",
    "load_steering",
    "load_logit_lens",
    "load_race",
    "load_attention",
    "load_activation_store",
    "load_labeled_activation_store",
    "load_sae_activations",
    "load_sae_examples",
    "load_sae_labels",
    "load_metric_score",
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
