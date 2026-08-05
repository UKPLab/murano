"""Murano pipeline steps."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from murano.steps.base import Step
from murano.steps.load import Load
from murano.steps.prompts import LoadPrompts
from murano.steps.paired import LoadPaired
from murano.steps.record import Record
from murano.steps.save import Save
from murano.steps.train import SteeringVector
from murano.steps.intervene import Intervene
from murano.steps.ablate import Ablate
from murano.steps.patch import Patch
from murano.steps.path_patch import PathPatch
from murano.steps.select import SelectComponents
from murano.steps.sweep import Sweep
from murano.steps.attention import (
    AblateAttention,
    AttentionResult,
    RecordAttention,
    ov_circuit,
)
from murano.steps.weight_ablation import WeightAblation
from murano.steps.evaluate import GenerationMetric
from murano.steps.probe import Probe
from murano.steps.plot import Plot
from murano.steps.logit_lens import LogitLens, LogitLensResult
from murano.steps.logit_attribution import LogitAttribution, LogitAttributionResult
from murano.steps.logits import Logits
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
from murano.steps.metrics import (
    CrossEntropyLossStep,
    AccuracyStep,
    ComparisonComputationStep,
    LogitDiffStep,
    KLDivergenceStep,
    AnswerLogProbStep,
    AnswerRankStep,
    RecoveredMetricStep,
)

if TYPE_CHECKING:
    from murano.steps.fega import (
        FEGAComputeEffect as FEGAComputeEffect,
        FEGADataPrep as FEGADataPrep,
        FEGAGeometryMetrics as FEGAGeometryMetrics,
        FEGAGeometryReporting as FEGAGeometryReporting,
        FEGAStability as FEGAStability,
        FEGAVisualize as FEGAVisualize,
        FEGAVMF as FEGAVMF,
        fega_steps as fega_steps,
    )

_FEGA_ATTRS = {
    "FEGADataPrep",
    "FEGAComputeEffect",
    "FEGAGeometryMetrics",
    "FEGAVMF",
    "FEGAStability",
    "FEGAGeometryReporting",
    "FEGAVisualize",
    "fega_steps",
}

__all__ = [
    "Step",
    "Load",
    "LoadPrompts",
    "LoadPaired",
    "Record",
    "Save",
    "SteeringVector",
    "Intervene",
    "Ablate",
    "Patch",
    "PathPatch",
    "SelectComponents",
    "Sweep",
    "RecordAttention",
    "AblateAttention",
    "AttentionResult",
    "ov_circuit",
    "WeightAblation",
    "GenerationMetric",
    "Probe",
    "Plot",
    "LogitLens",
    "LogitLensResult",
    "LogitAttribution",
    "LogitAttributionResult",
    "Logits",
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
    "CrossEntropyLossStep",
    "AccuracyStep",
    "ComparisonComputationStep",
    "LogitDiffStep",
    "KLDivergenceStep",
    "AnswerLogProbStep",
    "AnswerRankStep",
    "RecoveredMetricStep",
]


def __getattr__(name: str) -> Any:
    """Lazily expose FEGA steps without making its extra a core dependency."""
    # Keep the existing eager core-step imports while deferring only the optional app.
    if name not in _FEGA_ATTRS:
        raise AttributeError(f"module 'murano.steps' has no attribute {name!r}")
    value = getattr(import_module("murano.steps.fega"), name)
    globals()[name] = value
    return value
