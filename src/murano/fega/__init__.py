"""Native Feature-Effect Geometry Analysis primitives and artifacts."""

# ruff: noqa: E402

from murano._optional import require_optional

require_optional("fega")

from murano.fega.artifacts import (
    FEGADataPrepResult,
    FEGAEffectStore,
    FEGAFeatureEffects,
    FEGAGeometryResult,
    FEGAReportingResult,
    FEGAStabilityResult,
    FEGAVMFResult,
    FEGAVisualizationResult,
)
from murano.fega.config import FEGAConfig
from murano.fega.dictionary_sae import DictionarySAEModel

__all__ = [
    "FEGAConfig",
    "DictionarySAEModel",
    "FEGADataPrepResult",
    "FEGAEffectStore",
    "FEGAFeatureEffects",
    "FEGAGeometryResult",
    "FEGAVMFResult",
    "FEGAStabilityResult",
    "FEGAReportingResult",
    "FEGAVisualizationResult",
]
