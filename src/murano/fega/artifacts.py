"""Typed in-memory artifacts exchanged by native FEGA pipeline phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch import Tensor

from murano.fega.contexts import FEGAContext, FEGAContextBatch
from murano.fega.geometry import GeometryMetrics
from murano.fega.reporting import GeometryClassification
from murano.fega.vmf import VMFCandidateEvidence, VMFSelection


@dataclass(frozen=True)
class FEGADataPrepResult:
    """Encoded prompts, full-reconstruction baselines, and selected contexts."""

    contexts: FEGAContextBatch
    feature_ids: tuple[int, ...]
    latent_codes: Tensor
    baseline_readouts: Tensor
    selected_context_indices: dict[int, tuple[int, ...]]
    release: str
    sae_id: str
    hook_layer: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FEGAFeatureEffects:
    """Effect directions normalized under the logit-space Gram metric."""

    feature_id: int
    directions: Tensor
    magnitudes: Tensor
    context_indices: tuple[int, ...]
    feature_activations: Tensor
    retained_mask: tuple[bool, ...]
    contexts: tuple[FEGAContext, ...] = ()


@dataclass(frozen=True)
class FEGAEffectStore:
    """All retained feature effects and their logit-space metric."""

    features: dict[int, FEGAFeatureEffects]
    gram: Tensor
    unembedding_fingerprint: str
    metadata: dict[str, Any] = field(default_factory=dict)
    analysis_id: str = ""


@dataclass(frozen=True)
class FEGAGeometryResult:
    """Per-feature point geometry metrics in ascending feature order."""

    features: dict[int, GeometryMetrics]
    analysis_id: str = ""


@dataclass(frozen=True)
class FEGAVMFResult:
    """Per-feature dense vMF selection and assignment stability."""

    features: dict[int, VMFSelection]
    assignment_stability: dict[int, float | None]
    unembedding_fingerprint: str
    analysis_id: str = ""
    fit_status: dict[int, str] = field(default_factory=dict)
    failures: dict[int, tuple[VMFCandidateEvidence, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class FEGAStabilityResult:
    """Selected-family stability evidence keyed by feature ID."""

    features: dict[int, dict[str, Any]]
    analysis_id: str = ""


@dataclass(frozen=True)
class FEGAReportingResult:
    """Final FEGA classifications in deterministic feature order."""

    features: dict[int, GeometryClassification]
    feature_ids: tuple[int, ...]
    analysis_id: str = ""


@dataclass(frozen=True)
class FEGAVisualizationResult:
    """Rendered FEGA figures and explicit family skips."""

    files: tuple[Path, ...]
    skipped: dict[str, str] = field(default_factory=dict)
    analysis_id: str = ""
