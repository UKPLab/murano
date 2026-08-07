"""Typed in-memory artifacts exchanged by notebook-local FEGA steps."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .contexts import FEGAContext, FEGAContextBatch
from .effects import normalize_delta_rows
from .geometry import GeometryMetrics
from .reporting import GeometryClassification, GeometryRecord
from .vmf import VMFCandidateEvidence, VMFSelection


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
class FEGACuratedRowMetadata:
    """One saved effect row and its prompt metadata."""

    source_context_index: int
    attribute_label: str | None = None
    pair_role: str | None = None
    pair_index: int | None = None
    stability_group: str | None = None
    prompt: str | None = None
    source_token_ids: tuple[int, ...] = ()
    logical_target_position: int | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> FEGACuratedRowMetadata:
        """Load one row from `inputs.json`."""
        # Normalize optional fields used by the live example.
        token_ids = payload.get("source_token_ids") or ()
        return cls(
            source_context_index=int(payload["source_context_index"]),
            attribute_label=_optional_str(payload.get("attribute_label")),
            pair_role=_optional_str(payload.get("pair_role")),
            pair_index=_optional_int(payload.get("pair_index")),
            stability_group=_optional_str(payload.get("stability_group")),
            prompt=_optional_str(payload.get("prompt")),
            source_token_ids=tuple(int(token) for token in token_ids),
            logical_target_position=_optional_int(
                payload.get("logical_target_position")
            ),
        )

    def as_effect_context(self) -> FEGAContext:
        """Project compact row metadata onto the existing effect-store context type."""
        # Preserve group/pair metadata exactly while using inert placeholders off the
        # live prompt path, where prompt text and token IDs are unavailable.
        prompt = self.prompt or f"source_context:{self.source_context_index}"
        input_ids = self.source_token_ids or (0,)
        target_position = (
            0
            if self.logical_target_position is None
            else int(self.logical_target_position)
        )
        return FEGAContext(
            index=int(self.source_context_index),
            prompt=prompt,
            input_ids=input_ids,
            target_position=target_position,
            attribute_label=self.attribute_label,
            pair_role=self.pair_role,
            pair_index=self.pair_index,
            group_label=self.stability_group,
        )


@dataclass(frozen=True)
class FEGACuratedFeatureInput:
    """One feature's ordered rows in the compact artifact bundle."""

    feature_id: int
    source_run: str
    sae_kind: str
    row_start: int
    row_stop: int
    rows: tuple[FEGACuratedRowMetadata, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> FEGACuratedFeatureInput:
        """Load one feature record from `inputs.json`."""
        # Parse the rows before constructing the immutable record.
        rows = tuple(
            FEGACuratedRowMetadata.from_mapping(row) for row in payload.get("rows", ())
        )
        return cls(
            feature_id=int(payload["feature_id"]),
            source_run=str(payload["source_run"]),
            sae_kind=str(payload["sae_kind"]),
            row_start=int(payload["row_start"]),
            row_stop=int(payload["row_stop"]),
            rows=rows,
        )

    @property
    def row_count(self) -> int:
        """Return the declared size of the feature's raw-delta block."""
        # Derive the count from the stored slice boundaries.
        return int(self.row_stop - self.row_start)


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
    feature_ids: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ordered_feature_ids(self) -> tuple[int, ...]:
        """Return the stable feature order chosen when the store was built."""
        # Respect the saved feature order when present; otherwise preserve insertion.
        return self.feature_ids or tuple(self.features)


@dataclass(frozen=True)
class FEGAGeometryResult:
    """Per-feature point geometry metrics in ascending feature order."""

    features: dict[int, GeometryMetrics]


@dataclass(frozen=True)
class FEGAVMFResult:
    """Per-feature dense vMF selection and assignment stability."""

    features: dict[int, VMFSelection]
    assignment_stability: dict[int, float | None]
    fega_config: dict[str, Any]
    fit_status: dict[int, str] = field(default_factory=dict)
    failures: dict[int, tuple[VMFCandidateEvidence, ...]] = field(default_factory=dict)
    assignment_stability_counts: dict[int, dict[str, int]] = field(default_factory=dict)


@dataclass(frozen=True)
class FEGAStabilityResult:
    """Selected-family stability evidence keyed by feature ID."""

    features: dict[int, dict[str, Any]]


@dataclass(frozen=True)
class FEGAReportingResult:
    """Final FEGA classifications in deterministic feature order."""

    records: dict[int, GeometryRecord]
    features: dict[int, GeometryClassification]
    feature_ids: tuple[int, ...]


@dataclass(frozen=True)
class FEGARenderCapture:
    """One immutable local renderer invocation captured for comparison."""

    renderer: str
    coordinates: np.ndarray
    kwargs: dict[str, Any]
    point_colors: tuple[str, ...] | None = None
    assignments: tuple[int, ...] | None = None
    selected_k: int | None = None


@dataclass(frozen=True)
class FEGAVisualizationResult:
    """Rendered FEGA figures and explicit family skips."""

    files: tuple[Path, ...]
    skipped: dict[str, str] = field(default_factory=dict)
    captures: dict[int, dict[str, FEGARenderCapture]] = field(default_factory=dict)


def build_effect_store_from_raw_cloud(
    delta_rows: Tensor | np.ndarray,
    features: Sequence[FEGACuratedFeatureInput],
    gram: Tensor,
    *,
    tau_zero: float = 1e-12,
) -> FEGAEffectStore:
    """Convert saved raw effects and row metadata into an effect store.

    Args:
        delta_rows: Concatenated `ablated - baseline` residual rows.
        features: Ordered compact feature records aligned to `delta_rows`.
        gram: Runtime logit-space Gram matrix derived from the active model.
        tau_zero: Near-zero Gram-magnitude threshold.

    Returns:
        A typed `FEGAEffectStore` in the same stable feature order as `features`.

    Raises:
        ValueError: If the row blocks are inconsistent with the compact metadata or
            the raw cloud does not match the runtime Gram width.
        FloatingPointError: If normalization sees a persistent negative quadratic
            form under the provided Gram matrix.
    """
    # Validate the shared tensor shape before feature-wise normalization.
    cloud = torch.as_tensor(delta_rows, dtype=torch.float32)
    if cloud.ndim != 2:
        raise ValueError("delta_rows must be a rank-2 raw effect cloud")
    if gram.shape != (cloud.shape[1], cloud.shape[1]):
        raise ValueError("gram must match the raw effect width exactly")

    feature_map: dict[int, FEGAFeatureEffects] = {}
    feature_ids: list[int] = []
    for feature in features:
        if feature.row_start < 0 or feature.row_stop < feature.row_start:
            raise ValueError("raw-cloud row offsets must be monotone and non-negative")
        if feature.row_stop > cloud.shape[0]:
            raise ValueError("raw-cloud row offsets exceed the concatenated delta rows")
        if feature.row_count != len(feature.rows):
            raise ValueError("feature row metadata does not match its declared block")
        rows = cloud[feature.row_start : feature.row_stop]
        directions, magnitudes, mask = normalize_delta_rows(rows, gram, tau_zero)
        retained_rows = tuple(
            row for row, keep in zip(feature.rows, mask.tolist(), strict=True) if keep
        )
        retained_count = int(mask.sum().item())
        feature_ids.append(feature.feature_id)
        feature_map[feature.feature_id] = FEGAFeatureEffects(
            feature_id=feature.feature_id,
            directions=directions.cpu(),
            magnitudes=magnitudes.cpu(),
            context_indices=tuple(row.source_context_index for row in retained_rows),
            feature_activations=torch.full(
                (retained_count,), float("nan"), dtype=torch.float32
            ),
            retained_mask=tuple(bool(value) for value in mask.tolist()),
            contexts=tuple(row.as_effect_context() for row in retained_rows),
        )
    return FEGAEffectStore(
        features=feature_map,
        gram=gram,
        feature_ids=tuple(feature_ids),
        metadata={
            "effect_sign": "ablated-minus-baseline",
            "tau_zero": float(tau_zero),
        },
    )


def _optional_int(value: Any) -> int | None:
    """Normalize an optional integer without inventing a missing value."""
    # Preserve missing values as `None`.
    return None if value is None else int(value)


def _optional_str(value: Any) -> str | None:
    """Normalize an optional label without introducing a compatibility layer."""
    # Preserve missing labels as `None`.
    return None if value is None else str(value)
