"""Explicit phase checkpoints for FEGA pipelines.

Checkpoint files use ``torch.save`` and therefore contain pickle data.  Load
only checkpoints produced by trusted local runs; untrusted files can execute
code while being deserialized.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from murano import keys
from murano.artifacts import PromptBatch
from murano.fega.artifacts import FEGADataPrepResult, FEGAEffectStore
from murano.fega.config import FEGAConfig
from murano.results import Results
from murano.steps.base import ExpectedType, Step


FEGA_PHASES = (
    "data_prep",
    "compute_effect",
    "geometry_metrics",
    "vmf",
    "stability",
    "geometry_reporting",
    "visualize",
)
FEGA_LABEL_VERSION = 1
_PROMPT_METADATA_FIELDS = (
    "attribute_label",
    "pair_role",
    "pair_index",
    "group_label",
)


def _validate_prompt_identity(
    prompts: PromptBatch, prepared: FEGADataPrepResult
) -> None:
    """Require a data-prep checkpoint to match current ordered prompt rows."""

    # Compare raw prompt order and provenance without introducing a hash inventory.
    contexts = prepared.contexts.contexts
    if tuple(prompts.prompts) != tuple(context.prompt for context in contexts):
        raise ValueError("FEGA data_prep checkpoint prompt rows do not match")
    if prompts.source != prepared.contexts.source:
        raise ValueError("FEGA data_prep checkpoint prompt source does not match")

    # Bind optional aligned FEGA fields that can change grouped scientific results.
    metadata = prompts.metadata.get("fega", {})
    if not isinstance(metadata, dict):
        raise ValueError("PromptBatch.metadata['fega'] must be a mapping")
    for field_name in _PROMPT_METADATA_FIELDS:
        values = metadata.get(field_name)
        actual = (None,) * len(contexts) if values is None else tuple(values)
        expected = tuple(getattr(context, field_name) for context in contexts)
        if actual != expected:
            raise ValueError(
                f"FEGA data_prep checkpoint {field_name} rows do not match"
            )


def effect_resume_metadata(
    effects: FEGAEffectStore, config: FEGAConfig
) -> dict[str, Any]:
    """Identify the effect computation and settings eligible for partial resume."""
    # A fresh compute-effect pass gets a new opaque ID; downstream work propagates it.
    if not effects.analysis_id:
        raise ValueError("FEGA effects are missing their analysis identity")
    identity = effects.metadata.get("identity", {})
    return {
        "identity": identity if isinstance(identity, dict) else {},
        "config": asdict(config),
        "analysis_id": effects.analysis_id,
        "unembedding_fingerprint": effects.unembedding_fingerprint,
    }


def phase_checkpoint_path(checkpoint_dir: str | Path, phase: str) -> Path:
    """Return the canonical ``<checkpoint_dir>/<phase>.pt`` checkpoint path."""
    # Reject misspelled phases before constructing a non-canonical path.
    if phase not in FEGA_PHASES:
        raise ValueError(f"Unknown FEGA phase {phase!r}; expected one of {FEGA_PHASES}")
    return Path(checkpoint_dir) / f"{phase}.pt"


def save_phase_checkpoint(
    checkpoint_dir: str | Path,
    phase: str,
    artifact: Any,
    metadata: dict[str, Any],
) -> Path:
    """Atomically save one trusted-local FEGA phase artifact and its metadata.

    Args:
        checkpoint_dir: Directory that owns the canonical phase checkpoint.
        phase: One of :data:`FEGA_PHASES`.
        artifact: Phase result accepted by :func:`torch.save`.
        metadata: Run metadata stored beside the artifact.

    Returns:
        The canonical checkpoint path.
    """
    # Write beside the destination so os.replace remains an atomic rename.
    path = phase_checkpoint_path(checkpoint_dir, phase)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{phase}.", suffix=".tmp", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        torch.save(
            {"phase": phase, "artifact": artifact, "metadata": metadata},
            temporary_path,
        )
        os.replace(temporary_path, path)
    finally:
        # Remove an incomplete temporary file when serialization fails.
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return path


def load_phase_checkpoint(
    checkpoint_dir: str | Path, phase: str
) -> tuple[Any, dict[str, Any]]:
    """Load one trusted-local checkpoint and verify its exact FEGA phase.

    ``torch.load`` uses pickle-backed deserialization.  This function is only
    safe for files created by trusted local runs.

    Raises:
        FileNotFoundError: If the canonical phase checkpoint does not exist.
        ValueError: If the file records a different phase.
    """
    # Resolve the exact canonical file and fail with phase-aware context.
    path = phase_checkpoint_path(checkpoint_dir, phase)
    if not path.exists():
        raise FileNotFoundError(f"Missing FEGA checkpoint for phase {phase!r}: {path}")
    payload = torch.load(path, weights_only=False)
    stored_phase = payload.get("phase")
    if stored_phase != phase:
        raise ValueError(
            f"FEGA checkpoint phase mismatch at {path}: expected {phase!r}, "
            f"found {stored_phase!r}"
        )
    return payload["artifact"], payload["metadata"]


class FEGAWriteCheckpoint(Step):
    """Pipeline step that checkpoints one configured result without new outputs."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        phase: str,
        key: str,
        artifact_type: type[Any],
        metadata: dict[str, Any],
    ) -> None:
        """Configure the exact phase, result key, type, and checkpoint metadata."""
        # Expose instance-specific dependencies to Pipeline.validate.
        self.checkpoint_dir = Path(checkpoint_dir)
        self.phase = phase
        self.key = key
        self.artifact_type = artifact_type
        self.metadata = metadata
        self.reads = [key]
        self.writes = []

    def expected_read_types(
        self,
        results: Results | None = None,
        available_types: Mapping[str, ExpectedType] | None = None,
    ) -> Mapping[str, ExpectedType]:
        """Return the configured input type for static pipeline validation."""
        # The configured key is the step's only scientific dependency.
        return {self.key: self.artifact_type}

    def expected_write_types(
        self,
        results: Results | None = None,
        available_types: Mapping[str, ExpectedType] | None = None,
    ) -> Mapping[str, ExpectedType]:
        """Return no output types because checkpointing adds no scientific key."""
        # Checkpoint persistence is a side effect, not a Results output.
        return {}

    def __call__(self, results: Results) -> Results:
        """Save the configured result value and return the unchanged Results."""
        # Persist the single declared input without mutating pipeline results.
        save_phase_checkpoint(
            self.checkpoint_dir,
            self.phase,
            results[self.key],
            self.metadata,
        )
        return results


class FEGALoadCheckpoint(Step):
    """Pipeline step that restores one configured phase artifact into Results."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        phase: str,
        key: str,
        artifact_type: type[Any],
        expected_metadata: dict[str, Any] | None = None,
        bind_prompts: bool = False,
    ) -> None:
        """Configure the exact phase checkpoint and result key/type it provides."""
        # Expose instance-specific production information to Pipeline.validate.
        self.checkpoint_dir = Path(checkpoint_dir)
        self.phase = phase
        self.key = key
        self.artifact_type = artifact_type
        self.expected_metadata = expected_metadata
        self.bind_prompts = bind_prompts
        self.reads = [keys.PROMPTS] if bind_prompts else []
        self.writes = [key]

    def expected_read_types(
        self,
        results: Results | None = None,
        available_types: Mapping[str, ExpectedType] | None = None,
    ) -> Mapping[str, ExpectedType]:
        """Require current prompts only when restoring data preparation."""
        # Later checkpoints are self-identifying; data prep binds raw input rows.
        return {keys.PROMPTS: PromptBatch} if self.bind_prompts else {}

    def expected_write_types(
        self,
        results: Results | None = None,
        available_types: Mapping[str, ExpectedType] | None = None,
    ) -> Mapping[str, ExpectedType]:
        """Return the exact configured output type for pipeline validation."""
        # The restored artifact is the step's single scientific output.
        return {self.key: self.artifact_type}

    def __call__(self, results: Results) -> Results:
        """Load the configured artifact into Results and return those Results."""
        # Restore only the declared result key; metadata remains checkpoint context.
        artifact, metadata = load_phase_checkpoint(self.checkpoint_dir, self.phase)
        if self.expected_metadata is not None:
            mismatched = {
                key: (metadata.get(key), expected)
                for key, expected in self.expected_metadata.items()
                if metadata.get(key) != expected
            }
            if mismatched:
                raise ValueError(
                    f"FEGA checkpoint metadata mismatch for phase {self.phase!r}: "
                    f"{mismatched}"
                )
        if not isinstance(artifact, self.artifact_type):
            raise TypeError(
                f"FEGA checkpoint for phase {self.phase!r} contains "
                f"{type(artifact).__name__}, expected {self.artifact_type.__name__}"
            )
        if self.bind_prompts:
            if not isinstance(artifact, FEGADataPrepResult):
                raise TypeError(
                    "prompt binding requires a FEGADataPrepResult checkpoint"
                )
            _validate_prompt_identity(results[keys.PROMPTS], artifact)
        results[self.key] = artifact
        return results
