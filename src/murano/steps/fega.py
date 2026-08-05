"""Native Murano pipeline steps for Feature-Effect Geometry Analysis."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence
from uuid import uuid4

import torch

from murano import keys
from murano.artifacts import PromptBatch
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
from murano.fega.checkpoints import (
    FEGA_LABEL_VERSION,
    FEGA_PHASES,
    FEGALoadCheckpoint,
    FEGAWriteCheckpoint,
    effect_resume_metadata,
    load_phase_checkpoint,
    phase_checkpoint_path,
    save_phase_checkpoint,
)
from murano.fega.config import FEGAConfig
from murano.fega.contexts import (
    FEGAContext,
    PositionSpec,
    left_pad_contexts,
    prepare_contexts,
)
from murano.fega.effects import normalize_effect_rows, run_reconstruction_readout_batch
from murano.fega.vmf import (
    NoFiniteVMFCandidate,
    assignment_stability,
    feature_seed,
    select_vmf,
)
from murano.results import Results
from murano.steps.base import Step
from murano.steps.fega_analysis import (
    FEGAGeometryMetrics,
    FEGAGeometryReporting,
    FEGAStability,
    FEGAVisualize,
)
from murano.steps.sae import SAEModel, _resolve_hook

if TYPE_CHECKING:
    from murano.backend import ModelBackend


_PHASE_OUTPUTS: dict[str, tuple[str, type[Any]]] = {
    "data_prep": (keys.FEGA_DATA_PREP, FEGADataPrepResult),
    "compute_effect": (keys.FEGA_EFFECTS, FEGAEffectStore),
    "geometry_metrics": (keys.FEGA_GEOMETRY, FEGAGeometryResult),
    "vmf": (keys.FEGA_VMF, FEGAVMFResult),
    "stability": (keys.FEGA_STABILITY, FEGAStabilityResult),
    "geometry_reporting": (keys.FEGA_REPORTING, FEGAReportingResult),
    "visualize": (keys.FEGA_VISUALIZATION, FEGAVisualizationResult),
}

_PHASE_PREREQUISITES: dict[str, tuple[str, ...]] = {
    "data_prep": (),
    "compute_effect": ("data_prep",),
    "geometry_metrics": ("compute_effect",),
    "vmf": ("compute_effect",),
    "stability": ("compute_effect", "geometry_metrics", "vmf"),
    "geometry_reporting": ("compute_effect", "geometry_metrics", "vmf", "stability"),
    "visualize": ("compute_effect", "geometry_metrics", "vmf", "geometry_reporting"),
}

_VMF_VOCAB_CHUNK_SIZE = 16_384


def _unembedding_fingerprint(unembedding: torch.Tensor) -> str:
    """Hash the canonical unembedding's exact stored values, dtype, and shape."""
    # Preserve the source fingerprint contract at this single scientific boundary.
    tensor = unembedding.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(header + b"\0")
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _materialize_vmf_coordinates(
    directions: torch.Tensor, unembedding: torch.Tensor
) -> torch.Tensor:
    """Project residual directions with the source's fixed float32 chunk order."""
    # Preserve the source GPU reduction shape while bounding the derived vocab buffer.
    if directions.ndim != 2 or unembedding.ndim != 2:
        raise ValueError("FEGA directions and unembedding must be rank-2 tensors")
    if directions.shape[1] != unembedding.shape[1]:
        raise ValueError("FEGA direction width does not match the unembedding")
    device_rows = directions.to(device=unembedding.device, dtype=torch.float32)
    coordinates = torch.empty(
        (directions.shape[0], unembedding.shape[0]),
        dtype=torch.float32,
        device="cpu",
    )
    for start in range(0, unembedding.shape[0], _VMF_VOCAB_CHUNK_SIZE):
        end = min(start + _VMF_VOCAB_CHUNK_SIZE, unembedding.shape[0])
        readout_chunk = unembedding[start:end].to(dtype=torch.float32)
        projected = device_rows @ readout_chunk.T
        coordinates[:, start:end].copy_(projected.detach().cpu())
        del projected, readout_chunk
    norms = torch.linalg.vector_norm(coordinates, dim=1)
    if not torch.isfinite(norms).all() or bool(torch.any(norms <= 0).item()):
        raise ValueError("FEGA vocabulary coordinates have invalid norms")
    coordinates.div_(norms[:, None])
    return coordinates


def _warn_reduced_precision(unembedding: torch.Tensor) -> None:
    """Warn when live FEGA model arithmetic can depend on batch shape."""

    # The Gemma bfloat16 source recipe is exact at its original batch, not invariant.
    if unembedding.dtype in {torch.float16, torch.bfloat16}:
        warnings.warn(
            "reduced-precision FEGA effect collection can be batch-sensitive; "
            "batch size is part of the result; float32 model weights reduced "
            "the observed sensitivity in the validated Gemma run",
            RuntimeWarning,
            stacklevel=3,
        )


def _target_rows(readout: torch.Tensor, positions: Sequence[int]) -> torch.Tensor:
    """Gather one target-position row from each rank-three model readout."""
    # Use the same physical positions that the residual hook patched.
    indices = torch.as_tensor(positions, device=readout.device, dtype=torch.long)
    rows = torch.arange(readout.shape[0], device=readout.device)
    return readout[rows, indices]


def _chunks(count: int, batch_size: int) -> list[tuple[int, int]]:
    """Return deterministic half-open batch ranges covering ``count`` rows."""
    # Fix chunk boundaries from row count and user batch size only.
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return [
        (start, min(start + batch_size, count)) for start in range(0, count, batch_size)
    ]


def _position_identity(position: PositionSpec) -> str | int | tuple[int, ...]:
    """Return the stable checkpoint representation of a FEGA position selector."""
    # Preserve shared selectors and freeze aligned per-row positions as a tuple.
    if isinstance(position, str | int):
        return position
    return tuple(int(value) for value in position)


def _run_identity(
    model: ModelBackend,
    release: str,
    sae_id: str,
    feature_ids: Sequence[int],
    position: PositionSpec,
    activation_threshold: float,
    max_contexts: int,
) -> dict[str, Any]:
    """Return upstream inputs shared by every FEGA checkpoint phase."""
    # Keep stochastic and reporting settings in the phases that consume them.
    return {
        "model_id": getattr(model, "model_id", None),
        "release": release,
        "sae_id": sae_id,
        "feature_ids": tuple(sorted({int(feature) for feature in feature_ids})),
        "position": _position_identity(position),
        "activation_threshold": float(activation_threshold),
        "max_contexts": int(max_contexts),
    }


def _select_active_contexts(
    contexts: Sequence[FEGAContext],
    latent_codes: torch.Tensor,
    feature_ids: Sequence[int],
    activation_threshold: float,
    max_contexts: int,
) -> dict[int, tuple[int, ...]]:
    """Select prompt-generic active rows by activation with stable tie breaks.

    Source FEGA's RAVEL adapter also filters and stratifies on dataset-specific
    pair/entity fields. Native Murano receives an already chosen prompt set, so
    it retains the shared finite-threshold, activation-rank, and cap behavior.
    Prompt identity breaks exact ties independently of model batching.
    """
    # Rank each feature independently and cap only at the configured population.
    selected: dict[int, tuple[int, ...]] = {}
    missing_pair_index = 2**63 - 1
    for feature in feature_ids:
        active = torch.nonzero(
            torch.isfinite(latent_codes[:, feature])
            & (latent_codes[:, feature] > activation_threshold),
            as_tuple=False,
        ).flatten()

        def selection_key(index: int) -> tuple[float, int, str, int]:
            """Return activation rank followed by stable prompt identity."""
            # Preserve deterministic ordering when reduced precision creates ties.
            context = contexts[index]
            identity = "\x1f".join(
                (
                    context.prompt,
                    context.group_label or "",
                    context.attribute_label or "",
                )
            )
            digest = hashlib.sha256(identity.encode()).hexdigest()
            pair_index = (
                context.pair_index
                if context.pair_index is not None
                else missing_pair_index
            )
            return (
                -float(latent_codes[index, feature]),
                pair_index,
                digest,
                context.index,
            )

        ranked = sorted((int(index) for index in active), key=selection_key)
        selected[feature] = tuple(ranked[:max_contexts])
    return selected


class FEGADataPrep(Step):
    """Prepare contexts and collect the full-SAE-reconstruction baseline."""

    reads = [keys.PROMPTS]
    writes = [keys.FEGA_DATA_PREP]
    read_types = {keys.PROMPTS: PromptBatch}
    write_types = {keys.FEGA_DATA_PREP: FEGADataPrepResult}

    def __init__(
        self,
        model: ModelBackend,
        release: str,
        sae_id: str,
        feature_ids: Sequence[int],
        *,
        position: PositionSpec = "last",
        batch_size: int = 8,
        activation_threshold: float = 0.0,
        max_contexts: int = 64,
        sae_model: SAEModel | None = None,
    ) -> None:
        """Configure prompt positions, SAE identity, features, and model batching."""
        # Canonicalize explicit feature IDs without making dataset-specific choices.
        ids = tuple(sorted({int(feature) for feature in feature_ids}))
        if not ids or ids[0] < 0:
            raise ValueError("feature_ids must contain non-negative feature indices")
        self.model = model
        self.release = release
        self.sae_id = sae_id
        self.feature_ids = ids
        self.position: PositionSpec = position
        self.position_identity = _position_identity(position)
        self.batch_size = batch_size
        self.activation_threshold = float(activation_threshold)
        if max_contexts < 1:
            raise ValueError("max_contexts must be positive")
        self.max_contexts = int(max_contexts)
        self.sae_model = sae_model

    def __call__(self, results: Results) -> Results:
        """Tokenize, left-pad, reconstruct, and save ordered baseline rows."""
        # Resolve the supported raw residual site and load the SAE on model device.
        contexts = prepare_contexts(
            results[keys.PROMPTS], self.model.tokenizer, self.position
        )
        unembedding = self.model.unembed_weight
        _warn_reduced_precision(unembedding)
        device = unembedding.device
        sae = self.sae_model or SAEModel(self.release, self.sae_id, device=str(device))
        if (sae.release, sae.sae_id) != (self.release, self.sae_id):
            raise ValueError("injected SAE identity does not match release and sae_id")
        sae._ensure_loaded()
        hook_layer, hook_kind = _resolve_hook(sae)
        if hook_kind != "resid_post":
            raise NotImplementedError(
                "native FEGA currently supports resid_post SAEs only"
            )
        if hook_layer < 0 or hook_layer >= self.model.n_layers:
            raise ValueError(f"SAE hook layer {hook_layer} is outside this model")
        if self.feature_ids[-1] >= sae.n_features:
            raise ValueError(
                f"feature id {self.feature_ids[-1]} exceeds SAE width {sae.n_features}"
            )

        # Run full-reconstruction baselines in stable prompt order and detach promptly.
        pad_id = self.model.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.model.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("FEGA left padding requires a tokenizer pad or EOS token")
        latents: list[torch.Tensor] = []
        baselines: list[torch.Tensor] = []
        raw_model = self.model.raw_model
        output_embedding = raw_model.get_output_embeddings()
        with torch.inference_mode():
            for start, stop in _chunks(len(contexts.contexts), self.batch_size):
                padded = left_pad_contexts(
                    contexts.contexts[start:stop], pad_token_id=pad_id, device=device
                )
                _, z, readout = run_reconstruction_readout_batch(
                    raw_model,
                    self.model.raw_layer(hook_layer),
                    output_embedding,
                    sae,
                    padded.as_tokens(),
                    padded.target_positions,
                )
                latents.append(z.detach().float().cpu())
                baselines.append(
                    _target_rows(readout, padded.target_positions)
                    .detach()
                    .float()
                    .cpu()
                )
        latent_codes = torch.cat(latents)
        baseline_rows = torch.cat(baselines)

        # Select the strongest active population independently for every feature.
        selected = _select_active_contexts(
            contexts.contexts,
            latent_codes,
            self.feature_ids,
            self.activation_threshold,
            self.max_contexts,
        )
        results[keys.FEGA_DATA_PREP] = FEGADataPrepResult(
            contexts=contexts,
            feature_ids=self.feature_ids,
            latent_codes=latent_codes,
            baseline_readouts=baseline_rows,
            selected_context_indices=selected,
            release=self.release,
            sae_id=self.sae_id,
            hook_layer=hook_layer,
            metadata={
                "activation_threshold": self.activation_threshold,
                "batch_size": self.batch_size,
                "max_contexts": self.max_contexts,
                "model_dtype": str(self.model.unembed_weight.dtype),
                "position": self.position_identity,
            },
        )
        return results


class FEGAComputeEffect(Step):
    """Collect feature-zeroed readouts and normalize their logit-space effects."""

    reads = [keys.FEGA_DATA_PREP]
    writes = [keys.FEGA_EFFECTS]
    read_types = {keys.FEGA_DATA_PREP: FEGADataPrepResult}
    write_types = {keys.FEGA_EFFECTS: FEGAEffectStore}

    def __init__(
        self,
        model: ModelBackend,
        *,
        batch_size: int = 8,
        tau_zero: float = 1e-12,
        sae_model: SAEModel | None = None,
        run_identity: dict[str, Any] | None = None,
    ) -> None:
        """Configure model batching and the source near-zero effect threshold."""
        # Store only controls that affect effect collection.
        self.model = model
        self.batch_size = batch_size
        self.tau_zero = float(tau_zero)
        self.sae_model = sae_model
        self.run_identity = None if run_identity is None else dict(run_identity)

    def __call__(self, results: Results) -> Results:
        """Ablate each requested feature and normalize ``ablated - baseline``."""
        # Bind the effect metric to the exact output embedding used by the model.
        prepared = results[keys.FEGA_DATA_PREP]
        prepared_batch_size = prepared.metadata.get("batch_size")
        if prepared_batch_size is not None and prepared_batch_size != self.batch_size:
            raise ValueError(
                "FEGA compute_effect batch_size must match data_prep batch_size "
                f"({self.batch_size} != {prepared_batch_size}); rerun data_prep or "
                f"use batch_size={prepared_batch_size}"
            )
        unembedding = self.model.unembed_weight.detach()
        prepared_dtype = prepared.metadata.get("model_dtype")
        if prepared_dtype is not None and prepared_dtype != str(unembedding.dtype):
            raise ValueError("FEGA model dtype does not match the data_prep checkpoint")
        _warn_reduced_precision(unembedding)
        fingerprint = _unembedding_fingerprint(unembedding)
        unembedding64 = unembedding.to(torch.float64)
        gram = (unembedding64.T @ unembedding64).cpu()
        del unembedding64
        if gram.shape != (self.model.d_model, self.model.d_model):
            raise ValueError("unembedding Gram does not match model residual width")

        # Recreate the same baseline batches, changing only one latent coordinate.
        device = unembedding.device
        sae = self.sae_model or SAEModel(
            prepared.release, prepared.sae_id, device=str(device)
        )
        if (sae.release, sae.sae_id) != (prepared.release, prepared.sae_id):
            raise ValueError(
                "injected SAE identity does not match data_prep checkpoint"
            )
        sae._ensure_loaded()
        hook_layer, hook_kind = _resolve_hook(sae)
        if (hook_layer, hook_kind) != (prepared.hook_layer, "resid_post"):
            raise ValueError("loaded SAE hook does not match the data_prep checkpoint")
        pad_id = self.model.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.model.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("FEGA left padding requires a tokenizer pad or EOS token")
        raw_model = self.model.raw_model
        output_embedding = raw_model.get_output_embeddings()
        features: dict[int, FEGAFeatureEffects] = {}
        with torch.inference_mode():
            for feature in prepared.feature_ids:
                chosen = prepared.selected_context_indices[feature]
                if not chosen:
                    features[feature] = FEGAFeatureEffects(
                        feature_id=feature,
                        directions=torch.empty((0, self.model.d_model)),
                        magnitudes=torch.empty(0),
                        context_indices=(),
                        feature_activations=torch.empty(0),
                        retained_mask=(),
                        contexts=(),
                    )
                    continue
                chosen_tensor = torch.tensor(chosen, dtype=torch.long)
                chosen_contexts = tuple(
                    prepared.contexts.contexts[index] for index in chosen
                )
                ablated_rows: list[torch.Tensor] = []
                for start, stop in _chunks(len(chosen), self.batch_size):
                    batch_indices = chosen_tensor[start:stop]
                    padded = left_pad_contexts(
                        chosen_contexts[start:stop],
                        pad_token_id=pad_id,
                        device=device,
                    )
                    readout = run_reconstruction_readout_batch(
                        raw_model,
                        self.model.raw_layer(hook_layer),
                        output_embedding,
                        sae,
                        padded.as_tokens(),
                        padded.target_positions,
                        feature_ids=[feature] * (stop - start),
                        z_batch=prepared.latent_codes[batch_indices],
                    )
                    assert isinstance(readout, torch.Tensor)
                    ablated_rows.append(
                        _target_rows(readout, padded.target_positions)
                        .detach()
                        .float()
                        .cpu()
                    )
                all_ablated = torch.cat(ablated_rows)
                directions, magnitudes, mask = normalize_effect_rows(
                    prepared.baseline_readouts[chosen_tensor],
                    all_ablated,
                    gram,
                    self.tau_zero,
                )
                retained = tuple(
                    index
                    for index, keep in zip(chosen, mask.tolist(), strict=True)
                    if keep
                )
                features[feature] = FEGAFeatureEffects(
                    feature_id=feature,
                    directions=directions.cpu(),
                    magnitudes=magnitudes.cpu(),
                    context_indices=retained,
                    feature_activations=prepared.latent_codes[chosen_tensor, feature][
                        mask
                    ].cpu(),
                    retained_mask=tuple(bool(value) for value in mask.tolist()),
                    contexts=tuple(
                        prepared.contexts.contexts[index] for index in retained
                    ),
                )
        results[keys.FEGA_EFFECTS] = FEGAEffectStore(
            features=features,
            gram=gram,
            unembedding_fingerprint=fingerprint,
            metadata={
                "effect_sign": "ablated-minus-baseline",
                "tau_zero": self.tau_zero,
                "identity": self.run_identity
                or {
                    "model_id": getattr(self.model, "model_id", None),
                    "release": prepared.release,
                    "sae_id": prepared.sae_id,
                    "feature_ids": prepared.feature_ids,
                    "position": prepared.metadata["position"],
                    "activation_threshold": prepared.metadata["activation_threshold"],
                    "max_contexts": prepared.metadata["max_contexts"],
                },
                "provenance": {
                    "model_dtype": str(unembedding.dtype),
                    "data_prep_batch_size": prepared.metadata["batch_size"],
                    "effect_batch_size": self.batch_size,
                },
            },
            analysis_id=uuid4().hex,
        )
        return results


class FEGAVMF(Step):
    """Fit dense CPU vMF mixtures in vocabulary space."""

    reads = [keys.FEGA_EFFECTS]
    writes = [keys.FEGA_VMF]
    read_types = {keys.FEGA_EFFECTS: FEGAEffectStore}
    write_types = {keys.FEGA_VMF: FEGAVMFResult}

    def __init__(
        self,
        model: ModelBackend,
        config: FEGAConfig,
        *,
        n_jobs: int = 1,
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        """Configure the canonical unembedding, CPU workers, and optional resume."""
        # Keep partial resume local to this expensive feature loop.
        self.model = model
        self.config = config
        self.n_jobs = int(n_jobs)
        self.checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)

    def __call__(self, results: Results) -> Results:
        """Materialize one vocabulary cloud at a time, fit it, and release it."""
        # Reject model/effect mismatch before allocating any vocabulary cloud.
        effects = results[keys.FEGA_EFFECTS]
        if not effects.analysis_id:
            raise ValueError("FEGA effects are missing their analysis identity")
        unembedding = self.model.unembed_weight.detach()
        fingerprint = _unembedding_fingerprint(unembedding)
        if fingerprint != effects.unembedding_fingerprint:
            raise ValueError("FEGA vMF unembedding fingerprint does not match effects")
        selections: dict[int, Any] = {}
        stabilities: dict[int, float | None] = {}
        fit_status: dict[int, str] = {}
        failures: dict[int, Any] = {}
        resume_metadata = effect_resume_metadata(effects, self.config)
        if (
            self.checkpoint_dir is not None
            and phase_checkpoint_path(self.checkpoint_dir, "vmf").exists()
        ):
            prior, prior_metadata = load_phase_checkpoint(self.checkpoint_dir, "vmf")
            if isinstance(prior, FEGAVMFResult) and prior_metadata == resume_metadata:
                selections.update(prior.features)
                stabilities.update(prior.assignment_stability)
                fit_status.update(prior.fit_status)
                failures.update(prior.failures)

        # Fit only unfinished eligible features and checkpoint after each completion.
        for feature in sorted(effects.features):
            if feature in fit_status:
                continue
            rows = effects.features[feature].directions
            if len(rows) < self.config.min_contexts:
                stabilities[feature] = None
                fit_status[feature] = "insufficient_data"
            else:
                if len(rows) > self.config.warn_contexts:
                    estimated_bytes = int(len(rows) * unembedding.shape[0] * 4)
                    warnings.warn(
                        f"retaining {len(rows)} contexts requires an estimated "
                        f"{estimated_bytes} byte float32 vocabulary buffer",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                vocabulary_rows = _materialize_vmf_coordinates(rows, unembedding)
                cloud = vocabulary_rows.detach().cpu().numpy()
                del vocabulary_rows
                seed = feature_seed(self.config.seed, feature)
                try:
                    selection = select_vmf(
                        cloud,
                        self.config.vmf_k_values,
                        seed,
                        self.config.vmf_n_init,
                        self.config.vmf_max_iter,
                        self.config.vmf_bic_tolerance,
                        self.n_jobs,
                        warn_large=False,
                    )
                except NoFiniteVMFCandidate as error:
                    stabilities[feature] = None
                    fit_status[feature] = "fit_failed"
                    failures[feature] = error.candidates
                else:
                    selections[feature] = selection
                    fit_status[feature] = "fitted"
                    stabilities[feature] = assignment_stability(
                        cloud,
                        selection.selected,
                        seed,
                        self.config.vmf_resample_fraction,
                        self.config.vmf_resample_rounds,
                        self.n_jobs,
                        self.config.vmf_n_init,
                        self.config.vmf_max_iter,
                    )
            if self.checkpoint_dir is not None:
                save_phase_checkpoint(
                    self.checkpoint_dir,
                    "vmf",
                    FEGAVMFResult(
                        dict(selections),
                        dict(stabilities),
                        fingerprint,
                        effects.analysis_id,
                        dict(fit_status),
                        dict(failures),
                    ),
                    resume_metadata,
                )
        results[keys.FEGA_VMF] = FEGAVMFResult(
            selections,
            stabilities,
            fingerprint,
            effects.analysis_id,
            fit_status,
            failures,
        )
        if self.checkpoint_dir is not None:
            save_phase_checkpoint(
                self.checkpoint_dir,
                "vmf",
                results[keys.FEGA_VMF],
                resume_metadata,
            )
        return results


def _canonical_phases(phases: str | Sequence[str]) -> tuple[str, ...]:
    """Validate phase names, remove duplicates, and restore dependency-safe order."""
    # Treat one phase string as one request rather than iterating its characters.
    requested = (
        tuple(_PHASE_OUTPUTS)
        if phases == "all"
        else (phases,)
        if isinstance(phases, str)
        else tuple(str(phase) for phase in phases)
    )
    unknown = sorted(set(requested) - set(_PHASE_OUTPUTS))
    if unknown:
        raise ValueError(f"unknown FEGA phases: {unknown}; expected {FEGA_PHASES}")
    return tuple(phase for phase in _PHASE_OUTPUTS if phase in requested)


def fega_steps(
    model: ModelBackend,
    *,
    release: str,
    sae_id: str,
    feature_ids: Sequence[int],
    position: PositionSpec = "last",
    phases: str | Sequence[str] = "all",
    checkpoint_dir: str | Path | None = None,
    batch_size: int = 8,
    activation_threshold: float = 0.0,
    max_contexts: int = 64,
    seed: int = 42,
    n_jobs: int = 1,
    top_k_per_family: int = 3,
    figures: Sequence[str] = ("atlas", "sphere_surface", "projection_2d"),
    output_dir: str | Path | None = None,
    dpi: int = 300,
    sae_model: SAEModel | None = None,
) -> list[Step]:
    """Build the requested FEGA phases in dependency order.

    ``phases`` accepts ``"all"`` or names from :data:`FEGA_PHASES`. If a
    requested phase needs an omitted prerequisite, the builder inserts a loader
    from ``checkpoint_dir``; without that directory it raises instead of running
    the missing phase. Executed phases are checkpointed when a directory is set.
    """
    # Canonicalize once and require checkpoints only when a requested phase has gaps.
    requested = _canonical_phases(phases)
    checkpoint_path = None if checkpoint_dir is None else Path(checkpoint_dir)
    produced: set[str] = set()
    steps: list[Step] = []
    config = FEGAConfig(seed=seed)
    identity = _run_identity(
        model,
        release,
        sae_id,
        feature_ids,
        position,
        activation_threshold,
        max_contexts,
    )
    base_metadata = {"identity": identity}
    analysis_metadata = {**base_metadata, "config": asdict(config)}
    reporting_metadata = {
        **analysis_metadata,
        "label_version": FEGA_LABEL_VERSION,
    }
    checkpoint_metadata = {
        "data_prep": base_metadata,
        "compute_effect": base_metadata,
        "geometry_metrics": base_metadata,
        "vmf": analysis_metadata,
        "stability": analysis_metadata,
        "geometry_reporting": reporting_metadata,
        "visualize": reporting_metadata,
    }
    visualization_dir = (
        Path(output_dir)
        if output_dir is not None
        else (checkpoint_path or Path(keys.DEFAULT_OUTPUT_DIR) / "fega")
        / "visualizations"
    )
    phase_steps: dict[str, Step] = {
        "data_prep": FEGADataPrep(
            model,
            release,
            sae_id,
            feature_ids,
            position=position,
            batch_size=batch_size,
            activation_threshold=activation_threshold,
            max_contexts=max_contexts,
            sae_model=sae_model,
        ),
        "compute_effect": FEGAComputeEffect(
            model,
            batch_size=batch_size,
            tau_zero=config.eps,
            sae_model=sae_model,
            run_identity=identity,
        ),
        "geometry_metrics": FEGAGeometryMetrics(config),
        "vmf": FEGAVMF(model, config, n_jobs=n_jobs, checkpoint_dir=checkpoint_path),
        "stability": FEGAStability(config, checkpoint_dir=checkpoint_path),
        "geometry_reporting": FEGAGeometryReporting(),
        "visualize": FEGAVisualize(
            visualization_dir,
            top_k_per_family=top_k_per_family,
            figures=figures,
            dpi=dpi,
        ),
    }
    for phase in requested:
        for prerequisite in _PHASE_PREREQUISITES[phase]:
            if prerequisite in produced:
                continue
            if checkpoint_path is None:
                raise ValueError(
                    f"phase {phase!r} requires {prerequisite!r}; provide checkpoint_dir or include that phase"
                )
            key, artifact_type = _PHASE_OUTPUTS[prerequisite]
            steps.append(
                FEGALoadCheckpoint(
                    checkpoint_path,
                    prerequisite,
                    key,
                    artifact_type,
                    expected_metadata=checkpoint_metadata[prerequisite],
                    bind_prompts=prerequisite == "data_prep",
                )
            )
            produced.add(prerequisite)
        steps.append(phase_steps[phase])
        produced.add(phase)
        if checkpoint_path is not None and phase not in {"vmf", "stability"}:
            key, artifact_type = _PHASE_OUTPUTS[phase]
            steps.append(
                FEGAWriteCheckpoint(
                    checkpoint_path,
                    phase,
                    key,
                    artifact_type,
                    checkpoint_metadata[phase],
                )
            )
    return steps
