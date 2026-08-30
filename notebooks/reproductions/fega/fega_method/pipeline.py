"""Notebook-local pipeline steps for Feature-Effect Geometry Analysis."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Mapping
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Sequence

import torch

from murano import keys as murano_keys
from murano.artifacts import PromptBatch
from murano.results import Results
from murano.steps.base import Step
from murano.steps.sae import SAEModel

from . import keys
from .artifacts import (
    FEGADataPrepResult,
    FEGAEffectStore,
    FEGAFeatureEffects,
    FEGAVMFResult,
)
from .config import FEGAConfig
from .contexts import (
    FEGAContext,
    PositionSpec,
    left_pad_contexts,
    prepare_contexts,
)
from .effects import normalize_effect_rows, run_reconstruction_readout_batch
from .vmf import (
    NoFiniteVMFCandidate,
    assignment_stability,
    feature_seed,
    select_vmf,
)

if TYPE_CHECKING:
    from murano.backend import ModelBackend


_VMF_VOCAB_CHUNK_SIZE = 16_384


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


def compute_runtime_gram(model: ModelBackend) -> torch.Tensor:
    """Build the exact runtime Gram from the model's canonical output embedding.

    Args:
        model: Backend exposing the active output embedding through `unembed_weight`.

    Returns:
        The float64 Gram matrix on CPU with shape `(d_model, d_model)`.

    Raises:
        ValueError: If the resulting Gram does not match the model residual width.
    """
    # Match the source route: cast on the configured compute device, multiply there,
    # move only the finished Gram to CPU, and release the transient float64 buffer.
    unembedding = model.unembed_weight.detach()
    unembedding64 = unembedding.to(device=unembedding.device, dtype=torch.float64)
    gram = (unembedding64.T @ unembedding64).cpu()
    del unembedding64
    if gram.shape != (model.d_model, model.d_model):
        raise ValueError("unembedding Gram does not match model residual width")
    return gram


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
    """Return a stable metadata representation of a FEGA position selector."""
    # Preserve shared selectors and freeze aligned per-row positions as a tuple.
    if isinstance(position, str | int):
        return position
    return tuple(int(value) for value in position)


def _resid_post_layer(sae: SAEModel) -> int:
    """Read the supported residual-post hook layer from public SAE metadata."""
    # The notebook method only targets the dictionary SAE's residual-post hook.
    metadata = sae.metadata
    if metadata.hook_layer is None:
        raise ValueError("FEGA requires SAE metadata.hook_layer")
    if metadata.hook_name is not None and "resid_post" not in metadata.hook_name:
        raise NotImplementedError("FEGA supports resid_post SAEs only")
    return int(metadata.hook_layer)


def _select_active_contexts(
    contexts: Sequence[FEGAContext],
    latent_codes: torch.Tensor,
    feature_ids: Sequence[int],
    activation_threshold: float,
    max_contexts: int,
    preselected_context_indices: Mapping[int, Sequence[int]] | None = None,
) -> dict[int, tuple[int, ...]]:
    """Select prompt-generic active rows by activation with stable tie breaks.

    Source FEGA's RAVEL adapter also filters and stratifies on dataset-specific
    pair/entity fields. This method receives an already chosen prompt set, so
    it retains the shared finite-threshold, activation-rank, and cap behavior.
    Prompt identity breaks exact ties independently of model batching.
    """
    # Rank each feature independently and cap only at the configured population.
    selected: dict[int, tuple[int, ...]] = {}
    explicit = {
        int(feature): tuple(int(index) for index in indices)
        for feature, indices in (preselected_context_indices or {}).items()
    }
    unknown_features = sorted(set(explicit) - {int(feature) for feature in feature_ids})
    if unknown_features:
        raise ValueError(
            f"preselected contexts include unknown feature ids {unknown_features}"
        )
    missing_pair_index = 2**63 - 1
    for feature in feature_ids:
        if feature in explicit:
            chosen = explicit[feature]
            if len(chosen) > max_contexts:
                raise ValueError(
                    "preselected contexts exceed max_contexts "
                    f"for feature {feature} ({len(chosen)} > {max_contexts})"
                )
            if len(set(chosen)) != len(chosen):
                raise ValueError(f"duplicate preselected context for feature {feature}")
            invalid = [index for index in chosen if index < 0 or index >= len(contexts)]
            if invalid:
                raise ValueError(
                    f"preselected context index out of range for feature {feature}: "
                    f"{invalid}"
                )
            inactive = [
                index
                for index in chosen
                if not bool(
                    torch.isfinite(latent_codes[index, feature]).item()
                    and (latent_codes[index, feature] > activation_threshold).item()
                )
            ]
            if inactive:
                raise ValueError(
                    f"inactive preselected context for feature {feature}: {inactive}"
                )
            selected[feature] = chosen
            continue
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

    reads = [murano_keys.PROMPTS]
    writes = [keys.DATA_PREP]
    read_types = {murano_keys.PROMPTS: PromptBatch}
    write_types = {keys.DATA_PREP: FEGADataPrepResult}

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
        preselected_context_indices: Mapping[int, Sequence[int]] | None = None,
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
        self.preselected_context_indices = (
            {
                int(feature): tuple(int(index) for index in indices)
                for feature, indices in preselected_context_indices.items()
            }
            if preselected_context_indices is not None
            else None
        )

    def __call__(self, results: Results) -> Results:
        """Tokenize, left-pad, reconstruct, and save ordered baseline rows."""
        # Resolve the supported raw residual site and load the SAE on model device.
        contexts = prepare_contexts(
            results[murano_keys.PROMPTS], self.model.tokenizer, self.position
        )
        unembedding = self.model.unembed_weight
        _warn_reduced_precision(unembedding)
        device = unembedding.device
        sae = self.sae_model or SAEModel(self.release, self.sae_id, device=str(device))
        if (sae.release, sae.sae_id) != (self.release, self.sae_id):
            raise ValueError("injected SAE identity does not match release and sae_id")
        hook_layer = _resid_post_layer(sae)
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
            self.preselected_context_indices,
        )
        results[keys.DATA_PREP] = FEGADataPrepResult(
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
                "selection_mode": (
                    "explicit_preselected"
                    if self.preselected_context_indices is not None
                    else "activation_ranked"
                ),
            },
        )
        return results


class FEGAComputeEffect(Step):
    """Collect feature-zeroed readouts and normalize their logit-space effects."""

    reads = [keys.DATA_PREP]
    writes = [keys.EFFECTS]
    read_types = {keys.DATA_PREP: FEGADataPrepResult}
    write_types = {keys.EFFECTS: FEGAEffectStore}

    def __init__(
        self,
        model: ModelBackend,
        *,
        batch_size: int = 8,
        tau_zero: float = 1e-12,
        sae_model: SAEModel | None = None,
    ) -> None:
        """Configure model batching and the source near-zero effect threshold."""
        # Store only controls that affect effect collection.
        self.model = model
        self.batch_size = batch_size
        self.tau_zero = float(tau_zero)
        self.sae_model = sae_model

    def __call__(self, results: Results) -> Results:
        """Ablate each requested feature and normalize ``ablated - baseline``."""
        # Bind the effect metric to the exact output embedding used by the model.
        prepared = results[keys.DATA_PREP]
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
            raise ValueError("FEGA model dtype changed between effect steps")
        _warn_reduced_precision(unembedding)
        gram = compute_runtime_gram(self.model)

        # Recreate the same baseline batches, changing only one latent coordinate.
        device = unembedding.device
        sae = self.sae_model or SAEModel(
            prepared.release, prepared.sae_id, device=str(device)
        )
        if (sae.release, sae.sae_id) != (prepared.release, prepared.sae_id):
            raise ValueError("injected SAE identity does not match data preparation")
        hook_layer = _resid_post_layer(sae)
        if hook_layer != prepared.hook_layer:
            raise ValueError("loaded SAE hook does not match data preparation")
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
        results[keys.EFFECTS] = FEGAEffectStore(
            features=features,
            gram=gram,
            feature_ids=prepared.feature_ids,
            metadata={
                "effect_sign": "ablated-minus-baseline",
                "tau_zero": self.tau_zero,
                "provenance": {
                    "model_dtype": str(unembedding.dtype),
                    "gram_compute_device": str(unembedding.device),
                    "gram_readout": "final_residual_output_embedding",
                    "data_prep_batch_size": prepared.metadata["batch_size"],
                    "effect_batch_size": self.batch_size,
                },
            },
        )
        return results


class FEGAVMF(Step):
    """Fit dense CPU vMF mixtures in vocabulary space."""

    reads = [keys.EFFECTS]
    writes = [keys.VMF]
    read_types = {keys.EFFECTS: FEGAEffectStore}
    write_types = {keys.VMF: FEGAVMFResult}

    def __init__(
        self,
        model: ModelBackend,
        config: FEGAConfig,
        *,
        n_jobs: int = 1,
    ) -> None:
        """Configure the canonical unembedding and CPU worker count."""
        # Preserve the source's worker-count semantics.
        self.model = model
        self.config = config
        self.n_jobs = int(n_jobs)

    def __call__(self, results: Results) -> Results:
        """Materialize one vocabulary cloud at a time, fit it, and release it."""
        # Reject model/effect mismatch before allocating any vocabulary cloud.
        effects = results[keys.EFFECTS]
        unembedding = self.model.unembed_weight.detach()
        selections: dict[int, Any] = {}
        stabilities: dict[int, float | None] = {}
        stability_counts: dict[int, dict[str, int]] = {}
        fit_status: dict[int, str] = {}
        failures: dict[int, Any] = {}
        # Fit every eligible feature in deterministic feature order.
        for feature in effects.ordered_feature_ids:
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
                    counts: dict[str, int] = {}
                    stabilities[feature] = assignment_stability(
                        cloud,
                        selection.selected,
                        seed,
                        self.config.vmf_resample_fraction,
                        self.config.vmf_resample_rounds,
                        self.n_jobs,
                        self.config.vmf_n_init,
                        self.config.vmf_max_iter,
                        evidence=counts,
                    )
                    stability_counts[feature] = counts
        results[keys.VMF] = FEGAVMFResult(
            selections,
            stabilities,
            asdict(self.config),
            fit_status,
            failures,
            assignment_stability_counts=stability_counts,
        )
        return results
