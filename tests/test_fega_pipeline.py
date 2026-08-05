"""Pipeline, context, SAE, effect, and checkpoint tests for native FEGA."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import Tensor, nn

from murano import PromptBatch, keys
from murano.backend import ModelBackend
from murano.fega.artifacts import (
    FEGADataPrepResult,
    FEGAEffectStore,
    FEGAFeatureEffects,
    FEGAGeometryResult,
    FEGAVMFResult,
)
from murano.fega.checkpoints import (
    FEGALoadCheckpoint,
    effect_resume_metadata,
    load_phase_checkpoint,
    save_phase_checkpoint,
)
from murano.fega.config import FEGAConfig
from murano.fega.contexts import (
    FEGAContext,
    FEGAContextBatch,
    left_pad_contexts,
    prepare_contexts,
)
from murano.fega.dictionary_sae import DictionarySAEModel
from murano.fega.effects import (
    normalize_effect_rows,
    run_reconstruction_readout_batch,
)
from murano.fega.vmf import NoFiniteVMFCandidate, VMFCandidateEvidence
from murano.pipeline import Pipeline
from murano.results import Results
from murano.steps.fega import (
    FEGAComputeEffect,
    FEGADataPrep,
    FEGAStability,
    FEGAVMF,
    _unembedding_fingerprint,
    fega_steps,
)
from murano.steps.prompts import LoadPrompts
from murano.steps.sae import SAEModel


class FakeSAE:
    """Small SAE whose full decode visibly differs from its input residual."""

    release = "fake-release"
    sae_id = "fake-sae"
    n_features = 2
    metadata = SimpleNamespace(hook_name="blocks.0.hook_resid_post", hook_layer=0)

    def _ensure_loaded(self) -> None:
        """Match the loaded-SAE contract without external weights."""
        # The deterministic fake has no deferred resources.

    def encode(self, residual: Tensor) -> Tensor:
        """Encode residual rows without changing their dimensionality."""
        # Shift the latent so reconstruction cannot accidentally equal the input.
        return residual + 1.0

    def decode(self, latent: Tensor) -> Tensor:
        """Decode latent rows with a deterministic non-identity reconstruction."""
        # Scale the latent to make both reconstruction and ablation observable.
        return latent * 2.0


class TinyModel(nn.Module):
    """Tiny hookable model with a deliberate post-head logit transformation."""

    def __init__(self) -> None:
        """Create the layer, tail, and output embedding used by the checks."""
        # Keep every module deterministic so hook semantics are isolated.
        super().__init__()
        self.batch_sizes: list[int] = []
        self.layer = nn.Identity()
        self.tail = nn.Identity()
        self.output_embedding = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.output_embedding.weight.copy_(torch.eye(2))

    def forward(self, input_ids: Tensor, **_: object) -> Tensor:
        """Return transformed logits while exposing the pre-head hidden tensor."""
        # Apply the deliberate transform only after the output embedding returns.
        self.batch_sizes.append(input_ids.shape[0])
        values = input_ids.to(torch.float32)
        hidden = (
            torch.stack((values, values + 1.0), dim=-1) if values.ndim == 2 else values
        )
        hidden = self.tail(self.layer(hidden))
        return self.output_embedding(hidden) + 100.0

    def get_output_embeddings(self) -> nn.Module:
        """Return the hookable language-model head used by FEGA."""
        # Mirror the Hugging Face causal-LM accessor exactly.
        return self.output_embedding


class TinyTokenizer:
    """Tokenize prompt length into deterministic mixed-length integer rows."""

    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, prompts: list[str], **_: object) -> dict[str, list[list[int]]]:
        """Return one nonempty token row per prompt without padding."""
        # Vary length by prompt text so the step must translate left padding.
        return {"input_ids": [list(range(1, len(prompt) + 1)) for prompt in prompts]}


class TinyBackend:
    """Expose the minimal raw-model surface used by native FEGA steps."""

    model_id = "tiny-model"
    n_layers = 1
    d_model = 2

    def __init__(self) -> None:
        """Create the deterministic raw model and tokenizer."""
        # Share one model instance across both phase steps in a run.
        self.raw_model = TinyModel()
        self.tokenizer = TinyTokenizer()

    @property
    def unembed_weight(self) -> Tensor:
        """Return the tiny output embedding in canonical vocabulary-row order."""
        # Use the same weight read by the raw model output head.
        return self.raw_model.output_embedding.weight

    def raw_layer(self, index: int) -> nn.Module:
        """Return the only supported raw residual layer."""
        # Fail if the integration test accidentally requests another layer.
        if index != 0:
            raise IndexError(index)
        return self.raw_model.layer


def test_reconstruction_readout_and_ablation_effect_contract() -> None:
    """Baseline must decode full z; ablation effect must be ablated minus baseline."""
    # Run baseline reconstruction and verify capture occurs before logit transform.
    model = TinyModel()
    tokens = {"input_ids": torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])}
    original, z, baseline = run_reconstruction_readout_batch(
        model,
        model.layer,
        model.output_embedding,
        FakeSAE(),
        tokens,
        target_positions=[1],
    )
    assert torch.equal(original, torch.tensor([[3.0, 4.0]]))
    assert torch.equal(z, torch.tensor([[4.0, 5.0]]))
    assert torch.equal(baseline[0, 1], torch.tensor([8.0, 10.0]))
    assert baseline.max() < 100

    # Zero feature zero in the same latent and assert the signed readout effect.
    ablated = run_reconstruction_readout_batch(
        model,
        model.layer,
        model.output_embedding,
        FakeSAE(),
        tokens,
        target_positions=[1],
        feature_ids=[0],
        z_batch=z,
    )
    effect = ablated - baseline
    assert torch.equal(ablated[0, 1], torch.tensor([0.0, 10.0]))
    assert torch.equal(effect[0, 1], torch.tensor([-8.0, 0.0]))


def test_normalize_effect_rows_filters_in_order() -> None:
    """Normalization must retain finite nonzero rows in their source order."""
    # Mix valid, zero, and nonfinite effects to exercise the public mask contract.
    baseline = torch.zeros(4, 2, dtype=torch.float64)
    ablated = torch.tensor(
        [[3.0, 4.0], [0.0, 0.0], [float("nan"), 1.0], [0.0, -2.0]],
        dtype=torch.float64,
    )
    directions, magnitudes, mask = normalize_effect_rows(
        baseline,
        ablated,
        torch.eye(2, dtype=torch.float64),
    )
    assert torch.equal(mask, torch.tensor([True, False, False, True]))
    assert directions.dtype == torch.float32
    assert magnitudes.dtype == torch.float32
    assert torch.allclose(magnitudes, torch.tensor([5.0, 2.0]))
    assert torch.allclose(directions, torch.tensor([[0.6, 0.8], [0.0, -1.0]]))


def test_native_effect_steps_are_batch_invariant_for_mixed_lengths() -> None:
    """Changing model batch size must preserve selected rows and effect values."""
    # Execute the real data-prep/effect step loops at both batch boundaries.
    stores = []
    for batch_size in (1, 3):
        backend = cast(ModelBackend, TinyBackend())
        sae = cast(SAEModel, FakeSAE())
        results = Results()
        results[keys.PROMPTS] = PromptBatch(["a", "bb", "cccc"], source="unit")
        FEGADataPrep(
            backend,
            FakeSAE.release,
            FakeSAE.sae_id,
            [0],
            batch_size=batch_size,
            sae_model=sae,
        )(results)
        FEGAComputeEffect(backend, batch_size=batch_size, sae_model=sae)(results)
        stores.append(results[keys.FEGA_EFFECTS])

    # The opaque analysis IDs differ, while every scientific row remains invariant.
    left, right = stores
    assert left.analysis_id != right.analysis_id
    assert left.features[0].context_indices == right.features[0].context_indices
    assert left.features[0].contexts == right.features[0].contexts
    assert left.features[0].retained_mask == right.features[0].retained_mask
    assert torch.equal(left.features[0].directions, right.features[0].directions)
    assert torch.equal(left.features[0].magnitudes, right.features[0].magnitudes)


def test_native_effect_step_batches_only_activation_ranked_contexts() -> None:
    """Effect collection must run only the selected feature population."""
    # Cap three active prompts to the two strongest activations in ranked order.
    backend = cast(ModelBackend, TinyBackend())
    sae = cast(SAEModel, FakeSAE())
    results = Results()
    results[keys.PROMPTS] = PromptBatch(["a", "bb", "cccc"], source="unit")
    FEGADataPrep(
        backend,
        FakeSAE.release,
        FakeSAE.sae_id,
        [0],
        batch_size=3,
        max_contexts=2,
        sae_model=sae,
    )(results)
    prepared = results[keys.FEGA_DATA_PREP]
    assert prepared.selected_context_indices[0] == (2, 1)

    # Clear baseline calls; one two-row ablation proves no discarded row was run.
    backend.raw_model.batch_sizes.clear()  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="must match data_prep batch_size"):
        FEGAComputeEffect(backend, batch_size=8, sae_model=sae)(results)
    assert backend.raw_model.batch_sizes == []  # type: ignore[attr-defined]
    FEGAComputeEffect(backend, batch_size=3, sae_model=sae)(results)
    effects = results[keys.FEGA_EFFECTS].features[0]
    assert backend.raw_model.batch_sizes == [2]  # type: ignore[attr-defined]
    assert effects.context_indices == (2, 1)


def test_requested_feature_without_active_contexts_is_retained() -> None:
    """A requested feature with no active row must remain reportable downstream."""
    # Raise the activation threshold above every fake latent and run effect collection.
    backend = cast(ModelBackend, TinyBackend())
    sae = cast(SAEModel, FakeSAE())
    results = Results()
    results[keys.PROMPTS] = PromptBatch(["a", "bb"], source="unit")
    FEGADataPrep(
        backend,
        FakeSAE.release,
        FakeSAE.sae_id,
        [0],
        activation_threshold=1.0e6,
        sae_model=sae,
    )(results)
    FEGAComputeEffect(backend, sae_model=sae)(results)

    # Preserve the feature as an explicit empty population instead of dropping it.
    feature = results[keys.FEGA_EFFECTS].features[0]
    assert feature.directions.shape == (0, backend.d_model)
    assert feature.context_indices == ()
    assert feature.retained_mask == ()


class _Tokenizer:
    """Minimal tokenizer returning fixed unpadded token rows."""

    def __call__(self, prompts: list[str], **_: object) -> dict[str, list[list[int]]]:
        """Map each prompt to deterministic token IDs for positioning tests."""
        # Make row lengths differ so left-padding translation is observable.
        return {
            "input_ids": [[index + 1] * (index + 2) for index, _ in enumerate(prompts)]
        }


def test_context_positions_left_padding_and_metadata_are_aligned() -> None:
    """Logical positions and metadata must survive source-equivalent left padding."""
    # Use per-row positions and every optional field to lock the public contract.
    prompts = PromptBatch(
        ["short", "longer"],
        source="unit",
        metadata={
            "fega": {
                "attribute_label": ["a", "b"],
                "pair_role": ["base", "source"],
                "pair_index": [7, 7],
                "group_label": ["g1", "g2"],
            }
        },
    )
    prepared = prepare_contexts(prompts, _Tokenizer(), position=[0, 2])
    padded = left_pad_contexts(prepared.contexts, pad_token_id=0)

    # Row zero receives one left pad, while prompt-local position IDs remain aligned.
    assert padded.input_ids.tolist() == [[0, 1, 1], [2, 2, 2]]
    assert padded.attention_mask.tolist() == [[0, 1, 1], [1, 1, 1]]
    assert padded.position_ids.tolist() == [[0, 0, 1], [0, 1, 2]]
    assert padded.target_positions == (1, 2)
    assert prepared.contexts[0].pair_index == 7
    assert prepared.contexts[1].group_label == "g2"

    # Shared and last selectors are prompt-relative, while malformed alignment fails.
    assert [
        c.target_position for c in prepare_contexts(prompts, _Tokenizer(), 1).contexts
    ] == [1, 1]
    assert [
        c.target_position
        for c in prepare_contexts(prompts, _Tokenizer(), "last").contexts
    ] == [1, 2]
    with pytest.raises(ValueError, match="match the prompt count"):
        prepare_contexts(prompts, _Tokenizer(), [0])


def test_dictionary_relu_loader_preserves_reconstruction_when_normalizing(
    tmp_path: Path,
) -> None:
    """Match the source centered ReLU equations after decoder normalization."""
    # Save one deliberately unnormalized source-shaped checkpoint.
    checkpoint = tmp_path / "release" / "sae"
    checkpoint.mkdir(parents=True)
    config = {
        "trainer": {
            "trainer_class": "StandardTrainerAprilUpdate",
            "lm_name": "tiny-model",
            "layer": 3,
        }
    }
    (checkpoint / "config.json").write_text(json.dumps(config))
    state = {
        "bias": torch.tensor([0.5, -0.5]),
        "encoder.weight": torch.tensor([[1.0, 2.0], [-1.0, 0.5]]),
        "encoder.bias": torch.tensor([0.2, -0.1]),
        "decoder.weight": torch.tensor([[2.0, 0.0], [0.0, 3.0]]),
    }
    torch.save(state, checkpoint / "ae.pt")
    residual = torch.tensor([[1.0, 2.0]])
    original_acts = torch.relu(
        (residual - state["bias"]) @ state["encoder.weight"].T + state["encoder.bias"]
    )
    expected = original_acts @ state["decoder.weight"].T + state["bias"]

    # Load through Murano and verify hook identity plus unchanged reconstruction.
    sae = DictionarySAEModel(
        "unused/repo", "sae", "tiny-model", local_dir=tmp_path / "release"
    )
    actual = sae.decode(sae.encode(residual))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.linalg.vector_norm(sae.decoder, dim=1), torch.ones(2)
    )
    assert sae.metadata.hook_layer == 3


def test_dictionary_relu_loader_quantizes_before_bfloat16_normalization(
    tmp_path: Path,
) -> None:
    """Match SAEBench's bfloat16 load-then-normalize order."""
    # Save weights whose normalized result changes if quantization happens last.
    checkpoint = tmp_path / "release" / "sae"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "trainer": {
                    "trainer_class": "StandardTrainerAprilUpdate",
                    "lm_name": "tiny-model",
                    "layer": 3,
                }
            }
        )
    )
    state = {
        "bias": torch.tensor([0.12345, -0.54321]),
        "encoder.weight": torch.tensor([[1.2345, 2.3456], [-1.4567, 0.5678]]),
        "encoder.bias": torch.tensor([0.2345, -0.1234]),
        "decoder.weight": torch.tensor([[2.3456, 0.1234], [0.3456, 3.4567]]),
    }
    torch.save(state, checkpoint / "ae.pt")

    # Reproduce the source loader's quantize, float32 normalize, and restore sequence.
    expected = state["decoder.weight"].T.to(torch.bfloat16).float()
    expected /= torch.linalg.vector_norm(expected, dim=1, keepdim=True)
    expected = expected.to(torch.bfloat16)
    sae = DictionarySAEModel(
        "unused/repo",
        "sae",
        "tiny-model",
        dtype=torch.bfloat16,
        local_dir=tmp_path / "release",
    )
    torch.testing.assert_close(sae.decoder, expected, rtol=0.0, atol=0.0)


def test_phase_checkpoint_roundtrip_validation_and_missing_message(
    tmp_path: Path,
) -> None:
    """Preserve data, enable phase-only validation, and identify missing phases."""
    # Round-trip the canonical phase payload and its metadata.
    artifact = {"tokens": [1, 2, 3]}
    metadata = {"model": "test"}
    path = save_phase_checkpoint(tmp_path, "data_prep", artifact, metadata)
    assert path == tmp_path / "data_prep.pt"
    assert load_phase_checkpoint(tmp_path, "data_prep") == (artifact, metadata)

    # Refuse a phase-only load whose declared run identity differs.
    incompatible = FEGALoadCheckpoint(
        tmp_path,
        "data_prep",
        "prepared",
        dict,
        expected_metadata={"model": "other"},
    )
    with pytest.raises(ValueError, match="metadata mismatch"):
        incompatible(Results())

    # Keep missing-checkpoint failures actionable with both phase and path.
    missing_path = tmp_path / "vmf.pt"
    with pytest.raises(FileNotFoundError) as error:
        load_phase_checkpoint(tmp_path, "vmf")
    assert "vmf" in str(error.value)
    assert str(missing_path) in str(error.value)


def test_analysis_identity_prevents_stale_partial_and_cross_phase_resume() -> None:
    """A new effect computation must invalidate partial and sibling checkpoints."""
    # Build two runs with equal feature/context indices but distinct analysis IDs.
    feature = FEGAFeatureEffects(
        feature_id=7,
        directions=torch.tensor([[1.0, 0.0]]),
        magnitudes=torch.tensor([2.0]),
        context_indices=(0,),
        feature_activations=torch.tensor([3.0]),
        retained_mask=(True,),
    )
    original = FEGAEffectStore(
        features={7: feature},
        gram=torch.eye(2),
        unembedding_fingerprint="unembedding",
        analysis_id="effect-run-a",
    )
    changed = replace(
        original,
        features={7: replace(feature, directions=torch.tensor([[0.0, 1.0]]))},
        analysis_id="effect-run-b",
    )

    # Partial-resume metadata follows the computation, not coincidental row indices.
    assert (
        effect_resume_metadata(original, FEGAConfig())["analysis_id"]
        != (effect_resume_metadata(changed, FEGAConfig())["analysis_id"])
    )

    # A multi-input phase refuses geometry from a different effect computation.
    results = Results()
    results[keys.FEGA_EFFECTS] = original
    results[keys.FEGA_GEOMETRY] = FEGAGeometryResult({}, "effect-run-b")
    results[keys.FEGA_VMF] = FEGAVMFResult({}, {}, "unembedding", "effect-run-a")
    with pytest.raises(ValueError, match="different effect analyses"):
        FEGAStability(FEGAConfig())(results)


def test_data_prep_checkpoint_is_bound_to_current_prompt_rows(tmp_path: Path) -> None:
    """Phase-only effect collection must reject stale ordered prompt contexts."""
    # Save one minimal prepared row with aligned metadata and its run metadata.
    context = FEGAContext(0, "Paris", (1, 2), 1, attribute_label="city")
    prepared = FEGADataPrepResult(
        FEGAContextBatch((context,), "test"),
        (3,),
        torch.zeros((1, 4)),
        torch.zeros((1, 2)),
        {3: ()},
        "release",
        "sae",
        0,
    )
    save_phase_checkpoint(tmp_path, "data_prep", prepared, {"model": "test"})
    loader = FEGALoadCheckpoint(
        tmp_path,
        "data_prep",
        keys.FEGA_DATA_PREP,
        FEGADataPrepResult,
        expected_metadata={"model": "test"},
        bind_prompts=True,
    )

    # Accept exact rows and fail before publishing a checkpoint for different rows.
    matching = Results()
    matching[keys.PROMPTS] = PromptBatch(
        ["Paris"], source="test", metadata={"fega": {"attribute_label": ["city"]}}
    )
    restored = loader(matching)[keys.FEGA_DATA_PREP]
    assert restored.contexts.contexts == (context,)
    stale = Results()
    stale[keys.PROMPTS] = PromptBatch(
        ["Berlin"], source="test", metadata={"fega": {"attribute_label": ["city"]}}
    )
    with pytest.raises(ValueError, match="prompt rows"):
        loader(stale)


def test_fega_steps_validate_all_and_vmf_only(tmp_path: Path) -> None:
    """The full chain validates, while vMF-only uses an explicit effect loader."""
    # Construction does not touch the model, so a typed sentinel isolates dependencies.
    model = cast(ModelBackend, cast(Any, object()))
    full = Pipeline(
        [
            LoadPrompts(PromptBatch(["small prompt"])),
            *fega_steps(
                model,
                release="release",
                sae_id="sae",
                feature_ids=[2],
                phases="all",
                checkpoint_dir=tmp_path,
            ),
        ]
    )
    assert "fega_reporting" in full.validate()

    # A phase-only chain validates because the checkpoint loader declares its output.
    vmf_steps = fega_steps(
        model,
        release="release",
        sae_id="sae",
        feature_ids=[2],
        phases="vmf",
        checkpoint_dir=tmp_path,
        seed=7,
    )
    vmf_only = Pipeline(vmf_steps)
    assert vmf_only.validate() == ["fega_effects", "fega_vmf"]
    effect_loader = vmf_steps[0]
    assert isinstance(effect_loader, FEGALoadCheckpoint)
    assert effect_loader.expected_metadata is not None
    assert "seed" not in effect_loader.expected_metadata["identity"]
    with pytest.raises(FileNotFoundError, match="compute_effect"):
        vmf_only.run()

    # A downstream phase binds the seed only when it loads stochastic vMF output.
    stability_steps = fega_steps(
        model,
        release="release",
        sae_id="sae",
        feature_ids=[2],
        phases="stability",
        checkpoint_dir=tmp_path,
        seed=7,
    )
    vmf_loader = next(
        step
        for step in stability_steps
        if isinstance(step, FEGALoadCheckpoint) and step.phase == "vmf"
    )
    assert vmf_loader.expected_metadata is not None
    assert vmf_loader.expected_metadata["config"]["seed"] == 7

    # Phase-only loaders bind position and activation selection settings.
    effect_only = fega_steps(
        model,
        release="release",
        sae_id="sae",
        feature_ids=[2],
        position=3,
        activation_threshold=0.75,
        phases=["compute_effect"],
        checkpoint_dir=tmp_path,
    )
    loader = effect_only[0]
    assert isinstance(loader, FEGALoadCheckpoint)
    assert loader.reads == [keys.PROMPTS]
    assert loader.expected_metadata is not None
    assert loader.expected_metadata["identity"]["position"] == 3
    assert loader.expected_metadata["identity"]["activation_threshold"] == 0.75
    assert "seed" not in loader.expected_metadata["identity"]

    # A skipped prerequisite still requires an explicit checkpoint directory.
    with pytest.raises(ValueError, match="compute_effect"):
        fega_steps(
            model,
            release="release",
            sae_id="sae",
            feature_ids=[2],
            phases="vmf",
        )


def test_phase_only_vmf_and_stability_resume_their_own_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checkpoints written by the workflow must load in later phase-only runs."""
    # Produce every prerequisite with one shared run identity and a tiny population.
    backend = cast(ModelBackend, TinyBackend())
    sae = cast(SAEModel, FakeSAE())
    prompts = PromptBatch(
        ["a", "bb", "ccc", "dddd", "eeeee", "ffffff", "ggggggg", "hhhhhhhh"],
        source="unit",
    )
    initial = Pipeline(
        [
            LoadPrompts(prompts),
            *fega_steps(
                backend,
                release=FakeSAE.release,
                sae_id=FakeSAE.sae_id,
                feature_ids=[0],
                phases=(
                    "data_prep",
                    "compute_effect",
                    "geometry_metrics",
                    "vmf",
                    "stability",
                ),
                checkpoint_dir=tmp_path,
                batch_size=1,
                max_contexts=8,
                sae_model=sae,
            ),
        ]
    ).run()

    # Make any recomputation fail, then change only the runtime batch size.
    monkeypatch.setattr(
        "murano.steps.fega._materialize_vmf_coordinates",
        lambda *_args, **_kwargs: pytest.fail("vMF was recomputed"),
    )
    monkeypatch.setattr(
        "murano.steps.fega_analysis.FEGAStability._feature_record",
        lambda *_args, **_kwargs: pytest.fail("stability was recomputed"),
    )
    vmf_only = Pipeline(
        fega_steps(
            backend,
            release=FakeSAE.release,
            sae_id=FakeSAE.sae_id,
            feature_ids=[0],
            phases=["vmf"],
            checkpoint_dir=tmp_path,
            batch_size=8,
            max_contexts=8,
            sae_model=sae,
        )
    ).run()
    stability_only = Pipeline(
        fega_steps(
            backend,
            release=FakeSAE.release,
            sae_id=FakeSAE.sae_id,
            feature_ids=[0],
            phases=["stability"],
            checkpoint_dir=tmp_path,
            batch_size=8,
            max_contexts=8,
            sae_model=sae,
        )
    ).run()
    assert vmf_only[keys.FEGA_VMF].analysis_id == initial[keys.FEGA_VMF].analysis_id
    assert stability_only[keys.FEGA_STABILITY].analysis_id == (
        initial[keys.FEGA_STABILITY].analysis_id
    )


def test_vmf_phase_records_failed_features_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One numerical fit failure must not abort later FEGA features."""
    # Build two eligible compact features against one matching fake unembedding.
    unembedding = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    model = cast(
        ModelBackend,
        cast(Any, SimpleNamespace(unembed_weight=unembedding, model_id="fake")),
    )
    rows = torch.tensor([[1.0, 0.0], [0.0, 1.0]] * 4)
    feature_effects = {
        feature: FEGAFeatureEffects(
            feature,
            rows,
            torch.ones(8),
            tuple(range(8)),
            torch.ones(8),
            (True,) * 8,
        )
        for feature in (3, 7)
    }
    results = Results()
    results[keys.FEGA_EFFECTS] = FEGAEffectStore(
        feature_effects,
        torch.eye(2),
        _unembedding_fingerprint(unembedding),
        analysis_id="failed-fit-run",
    )

    def fail_selection(*args: object, **kwargs: object) -> None:
        """Represent one complete source-style failed candidate schedule."""
        # Raise the numerical sentinel consumed only at the feature boundary.
        del args, kwargs
        raise NoFiniteVMFCandidate((VMFCandidateEvidence(1, "fit_failed"),))

    monkeypatch.setattr("murano.steps.fega.select_vmf", fail_selection)
    output = FEGAVMF(model, FEGAConfig(vmf_n_init=1, vmf_max_iter=1))(results)
    vmf = output[keys.FEGA_VMF]
    assert vmf.fit_status == {3: "fit_failed", 7: "fit_failed"}
    assert set(vmf.failures) == {3, 7}
    assert vmf.features == {}
