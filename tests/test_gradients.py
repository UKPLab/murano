"""Tests for gradient recording (RolloutBatch, LoadRollouts, RecordGradients)."""

from __future__ import annotations

import pytest
import torch

from murano import keys
from murano.pipeline import Pipeline
from murano.results import Results
from murano.steps.gradients import (
    GradientStore,
    LoadRollouts,
    RecordGradients,
    RolloutBatch,
)
from tests.conftest import toy_rollouts


def _toy_rollouts(n: int = 3, length: int = 14, prompt_len: int = 3) -> RolloutBatch:
    return toy_rollouts(n, length, prompt_len, seed=7)


def test_rollout_batch_validates_alignment():
    with pytest.raises(ValueError, match="align"):
        RolloutBatch(input_ids=[torch.tensor([4, 5, 6])], prompt_lengths=[1, 2])


def test_rollout_batch_rejects_bad_prompt_length():
    with pytest.raises(ValueError, match="prompt_length"):
        RolloutBatch(input_ids=[torch.tensor([4, 5, 6])], prompt_lengths=[0])
    with pytest.raises(ValueError, match="prompt_length"):
        RolloutBatch(input_ids=[torch.tensor([4, 5, 6])], prompt_lengths=[3])


def test_rollout_batch_rejects_misaligned_extras():
    with pytest.raises(ValueError, match="rewards"):
        RolloutBatch(
            input_ids=[torch.tensor([4, 5, 6])],
            prompt_lengths=[1],
            rewards=[1.0, 0.0],
        )


def test_rollout_batch_advantages_standardize_within_group():
    batch = RolloutBatch(
        input_ids=[torch.tensor([4, 5, 6])] * 5,
        prompt_lengths=[1] * 5,
        rewards=[1.0, 0.0, 0.0, 1.0, 1.0],
        groups=[0, 0, 0, 1, 1],
    )
    adv = batch.advantages()
    # Group 1 is all-pass: no within-group signal, advantage 0.
    assert torch.allclose(adv[3:], torch.zeros(2))
    # Group 0 is centered and the passing rollout sits above the failing ones.
    assert abs(adv[:3].mean().item()) < 1e-6
    assert adv[0] > 0 > adv[1]


def test_rollout_batch_advantages_require_rewards_and_groups():
    batch = _toy_rollouts()
    with pytest.raises(ValueError, match="rewards and groups"):
        batch.advantages()


def test_load_rollouts_writes_key():
    batch = _toy_rollouts()
    results = LoadRollouts(batch)(Results())
    assert results[keys.ROLLOUTS] is batch


def test_load_rollouts_rejects_non_batch():
    with pytest.raises(TypeError, match="RolloutBatch"):
        LoadRollouts([torch.tensor([4, 5])])  # pyright: ignore[reportArgumentType]


def test_record_gradients_validates_arguments(murano_model):
    with pytest.raises(ValueError, match="layer"):
        RecordGradients(murano_model, layer=5)
    with pytest.raises(ValueError, match="window"):
        RecordGradients(murano_model, layer=1, window=0)


def test_record_gradients_shapes_and_finiteness(murano_model):
    batch = _toy_rollouts(n=3, length=14, prompt_len=3)
    pipeline = Pipeline(
        [LoadRollouts(batch), RecordGradients(murano_model, layer=1, window=8)]
    )
    store = pipeline.run()[keys.GRADIENT_RECORD]
    assert isinstance(store, GradientStore)
    assert len(store) == 3
    for grad in store.gradients:
        # Window of 8 completion tokens, residual width of the tiny model.
        assert grad.shape == (8, murano_model.d_model)
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0
    assert store.finite.all()
    assert torch.isfinite(store.nll).all()


def test_record_gradients_window_clips_to_completion(murano_model):
    batch = _toy_rollouts(n=1, length=10, prompt_len=6)
    store = Pipeline(
        [LoadRollouts(batch), RecordGradients(murano_model, layer=0, window=512)]
    ).run()[keys.GRADIENT_RECORD]
    # Only 4 completion tokens exist, so the window clips to them.
    assert store.gradients[0].shape == (4, murano_model.d_model)


def test_record_gradients_is_deterministic(murano_model):
    batch = _toy_rollouts()
    step = RecordGradients(murano_model, layer=1, window=8)
    first = Pipeline([LoadRollouts(batch), step]).run()[keys.GRADIENT_RECORD]
    second = Pipeline([LoadRollouts(batch), step]).run()[keys.GRADIENT_RECORD]
    for a, b in zip(first.gradients, second.gradients):
        assert torch.equal(a, b)


def test_record_gradients_causality(murano_model):
    """With one loss term, the gradient at the read position is exactly zero.

    The token at the window's only position is predicted from the logits one
    position earlier, and the model is causal, so the residual stream *at*
    that position cannot influence its own prediction.
    """
    batch = _toy_rollouts(n=1, length=12, prompt_len=4)
    store = Pipeline(
        [LoadRollouts(batch), RecordGradients(murano_model, layer=1, window=1)]
    ).run()[keys.GRADIENT_RECORD]
    assert torch.allclose(
        store.gradients[0], torch.zeros_like(store.gradients[0]), atol=1e-6
    )


def test_record_gradients_nll_matches_nnsight_forward(murano_model):
    """The native graph-preserving forward agrees with the nnsight path.

    The teacher-forced NLL computed inside the gradient pass must match the
    NLL recomputed from ``forward_logits`` (a completely separate capture
    path), pinning the prompt/completion alignment and the logits themselves.
    """
    batch = _toy_rollouts(n=2, length=12, prompt_len=3)
    store = Pipeline(
        [LoadRollouts(batch), RecordGradients(murano_model, layer=1, window=6)]
    ).run()[keys.GRADIENT_RECORD]

    for n, ids in enumerate(batch.input_ids):
        tokens = {
            "input_ids": ids.unsqueeze(0),
            "attention_mask": torch.ones(1, ids.shape[0], dtype=torch.long),
        }
        logits = murano_model.forward_logits(tokens)
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        start, end = 3, 3 + 6
        picked = logprobs[0, start - 1 : end - 1].gather(1, ids[start:end].unsqueeze(1))
        expected = -picked.mean().item()
        assert store.nll[n].item() == pytest.approx(expected, abs=1e-4)


def test_record_gradients_on_gpt2(gpt2_model):
    """The native forward path also works on the GPT-2 architecture."""
    batch = _toy_rollouts(n=1, length=10, prompt_len=2)
    store = Pipeline(
        [LoadRollouts(batch), RecordGradients(gpt2_model, layer=1, window=4)]
    ).run()[keys.GRADIENT_RECORD]
    assert store.gradients[0].shape == (4, gpt2_model.d_model)
    assert torch.isfinite(store.gradients[0]).all()
    assert store.gradients[0].abs().sum() > 0


def test_gradient_store_save_load_roundtrip(murano_model, tmp_path):
    from murano.io import load_gradient_store, save_gradient_store

    batch = _toy_rollouts(n=2)
    store = Pipeline(
        [LoadRollouts(batch), RecordGradients(murano_model, layer=1, window=5)]
    ).run()[keys.GRADIENT_RECORD]
    path = tmp_path / "gradient_record.pt"
    save_gradient_store(store, path)
    loaded = load_gradient_store(path)
    assert loaded.layer == store.layer
    assert loaded.window == store.window
    assert torch.equal(loaded.nll, store.nll)
    for a, b in zip(loaded.gradients, store.gradients):
        assert torch.equal(a, b)


def test_save_results_dispatches_gradient_store(murano_model, tmp_path):
    from murano.io import save_results

    batch = _toy_rollouts(n=1)
    results = Pipeline(
        [LoadRollouts(batch), RecordGradients(murano_model, layer=1, window=4)]
    ).run()
    out = save_results(results, output_dir=str(tmp_path), model_id="tiny")
    assert (out / "gradients" / "gradient_record.pt").exists()
