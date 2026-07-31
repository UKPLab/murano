"""Tests for the GSAE: trainer, sparsity, census, and readout."""

from __future__ import annotations

import pytest
import torch

from murano import keys
from murano.steps.gradients import GradientStore
from murano.steps.gsae import (
    GSAE,
    _reinit_dead_features,
    classify_features,
    firing_rates,
    normalized_gradient_inputs,
    promoted_tokens,
)


def _synthetic_inputs(n: int = 512, d: int = 16, rank: int = 4) -> torch.Tensor:
    """Low-rank data a small dictionary can actually reconstruct."""
    generator = torch.Generator().manual_seed(0)
    basis = torch.randn(rank, d, generator=generator)
    codes = torch.randn(n, rank, generator=generator)
    return codes @ basis


def _synthetic_store(n: int = 4, t: int = 6, d: int = 16) -> GradientStore:
    generator = torch.Generator().manual_seed(1)
    gradients = [torch.randn(t, d, generator=generator) for _ in range(n)]
    return GradientStore(
        gradients=gradients,
        nll=torch.zeros(n),
        finite=torch.ones(n).bool(),
        layer=0,
        window=t,
    )


# ------------------------------------------------------------------- inputs


def test_normalized_inputs_have_unit_rms_and_signed_rows():
    store = _synthetic_store(n=3)
    advantages = torch.tensor([1.5, -0.5, 0.0])
    inputs = normalized_gradient_inputs(store, advantages)
    # The zero-advantage rollout is dropped: 2 rollouts x 6 positions remain.
    assert inputs.shape == (12, 16)
    rms = inputs.pow(2).mean(dim=1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)
    # The negative-advantage rollout's rows are sign-flipped copies.
    grad = store.gradients[1].float()
    expected = -grad / grad.pow(2).mean(dim=1, keepdim=True).sqrt()
    assert torch.allclose(inputs[6:], expected, atol=1e-5)


def test_normalized_inputs_validation():
    store = _synthetic_store(n=2)
    with pytest.raises(ValueError, match="advantages"):
        normalized_gradient_inputs(store, torch.ones(3))
    with pytest.raises(ValueError, match="subsample"):
        normalized_gradient_inputs(store, subsample=1.5)


def test_normalized_inputs_subsample():
    store = _synthetic_store(n=4, t=8)
    inputs = normalized_gradient_inputs(store, subsample=0.25, seed=0)
    assert inputs.shape == (8, 16)


# ------------------------------------------------------------------ trainer


def test_encode_is_exactly_k_sparse():
    gsae = GSAE.train(_synthetic_inputs(), m=32, k=3, epochs=1, batch_size=128)
    codes = gsae.encode(_synthetic_inputs(n=64))
    assert codes.shape == (64, 32)
    assert (codes.ne(0).sum(dim=1) == 3).all()


def test_train_shapes_and_unit_decoder_rows():
    gsae = GSAE.train(_synthetic_inputs(), m=32, k=4, epochs=2, batch_size=128)
    assert gsae.w_enc.shape == (32, 16)
    assert gsae.w_dec.shape == (32, 16)
    assert gsae.b_pre.shape == (16,)
    norms = gsae.w_dec.norm(dim=1)
    assert torch.allclose(norms, torch.ones(32), atol=1e-4)
    assert not gsae.w_enc.requires_grad and not gsae.w_dec.requires_grad


def test_training_reduces_reconstruction_error():
    inputs = _synthetic_inputs()
    quick = GSAE.train(inputs, m=32, k=4, epochs=1, batch_size=128)
    longer = GSAE.train(inputs, m=32, k=4, epochs=3, batch_size=128)
    assert longer.fvu < quick.fvu
    assert longer.fvu < 1.0


def test_train_validation():
    with pytest.raises(ValueError, match="inputs"):
        GSAE.train(torch.zeros(4), m=8, k=2)
    with pytest.raises(ValueError, match="k <= m"):
        GSAE.train(_synthetic_inputs(), m=8, k=9)


def test_dead_feature_reinit_replaces_and_normalizes():
    gsae = GSAE.train(_synthetic_inputs(), m=16, k=2, epochs=1, batch_size=128)
    fired = torch.ones(16).bool()
    fired[3] = False
    fired[7] = False
    before = gsae.w_dec.clone()
    _reinit_dead_features(
        gsae, _synthetic_inputs(), fired, torch.Generator().manual_seed(0)
    )
    changed = ~torch.isclose(gsae.w_dec, before, atol=1e-8).all(dim=1)
    assert set(changed.nonzero(as_tuple=True)[0].tolist()) == {3, 7}
    assert torch.allclose(gsae.w_dec[3].norm(), torch.tensor(1.0), atol=1e-5)


def test_dead_feature_reinit_when_dead_outnumber_inputs():
    """More dead features than inputs to draw replacements from.

    Each revived feature takes one input's residual, so a dictionary far larger
    than the pooled corpus leaves some dead features with nothing to take. They
    keep their directions until the next epoch instead of failing the pass.
    """
    inputs = _synthetic_inputs(n=12)
    gsae = GSAE.train(inputs, m=64, k=2, epochs=1, batch_size=16)
    fired = torch.zeros(64).bool()
    fired[:2] = True
    before = gsae.w_dec.clone()
    _reinit_dead_features(gsae, inputs, fired, torch.Generator().manual_seed(0))
    changed = ~torch.isclose(gsae.w_dec, before, atol=1e-8).all(dim=1)
    assert int(changed.sum()) == inputs.shape[0]
    assert torch.allclose(
        gsae.w_dec[changed].norm(dim=1), torch.ones(int(changed.sum())), atol=1e-5
    )


def test_train_with_dictionary_larger_than_corpus():
    """The same shape through the public API: m far above the number of inputs."""
    gsae = GSAE.train(_synthetic_inputs(n=32), m=256, k=2, epochs=2, batch_size=16)
    assert gsae.m == 256
    assert torch.isfinite(torch.tensor(gsae.fvu))


# ------------------------------------------------------------------- census


def test_firing_rates_match_manual_count():
    gsae = GSAE.train(_synthetic_inputs(), m=16, k=4, epochs=1, batch_size=128)
    inputs = _synthetic_inputs(n=50)
    rates = firing_rates(gsae, inputs, batch_size=16)
    manual = gsae.encode(inputs).ne(0).float().mean(dim=0)
    assert torch.allclose(rates, manual, atol=1e-6)
    # Every input selects k features, so the rates sum to k.
    assert rates.sum().item() == pytest.approx(4.0, abs=1e-4)


def test_classify_features_thresholds():
    #               introduced  turned_up  suppressed  stable  boundary
    base_rates = torch.tensor([0.01, 0.10, 0.30, 0.20, 0.05])
    rl_rates = torch.tensor([0.20, 0.40, 0.05, 0.22, 0.90])
    census = classify_features(base_rates, rl_rates)
    assert census["introduced"] == [0]
    assert census["turned_up"] == [1, 4]  # f0=0.05 is NOT < 0.05: not introduced
    assert census["suppressed"] == [2]
    assert census["stable"] == [3]
    # Every feature lands in exactly one class.
    assert sum(len(ids) for ids in census.values()) == 5


def test_classify_features_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        classify_features(torch.zeros(3), torch.zeros(4))


def test_classify_features_min_gain_guard():
    # A 3x ratio on a tiny base rate still needs an absolute gain > 0.005.
    base_rates = torch.tensor([0.001])
    rl_rates = torch.tensor([0.004])
    census = classify_features(base_rates, rl_rates)
    assert census["turned_up"] == []
    assert census["stable"] == [0]


# ------------------------------------------------------- readout and io


def test_promoted_tokens_returns_token_strings(murano_model):
    gsae = GSAE.train(
        _synthetic_inputs(d=murano_model.d_model),
        m=8,
        k=2,
        epochs=1,
        batch_size=128,
    )
    tokens = promoted_tokens(gsae, murano_model, feature=0, top_k=3)
    assert len(tokens) == 3
    assert all(isinstance(token, str) for token in tokens)


def test_gsae_save_load_roundtrip(tmp_path):
    from murano.io import load_gsae, save_gsae

    gsae = GSAE.train(_synthetic_inputs(), m=16, k=4, epochs=1, batch_size=128)
    path = tmp_path / "gsae.pt"
    save_gsae(gsae, path)
    loaded = load_gsae(path)
    assert torch.equal(loaded.w_enc, gsae.w_enc)
    assert torch.equal(loaded.w_dec, gsae.w_dec)
    assert loaded.k == gsae.k
    assert loaded.fvu == pytest.approx(gsae.fvu)


def test_save_results_dispatches_gsae(tmp_path):
    from murano.io import save_results
    from murano.results import Results

    results = Results()
    results[keys.GSAE] = GSAE.train(
        _synthetic_inputs(), m=8, k=2, epochs=1, batch_size=128
    )
    out = save_results(results, output_dir=str(tmp_path), model_id="tiny")
    assert (out / "gsae" / "gsae.pt").exists()
