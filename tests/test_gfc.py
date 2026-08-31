"""Tests for the GFC routing operator and its comparison functions."""

from __future__ import annotations

import pytest
import torch

from murano import keys
from murano.pipeline import Pipeline
from murano.steps.gradients import (
    LoadRollouts,
    RolloutBatch,
    _completion_window,
)
from murano.steps.gfc import (
    GFCOperator,
    GFCOperatorResult,
    background_subtract,
    marginal,
    read_directions,
    pairing_operator,
    per_position_overlap,
    permutation_floor,
    pairing_overlap,
)
from tests.conftest import toy_rollouts


def _toy_rollouts(n: int = 2, length: int = 14, prompt_len: int = 3) -> RolloutBatch:
    return toy_rollouts(n, length, prompt_len, seed=11)


def _random_operator(seed: int, k: int = 16, d: int = 32) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(k, d, generator=generator)


# ---------------------------------------------------------------- pure helpers


def test_read_directions_are_orthonormal():
    directions = read_directions(32, 16, seed=0)
    assert directions.shape == (16, 32)
    gram = directions @ directions.T
    assert torch.allclose(gram, torch.eye(16), atol=1e-5)


def test_read_directions_seeded():
    same = read_directions(32, 8, seed=3)
    again = read_directions(32, 8, seed=3)
    other = read_directions(32, 8, seed=4)
    assert torch.equal(same, again)
    assert not torch.allclose(same, other)


def test_read_directions_rejects_k_above_d():
    with pytest.raises(ValueError, match="d_model"):
        read_directions(16, 32)


def test_background_subtract_zeroes_margins():
    operator = _random_operator(0)
    subtracted = background_subtract(operator)
    assert torch.allclose(subtracted.sum(dim=0), torch.zeros(32), atol=1e-5)
    assert torch.allclose(subtracted.sum(dim=1), torch.zeros(16), atol=1e-5)
    # marg(R) itself is rank one.
    assert torch.linalg.matrix_rank(marginal(operator)).item() == 1


def test_pairing_operator_has_unit_strength_norm():
    operator = background_subtract(_random_operator(1))
    for rank in (1, 4, 8):
        paired = pairing_operator(operator, rank=rank)
        assert paired.norm().item() == pytest.approx(rank**0.5, abs=1e-4)


def test_pairing_operator_rejects_bad_rank():
    with pytest.raises(ValueError, match="rank"):
        pairing_operator(_random_operator(2), rank=17)


def test_self_overlap_is_one():
    operator = _random_operator(3)
    assert pairing_overlap(operator, operator) == pytest.approx(1.0, abs=1e-5)


def test_overlap_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        pairing_overlap(_random_operator(0), _random_operator(0, k=8))


def test_permutation_floor_near_zero_after_subtraction():
    """Column-shuffled operators agree only by chance once the background
    (which the shuffle preserves) is removed."""
    a = _random_operator(4)
    b = _random_operator(5)
    mean, sd = permutation_floor(a, b, draws=10)
    assert abs(mean) < 0.15
    assert sd < 0.15
    # The floor sits far below a genuine self-match.
    assert pairing_overlap(a, a) - mean > 0.8


def test_independent_operators_overlap_near_floor():
    a = _random_operator(6)
    b = _random_operator(7)
    assert abs(pairing_overlap(a, b)) < 0.3


def test_per_position_overlap_shape():
    a = torch.stack([_random_operator(i) for i in range(4)])
    curve = per_position_overlap(a, a)
    assert curve.shape == (4,)
    assert torch.allclose(curve, torch.ones(4), atol=1e-5)
    with pytest.raises(ValueError, match="shape"):
        per_position_overlap(a, a[:2])


# ------------------------------------------------------------------- the step


def test_gfc_operator_validates_arguments(murano_model):
    with pytest.raises(ValueError, match="target_layer < source_layer"):
        GFCOperator(murano_model, source_layer=0, target_layer=1)
    with pytest.raises(ValueError, match="k must be"):
        GFCOperator(murano_model, source_layer=1, target_layer=0, k=64)
    with pytest.raises(ValueError, match="per_position"):
        GFCOperator(murano_model, source_layer=1, target_layer=0, k=4, per_position=0)
    with pytest.raises(ValueError, match="window"):
        GFCOperator(murano_model, source_layer=1, target_layer=0, k=4, window=0)


def test_gfc_operator_shape_and_positivity(murano_model):
    batch = _toy_rollouts(n=2)
    result = Pipeline(
        [
            LoadRollouts(batch),
            GFCOperator(murano_model, source_layer=1, target_layer=0, k=8, window=6),
        ]
    ).run()[keys.GFC_OPERATOR]
    assert isinstance(result, GFCOperatorResult)
    # The operator is [k, d_model] with nonnegative entries (summed magnitudes).
    assert result.operator.shape == (8, murano_model.d_model)
    assert (result.operator >= 0).all()
    assert result.operator.sum() > 0
    assert result.kept is not None and result.kept.all()
    assert result.nll is not None and torch.isfinite(result.nll).all()


def test_per_position_stack_sums_to_pooled_operator(murano_model):
    """Opening the target position is a refinement, not a different object:
    summing the per-position stack over positions recovers the pooled operator
    exactly (same VJPs, same magnitudes)."""
    batch = _toy_rollouts(n=2, length=12, prompt_len=3)
    result = Pipeline(
        [
            LoadRollouts(batch),
            GFCOperator(
                murano_model,
                source_layer=1,
                target_layer=0,
                k=6,
                window=6,
                per_position=6,
            ),
        ]
    ).run()[keys.GFC_OPERATOR]
    assert result.per_position is not None
    assert result.per_position.shape == (6, 6, murano_model.d_model)
    assert torch.allclose(result.per_position.sum(dim=0), result.operator, atol=1e-5)
    assert result.position_counts is not None
    assert result.position_counts.tolist() == [2] * 6


def test_gradient_off_matches_manual_bare_direction_vjp(murano_model):
    """Gradient-off means the loss never enters: the operator must equal the
    hand-computed transport of the bare directions at every window position."""
    batch = _toy_rollouts(n=1, length=10, prompt_len=2)
    k, window = 4, 5
    result = Pipeline(
        [
            LoadRollouts(batch),
            GFCOperator(
                murano_model,
                source_layer=1,
                target_layer=0,
                k=k,
                window=window,
                gradient_off=True,
            ),
        ]
    ).run()[keys.GFC_OPERATOR]

    ids = batch.input_ids[0]
    start, end = _completion_window(ids, 2, window)
    hidden = murano_model.grad_forward(ids.unsqueeze(0), [0, 1])
    directions = read_directions(murano_model.d_model, k)
    manual = torch.zeros(k, murano_model.d_model)
    for i in range(k):
        field = torch.zeros_like(hidden[1])
        field[0, start:end] = directions[i]
        (transported,) = torch.autograd.grad(
            hidden[1], hidden[0], grad_outputs=field, retain_graph=True
        )
        manual[i] = transported[0, start:end].detach().float().abs().sum(dim=0)
    assert torch.allclose(result.operator, manual, atol=1e-4)


def test_gradient_and_gradient_off_operators_differ(murano_model):
    batch = _toy_rollouts(n=1)
    results = {}
    for gradient_off in (False, True):
        results[gradient_off] = Pipeline(
            [
                LoadRollouts(batch),
                GFCOperator(
                    murano_model,
                    source_layer=1,
                    target_layer=0,
                    k=6,
                    window=6,
                    gradient_off=gradient_off,
                ),
            ]
        ).run()[keys.GFC_OPERATOR]
    assert not torch.allclose(results[False].operator, results[True].operator)


def test_direction_chunks_produce_identical_operators(murano_model):
    """Batched, chunked, and plain per-direction VJPs are the same math.

    ``direction_chunk=None`` runs one vmap-batched VJP, ``3`` chunks it, and
    ``1`` falls back to plain unbatched VJPs; the operator must be identical
    (up to float accumulation) in every case.
    """
    batch = _toy_rollouts(n=2, length=12, prompt_len=3)
    operators = []
    for chunk in (None, 3, 1):
        result = Pipeline(
            [
                LoadRollouts(batch),
                GFCOperator(
                    murano_model,
                    source_layer=1,
                    target_layer=0,
                    k=8,
                    window=6,
                    per_position=6,
                    direction_chunk=chunk,
                ),
            ]
        ).run()[keys.GFC_OPERATOR]
        operators.append(result)
    for other in operators[1:]:
        assert torch.allclose(operators[0].operator, other.operator, atol=1e-5)
        assert torch.allclose(
            operators[0].per_position, other.per_position, atol=1e-5
        )


def test_explicit_directions_match_seeded_draw(murano_model):
    """Passing the basis explicitly reproduces the seeded internal draw."""
    batch = _toy_rollouts(n=1)
    basis = read_directions(murano_model.d_model, 6, seed=9)
    explicit = Pipeline(
        [
            LoadRollouts(batch),
            GFCOperator(
                murano_model,
                source_layer=1,
                target_layer=0,
                k=6,
                window=6,
                directions=basis,
            ),
        ]
    ).run()[keys.GFC_OPERATOR]
    seeded = Pipeline(
        [
            LoadRollouts(batch),
            GFCOperator(
                murano_model,
                source_layer=1,
                target_layer=0,
                k=6,
                window=6,
                basis_seed=9,
            ),
        ]
    ).run()[keys.GFC_OPERATOR]
    assert torch.allclose(explicit.operator, seeded.operator, atol=1e-6)


def test_explicit_directions_validation(murano_model):
    with pytest.raises(ValueError, match="directions must be"):
        GFCOperator(
            murano_model,
            source_layer=1,
            target_layer=0,
            k=4,
            directions=torch.eye(3),
        )
    with pytest.raises(ValueError, match="unit-norm"):
        GFCOperator(
            murano_model,
            source_layer=1,
            target_layer=0,
            k=4,
            directions=torch.ones(4, murano_model.d_model),
        )
    with pytest.raises(ValueError, match="direction_chunk"):
        GFCOperator(
            murano_model,
            source_layer=1,
            target_layer=0,
            k=4,
            direction_chunk=0,
        )


def test_gfc_operator_deterministic_and_self_consistent(murano_model):
    batch = _toy_rollouts(n=2)
    step = GFCOperator(murano_model, source_layer=1, target_layer=0, k=8, window=6)
    first = Pipeline([LoadRollouts(batch), step]).run()[keys.GFC_OPERATOR]
    second = Pipeline([LoadRollouts(batch), step]).run()[keys.GFC_OPERATOR]
    assert torch.equal(first.operator, second.operator)


def test_pipeline_validates_wiring(murano_model):
    batch = _toy_rollouts(n=1)
    pipeline = Pipeline(
        [
            LoadRollouts(batch),
            GFCOperator(murano_model, source_layer=1, target_layer=0, k=4),
        ]
    )
    pipeline.validate()
    # Without the loader the chain is broken and validation says so.
    with pytest.raises(KeyError, match="rollouts"):
        Pipeline(
            [GFCOperator(murano_model, source_layer=1, target_layer=0, k=4)]
        ).validate()


def test_gfc_operator_save_load_roundtrip(murano_model, tmp_path):
    from murano.io import load_gfc_operator, save_gfc_operator

    batch = _toy_rollouts(n=1)
    result = Pipeline(
        [
            LoadRollouts(batch),
            GFCOperator(
                murano_model,
                source_layer=1,
                target_layer=0,
                k=4,
                window=5,
                per_position=5,
            ),
        ]
    ).run()[keys.GFC_OPERATOR]
    path = tmp_path / "gfc_operator.pt"
    save_gfc_operator(result, path)
    loaded = load_gfc_operator(path)
    assert torch.equal(loaded.operator, result.operator)
    assert loaded.per_position is not None
    assert torch.equal(loaded.per_position, result.per_position)
    assert loaded.gradient_off == result.gradient_off
    assert (loaded.source_layer, loaded.target_layer) == (1, 0)


def test_save_results_dispatches_gfc_operator(murano_model, tmp_path):
    from murano.io import save_results

    batch = _toy_rollouts(n=1)
    results = Pipeline(
        [
            LoadRollouts(batch),
            GFCOperator(murano_model, source_layer=1, target_layer=0, k=4),
        ]
    ).run()
    out = save_results(results, output_dir=str(tmp_path), model_id="tiny")
    assert (out / "gfc" / "gfc_operator.pt").exists()
