"""Scientific geometry, vMF, stability, and reporting tests for FEGA."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from murano import keys
from murano.fega.artifacts import FEGAEffectStore, FEGAFeatureEffects, FEGAVMFResult
from murano.fega.config import FEGAConfig
from murano.fega.geometry import compute_geometry_metrics, geometry_metrics_from_kernel
from murano.fega.reporting import GeometryRecord, _qualified_flags, classify_geometry
from murano.fega.stability import (
    derive_seed,
    g_orthonormal_basis,
    low_context_protocol,
    principal_angles_degrees,
    subspace_resample_indices,
)
from murano.fega.vmf import (
    NoFiniteVMFCandidate,
    VMFFit,
    VMFSelectedFit,
    assignment_stability,
    fit_vmf,
    select_vmf,
)
from murano.fega.vmf import _derived_seed, _log_normalizer_plus_kappa
from murano.results import Results
from murano.steps.fega_analysis import (
    FEGAGeometryMetrics,
    FEGAGeometryReporting,
    FEGAStability,
)


def test_source_two_mode_fit_keeps_component_order() -> None:
    """The source's separated two-mode example must retain component ordering."""

    # Keep the tiny source test inline; only real derived data belongs in fixtures.
    rows = np.asarray([[1.0, 0.0]] * 10 + [[0.0, 1.0]] * 10)
    fit = fit_vmf(
        rows,
        2,
        0,
        n_init=1,
        max_iter=5,
    )
    expected_responsibilities = np.asarray(
        [[0.0] * 10 + [1.0] * 10, [1.0] * 10 + [0.0] * 10]
    )
    np.testing.assert_allclose(fit.weights, [0.5, 0.5])
    np.testing.assert_allclose(fit.concentrations, [1.0e10, 1.0e10])
    np.testing.assert_allclose(
        fit.responsibilities, expected_responsibilities, atol=1e-12
    )
    np.testing.assert_array_equal(fit.labels, [1] * 10 + [0] * 10)
    assert fit.log_likelihood == pytest.approx(198.0167950238622, abs=1e-10)


def test_dense_float32_fit_matches_source_operation_order() -> None:
    """Float32 preprocessing and scalar M-step reductions must match the source."""

    # Freeze a source-scored high-dimensional fixture that exposes reduction order.
    random_state = np.random.RandomState(19)
    centers = np.zeros((12, 64))
    centers[:6, 0] = 3.0
    centers[6:, 1] = 3.0
    rows = random_state.normal(centers, 0.3).astype(np.float32)
    fit = fit_vmf(rows, 2, seed=41, n_init=1, max_iter=8)

    np.testing.assert_allclose(fit.weights, [0.5, 0.5], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(
        fit.concentrations,
        [190.9758631557932, 164.71840083174075],
        rtol=3e-15,
        atol=0.0,
    )
    np.testing.assert_array_equal(fit.labels, [0] * 6 + [1] * 6)
    np.testing.assert_allclose(
        fit.responsibilities,
        [
            [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                6.780860760920258e-69,
                5.6906392781807743e-64,
                6.050300944034788e-76,
                7.234328920737285e-64,
                8.705836727825392e-66,
                6.799012253903185e-59,
            ],
            [
                1.5886342636807923e-63,
                1.4260423809482717e-57,
                1.299152429209502e-60,
                1.4089508999744281e-55,
                3.315384425672932e-53,
                8.781547549507773e-62,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
        ],
        rtol=2e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        fit.means[:, :8],
        [
            [
                0.9551168938221681,
                0.04391681754260325,
                0.013062363172993528,
                0.03702772736351059,
                -0.013176078617325164,
                -0.03373886109536264,
                0.010265503604358244,
                0.01103949622931166,
            ],
            [
                -0.002373136413608783,
                0.945364808732499,
                -0.017719319946326192,
                -0.02284649816189639,
                -0.04385865201014076,
                0.015876602185076215,
                -0.004371571802901079,
                0.06775845741579706,
            ],
        ],
        rtol=2e-12,
        atol=0.0,
    )
    assert fit.log_likelihood == pytest.approx(940.8325658808001, abs=1e-10)


def test_selection_is_seeded_and_worker_invariant() -> None:
    """Parallel candidate evaluation must not alter seeded model selection."""

    # Fit separated directions twice and across both supported worker counts.
    angles = np.r_[np.linspace(-0.1, 0.1, 12), np.linspace(1.4, 1.7, 12)]
    data = np.column_stack((np.cos(angles), np.sin(angles)))
    serial = select_vmf(data, (1, 2, 3), seed=17, n_init=2, n_jobs=1)
    repeated = select_vmf(data, (1, 2, 3), seed=17, n_init=2, n_jobs=1)
    parallel = select_vmf(data, (1, 2, 3), seed=17, n_init=2, n_jobs=4)
    np.testing.assert_array_equal(serial.selected.labels, repeated.selected.labels)
    np.testing.assert_array_equal(serial.selected.labels, parallel.selected.labels)
    assert serial.selected.log_likelihood == parallel.selected.log_likelihood
    assert not hasattr(serial.selected, "means")
    assert all(not hasattr(candidate, "means") for candidate in serial.candidates)

    # A one-component fit has no assignment boundary whose stability can vary.
    one_mode_angles = np.linspace(-0.2, 0.2, 8)
    one_mode_data = np.column_stack((np.cos(one_mode_angles), np.sin(one_mode_angles)))
    one_mode = fit_vmf(one_mode_data, 1, seed=3, n_init=1, max_iter=20)
    assert assignment_stability(one_mode_data, one_mode, seed=3) is None


def test_bic_tolerance_prefers_smaller_component_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BIC values inside tolerance must keep the earlier, smaller candidate."""

    # Replace fitting only to isolate the documented selection tie rule.
    data = np.tile(np.array([[1.0, 0.0]]), (8, 1))

    def fake_fit_many(
        directions: np.ndarray, n_components: int, seed: int, n_init: int, max_iter: int
    ) -> VMFFit:
        """Return a minimal candidate whose larger k is insignificantly better."""

        # Encode a sub-tolerance BIC improvement for the larger candidate.
        del seed, n_init, max_iter
        return VMFFit(
            n_components,
            np.full(n_components, 1 / n_components),
            np.zeros((n_components, directions.shape[1])),
            np.zeros(n_components),
            np.zeros(len(directions), dtype=int),
            np.full((n_components, len(directions)), 1 / n_components),
            0.0,
            10.0 - n_components * 1e-10,
            True,
            1,
        )

    monkeypatch.setattr("murano.fega.vmf._fit_many", fake_fit_many)
    selection = select_vmf(data, (2, 1, 2), bic_tolerance=1e-9)
    assert selection.selected.n_components == 1


def test_failed_initialization_does_not_abort_fixed_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One numerical start must not hide a later finite source-scheduled start."""

    # Fail the first start and return a finite fit from the second scheduled start.
    calls = 0

    def fake_fit_once(
        directions: np.ndarray, n_components: int, seed: int, max_iter: int
    ) -> VMFFit:
        """Expose whether the fixed initialization budget continues after failure."""

        # Keep the replacement minimal while retaining every public fit field.
        nonlocal calls
        del seed, max_iter
        calls += 1
        if calls == 1:
            raise FloatingPointError("first start failed")
        return VMFFit(
            n_components,
            np.ones(1),
            np.ones((1, directions.shape[1])),
            np.ones(1),
            np.zeros(len(directions), dtype=int),
            np.ones((1, len(directions))),
            1.0,
            2.0,
            True,
            1,
        )

    monkeypatch.setattr("murano.fega.vmf._fit_once", fake_fit_once)
    fit = fit_vmf(np.tile(np.array([[1.0, 0.0]]), (8, 1)), 1, 7, n_init=2)
    assert calls == 2
    assert fit.log_likelihood == 1.0

    def fail_programming_error(
        directions: np.ndarray, n_components: int, seed: int, max_iter: int
    ) -> VMFFit:
        """Represent a programming error that must not become fit evidence."""

        # Keep the injected failure at the same shared initialization boundary.
        del directions, n_components, seed, max_iter
        raise ValueError("shape bug")

    monkeypatch.setattr("murano.fega.vmf._fit_once", fail_programming_error)
    with pytest.raises(ValueError, match="shape bug"):
        fit_vmf(np.tile(np.array([[1.0, 0.0]]), (8, 1)), 1, 7, n_init=1)


def test_failed_candidates_retain_their_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An all-failed feature must expose every attempted component count."""

    # Fail numerically at the shared fit boundary and inspect the recorded schedule.
    def fail_fit(*args: object, **kwargs: object) -> VMFFit:
        """Represent a source-style fixed-candidate numerical failure."""

        # Keep the failure numeric so programming errors still propagate.
        del args, kwargs
        raise FloatingPointError("numeric failure")

    monkeypatch.setattr("murano.fega.vmf._fit_many", fail_fit)
    data = np.tile(np.array([[1.0, 0.0]]), (8, 1))
    with pytest.raises(NoFiniteVMFCandidate) as caught:
        select_vmf(data, (1, 2, 3), n_init=1)
    assert [item.n_components for item in caught.value.candidates] == [1, 2, 3]
    assert {item.status for item in caught.value.candidates} == {"fit_failed"}


def test_large_input_warns_without_truncating() -> None:
    """The memory warning must remain advisory and retain every direction row."""

    # Fit all 65 rows and verify the warning includes the float32 byte estimate.
    data = np.tile(np.array([[1.0, 0.0]]), (65, 1))
    with pytest.warns(RuntimeWarning, match=r"65.*520"):
        fit = fit_vmf(data, 1, seed=0, n_init=1, max_iter=2)
    assert fit.responsibilities.shape == (1, 65)


def test_assignment_subset_uses_full_width_source_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The source's 64-bit subset seed must retain its exact sampled membership."""

    # Capture the scheduled subset while replacing only its numerical refit.
    assert _derived_seed(3, 2, 0, "subset") == 10_635_222_172_975_432_093
    angles = np.arange(8) / 10
    data = np.column_stack((np.cos(angles), np.sin(angles)))
    fit = VMFSelectedFit(
        2,
        np.asarray([0.5, 0.5]),
        np.asarray([1.0, 1.0]),
        np.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
        0.0,
        0.0,
        True,
        1,
    )
    sampled: list[int] = []

    def capture_refit(
        directions: np.ndarray,
        n_components: int,
        seed: int,
        n_init: int,
        max_iter: int,
    ) -> VMFFit:
        """Return matching labels after recording the source-scheduled rows."""

        # Recover original row identities from their unique unit directions.
        del seed, n_init, max_iter
        sampled.extend(np.argmax(directions @ data.T, axis=1).tolist())
        labels = fit.labels[sampled]
        return VMFFit(
            n_components,
            np.full(n_components, 1 / n_components),
            np.ones((n_components, directions.shape[1])),
            np.ones(n_components),
            labels,
            np.full((n_components, len(directions)), 1 / n_components),
            0.0,
            0.0,
            True,
            1,
        )

    monkeypatch.setattr("murano.fega.vmf._fit_many", capture_refit)
    assert assignment_stability(data, fit, seed=3, fraction=0.75, rounds=1) == 1.0
    assert sampled == [1, 2, 4, 5, 6, 7]


def test_failed_stability_refit_makes_the_aggregate_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One required failed replicate must not be hidden by successful refits."""

    # Match the source contract captured by cached feature 14513: 7/8 is unavailable.
    angles = np.linspace(-0.2, 1.7, 8)
    data = np.column_stack((np.cos(angles), np.sin(angles)))
    fit = VMFSelectedFit(
        2,
        np.asarray([0.5, 0.5]),
        np.asarray([1.0, 1.0]),
        np.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
        0.0,
        0.0,
        True,
        1,
    )
    calls = 0

    def fail_one_refit(
        directions: np.ndarray,
        n_components: int,
        seed: int,
        n_init: int,
        max_iter: int,
    ) -> VMFFit:
        """Fail one scheduled replicate and return a finite fit for the rest."""

        # Preserve the complete fixed schedule while injecting one numeric failure.
        nonlocal calls
        del seed, n_init, max_iter
        calls += 1
        if calls == 2:
            raise FloatingPointError("source-style numeric failure")
        labels = np.zeros(len(directions), dtype=int)
        return VMFFit(
            n_components,
            np.full(n_components, 1 / n_components),
            np.ones((n_components, directions.shape[1])),
            np.ones(n_components),
            labels,
            np.full((n_components, len(directions)), 1 / n_components),
            0.0,
            0.0,
            True,
            1,
        )

    monkeypatch.setattr("murano.fega.vmf._fit_many", fail_one_refit)
    assert assignment_stability(data, fit, seed=3, rounds=8) is None
    assert calls == 8


def test_gemma_vocabulary_normalizer_preserves_source_error_contract() -> None:
    """The 256k-dimensional path must preserve source quadrature failures."""

    # Freeze the source failure observed for this high-concentration SciPy path.
    with pytest.raises(FloatingPointError, match="error contract"):
        _log_normalizer_plus_kappa(
            np.asarray([1_149_515.6040200791]), dimension=256_000
        )

    # Neighboring finite source values still use the exact fallback successfully.
    actual = _log_normalizer_plus_kappa(np.asarray([1.0e6, 1.0e10]), dimension=256_000)
    np.testing.assert_allclose(
        actual,
        [1541311.844278737, 2712050.8797322507],
        rtol=1.0e-13,
    )


def test_geometry_metrics_match_analytic_identity_gram_case() -> None:
    """Lock ray, span, balance, centering, and ambient completion formulas."""
    # Two opposing unit directions have exact one-axis centered geometry.
    vectors = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    result = compute_geometry_metrics(
        vectors, np.eye(2, dtype=np.float32), k_values=(1, 2), residual_k_values=(1,)
    )

    # These values fail if centering, sign balance, or zero completion changes.
    np.testing.assert_allclose(result.eigenvalues, [2.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(result.centered_eigenvalues, [2.0, 0.0], atol=1e-12)
    assert result.c_ray == -1.0
    assert result.r2 == 0.0
    assert result.s_span[1] == 2.0 / (2.0 + 1e-12)
    assert result.u_span[1] == 2.0 / (2.0 + 1e-12)
    assert result.d_span[1] == 0.0
    assert result.d_span[2] is None
    assert result.b_axis == 0.5
    assert result.e_res == 2.0 / (2.0 + 1e-12)
    assert result.s_res[1] == 2.0 / (2.0 + 1e-12)


def test_real_feature_kernel_matches_curated_source_metrics() -> None:
    """Match the compact feature-33760 kernel against source FEGA metadata."""
    # Load the committed compact fixture without the source repository.
    fixture_dir = Path(__file__).parent / "fixtures" / "fega"
    with np.load(fixture_dir / "real_feature_33760.npz") as fixture:
        kernel = fixture["kernel"]
        magnitudes = fixture["magnitudes"]
    metadata = json.loads((fixture_dir / "real_feature_33760.json").read_text())
    source_metrics = metadata["source_metrics"]
    expected = source_metrics["family_metrics"]

    # Factor the PSD kernel so the public D G D.T path is checked too.
    values, vectors = np.linalg.eigh((kernel + kernel.T) / 2.0)
    positive = values > 0.0
    coordinates = vectors[:, positive] * np.sqrt(values[positive])
    result = compute_geometry_metrics(coordinates, np.eye(coordinates.shape[1]))
    compact = geometry_metrics_from_kernel(kernel, ambient_dim=coordinates.shape[1])

    # Compare the frozen source point metrics and both computation routes.
    assert result.n_valid == source_metrics["n_valid"]
    np.testing.assert_allclose(result.c_ray, expected["c_ray"], rtol=1e-7)
    np.testing.assert_allclose(result.s_span[1], expected["s_span_1"], rtol=1e-7)
    np.testing.assert_allclose(result.e_res, expected["e_res"], rtol=1e-7)
    np.testing.assert_allclose(
        result.eigenvalues, compact.eigenvalues, rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(
        result.centered_eigenvalues,
        compact.centered_eigenvalues,
        rtol=1e-9,
        atol=1e-12,
    )

    # Run the public reporting phases and match the source label and qualifiers.
    feature = FEGAFeatureEffects(
        feature_id=metadata["feature_id"],
        directions=torch.from_numpy(coordinates).to(torch.float32),
        magnitudes=torch.from_numpy(magnitudes),
        context_indices=tuple(range(len(coordinates))),
        feature_activations=torch.ones(len(coordinates)),
        retained_mask=(True,) * len(coordinates),
    )
    effects = FEGAEffectStore(
        features={feature.feature_id: feature},
        gram=torch.eye(coordinates.shape[1]),
        unembedding_fingerprint="source-fixture",
        analysis_id="source-feature-33760",
    )
    results = Results()
    results[keys.FEGA_EFFECTS] = effects
    FEGAGeometryMetrics(FEGAConfig())(results)
    results[keys.FEGA_VMF] = FEGAVMFResult(
        {},
        {feature.feature_id: None},
        effects.unembedding_fingerprint,
        effects.analysis_id,
    )
    FEGAStability(FEGAConfig())(results)
    FEGAGeometryReporting()(results)
    reported = results[keys.FEGA_REPORTING].features[feature.feature_id]
    assert reported.primary_label == source_metrics["primary_label"]
    assert reported.confidence == source_metrics["label_confidence"]
    assert reported.secondary_flags == tuple(source_metrics["secondary_flags"])
    assert reported.global_flags == ("long_tail_spectrum",)


def test_eligibility_boundaries_and_strict_primary_priority() -> None:
    """Exact eligibility bounds pass and directed evidence wins all candidates."""
    # Supply exact-boundary evidence for every class to lock eligibility and order.
    record = GeometryRecord(
        n_valid=8,
        zero_filter_frac=0.30,
        c_ray=0.80,
        b_axis=0.15,
        s_span={1: 0.80, 2: 0.90},
        u_span={2: 0.08},
        d_span={2: 0.60},
        r_span_pr=1.60,
        selected_mode_count=2,
        delta_mix=0.10,
        mode_mass_min=0.10,
        min_mode_c_ray=0.70,
        mode_kappa_min=1.0,
        assignment_stability=0.80,
        e_res=0.10,
        s_res={2: 0.80},
        r_ctr_pr=1.50,
    )

    eligible = classify_geometry(record)
    ineligible = classify_geometry(replace(record, n_valid=7))

    assert eligible.primary_label == "directed_ray"
    assert eligible.span_selected_k == 2
    assert eligible.residual_selected_k == 2
    assert eligible.flags["multi_mode_directional_geometry"]
    assert set(eligible.secondary_flags) == {
        "directed_ray_with_lowD_residual",
        "lowD_candidate_blocked",
        "residual_selected_k",
        "span_selected_k",
    }
    assert ineligible.primary_label == "insufficient_effect_evidence"
    assert not ineligible.flags["eligible"]


def test_reporting_uses_attempted_effect_rows_for_filter_fraction() -> None:
    """Seven rejected rows out of twenty must fail FEGA's 30% quality gate."""
    # Geometry receives thirteen normalized rows, while the mask retains the denominator.
    rows = torch.tensor([[1.0, 0.0]] * 13)
    feature = FEGAFeatureEffects(
        feature_id=7,
        directions=rows,
        magnitudes=torch.ones(13),
        context_indices=tuple(range(13)),
        feature_activations=torch.ones(13),
        retained_mask=(True,) * 13 + (False,) * 7,
    )
    effects = FEGAEffectStore(
        features={7: feature},
        gram=torch.eye(2),
        unembedding_fingerprint="test-unembedding",
        analysis_id="filter-fraction",
    )
    results = Results()
    results[keys.FEGA_EFFECTS] = effects
    FEGAGeometryMetrics(FEGAConfig())(results)
    results[keys.FEGA_VMF] = FEGAVMFResult(
        {}, {7: None}, effects.unembedding_fingerprint, effects.analysis_id
    )
    FEGAStability(FEGAConfig())(results)
    FEGAGeometryReporting()(results)

    # The retained geometry alone looks directed, but 35% rejected rows make it insufficient.
    assert results[keys.FEGA_REPORTING].features[7].primary_label == (
        "insufficient_effect_evidence"
    )


def test_source_candidate_fallbacks_keep_their_anchored_dimension() -> None:
    """Keep source span/residual candidates when only their anchor passes."""
    # A span candidate needs its captured variance anchor even if strict rank fails.
    span = classify_geometry(
        GeometryRecord(
            n_valid=64,
            zero_filter_frac=0.0,
            c_ray=0.2,
            b_axis=0.0,
            s_span={1: 0.4, 2: 0.7, 3: 0.8, 4: 0.85, 8: 0.91},
            u_span={8: 0.02},
            d_span={8: 0.9},
            r_span_pr=1.7,
            r_span_ent=1.8,
            selected_mode_count=1,
            delta_mix=0.0,
            e_res=0.05,
        )
    )
    assert (span.primary_label, span.selected_k, span.selection_mode) == (
        "global_kD_directional_subspace",
        8,
        "fallback",
    )
    assert set(span.secondary_flags) == {"lowD_candidate_blocked", "span_selected_k"}

    # A centered residual candidate likewise keeps the smallest anchored k.
    residual = classify_geometry(
        GeometryRecord(
            n_valid=35,
            zero_filter_frac=0.0,
            c_ray=0.2,
            b_axis=0.0,
            s_span={1: 0.4, 2: 0.6, 3: 0.7, 4: 0.75, 8: 0.8},
            r_span_pr=2.0,
            r_span_ent=2.1,
            selected_mode_count=1,
            delta_mix=0.0,
            e_res=0.88,
            s_res={1: 0.4, 2: 0.7, 3: 0.75, 4: 0.81},
            r_ctr_pr=1.9,
        )
    )
    assert (residual.primary_label, residual.selected_k, residual.selection_mode) == (
        "residual_lowD_k",
        4,
        "fallback",
    )
    assert residual.secondary_flags == ("residual_selected_k",)


def test_source_point_flags_cover_remaining_candidate_families() -> None:
    """Keep the source diagnostics for diffuse, blocked mixture, and high-D points."""
    # Exercise the three candidate-specific flag branches not covered by the fixture.
    one_d = classify_geometry(
        GeometryRecord(
            n_valid=8,
            zero_filter_frac=0.0,
            c_ray=0.5,
            s_span={1: 0.9},
        )
    )
    multimode = classify_geometry(
        GeometryRecord(
            n_valid=8,
            zero_filter_frac=0.0,
            s_span={1: 0.4},
            selected_mode_count=2,
            delta_mix=0.10,
            mode_mass_min=0.10,
            min_mode_c_ray=0.60,
            mode_kappa_min=1.0,
        )
    )
    high_d = classify_geometry(
        GeometryRecord(
            n_valid=8,
            zero_filter_frac=0.0,
            s_span={1: 0.4, 2: 0.5, 3: 0.6, 4: 0.7, 8: 0.8},
            selected_mode_count=1,
            delta_mix=0.0,
            e_res=0.05,
        )
    )

    assert set(one_d.secondary_flags) == {"b_axis_missing", "oneD_not_ray_not_axis"}
    assert multimode.flags["multi_mode_directional_geometry"]
    assert set(multimode.secondary_flags) == {
        "assignment_stability_missing",
        "mode_c_ray_failed",
        "multimode_candidate_blocked",
    }
    assert high_d.secondary_flags == ("positive_highD_evidence",)


def test_mixed_unavailable_stability_keeps_both_selected_family_flags() -> None:
    """Observed instability must not hide unavailable selected-family replicates."""
    # Selected-family sample/leave outcomes qualify the family, not global point flags.
    secondary, global_flags = _qualified_flags(
        (),
        (),
        {
            "decision": "unstable",
            "protocol": {"status": "exploratory"},
            "groups": {"status": "ok"},
            "leave_out": {"status": "unstable", "unavailable_count": 1},
            "sample_size": {"status": "unstable", "unavailable_count": 0},
        },
    )
    assert secondary == (
        "selected_family_evidence_unavailable",
        "selected_family_unstable",
    )
    assert global_flags == ()


def test_stability_primitives_are_deterministic_and_metric_correct() -> None:
    """Lock protocol boundaries, seed mixing, basis geometry, and resampling."""
    # Verify every deterministic contract in one compact numerical example.
    expected_seed = (7 + 3 * 104729 + (1 * ord("x") + 2 * ord("y")) * 37 + 11) % (
        2**31 - 1
    )
    assert derive_seed(7, 3, "xy", salt=11) == expected_seed
    assert [low_context_protocol(n) for n in (7, 8, 15, 16, 31, 32)] == [
        {"status": "insufficient_contexts", "protocol": "descriptive", "n_valid": 7},
        {"status": "exploratory", "protocol": "leave_out_sensitivity", "n_valid": 8},
        {"status": "exploratory", "protocol": "leave_out_sensitivity", "n_valid": 15},
        {
            "status": "exploratory",
            "protocol": "exploratory_subsampling",
            "n_valid": 16,
        },
        {
            "status": "exploratory",
            "protocol": "exploratory_subsampling",
            "n_valid": 31,
        },
        {"status": "ok", "protocol": "principal_angle", "n_valid": 32},
    ]

    gram = np.diag([2.0, 1.0, 3.0])
    result = g_orthonormal_basis(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), gram)
    basis = result["basis"]
    np.testing.assert_allclose(basis.T @ gram @ basis, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(
        principal_angles_degrees(basis, basis, gram, 2), [0.0, 0.0], atol=2e-6
    )

    first = subspace_resample_indices(10, 0.41, 4, seed=expected_seed)
    second = subspace_resample_indices(10, 0.41, 4, seed=expected_seed)
    assert all(len(indices) == 5 for indices in first)
    assert all(
        np.array_equal(left, right) for left, right in zip(first, second, strict=True)
    )
