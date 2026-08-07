"""Scientific behavior checks for the notebook-local FEGA method."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from murano import PromptBatch, keys as murano_keys
from murano.backend import ModelBackend
from murano.results import Results
from murano.steps.sae import SAEModel

from notebooks.reproductions.fega.fega_method import stability as stability_module
from notebooks.reproductions.fega.fega_method import keys
from notebooks.reproductions.fega.fega_method.analysis import (
    FEGAVisualize,
    _record,
)
from notebooks.reproductions.fega.fega_method.artifacts import (
    FEGAEffectStore,
    FEGAFeatureEffects,
    FEGAVMFResult,
)
from notebooks.reproductions.fega.fega_method.geometry import GeometryMetrics
from notebooks.reproductions.fega.fega_method.effects import normalize_delta_rows
from notebooks.reproductions.fega.fega_method.config import FEGAConfig
from notebooks.reproductions.fega.fega_method.pipeline import (
    FEGAComputeEffect,
    FEGADataPrep,
)
from notebooks.reproductions.fega.fega_method.reporting import (
    GeometryRecord,
    classify_geometry,
    qualify_geometry,
)
from notebooks.reproductions.fega.fega_method.stability import (
    _subset_protocol,
)
from notebooks.reproductions.fega.fega_method.vmf import (
    VMFSelectedFit,
    VMFSelection,
    _source_c_ray,
    feature_seed,
    select_vmf,
)


def _historical_metric_rows(directions: torch.Tensor) -> torch.Tensor:
    """Normalize contiguous CPU float32 rows with the historical Torch path."""
    # Normalize each source row independently before the reporting reduction.
    rows = directions.to(device="cpu", dtype=torch.float32).contiguous()
    return torch.stack(
        [row * (1.0 / torch.linalg.vector_norm(row)) for row in rows]
    ).contiguous()


def _historical_c_ray(unit_rows: torch.Tensor) -> float | None:
    """Match the source float32 sum-vector ray reduction exactly."""
    # Keep the source Torch accumulation order and float64 terminal reduction.
    rows = unit_rows.to(device="cpu", dtype=torch.float32).contiguous()
    n_valid = int(rows.shape[0])
    if n_valid < 2:
        return None
    summed_norm_sq = rows.sum(dim=0).square().sum(dtype=torch.float64).item()
    return float((summed_norm_sq - float(n_valid)) / float(n_valid * (n_valid - 1)))


def _historical_reporting_metrics(
    unit_rows: torch.Tensor,
    labels: np.ndarray,
    weights: np.ndarray,
    concentrations: np.ndarray,
) -> dict[str, float]:
    """Derive the retained source reporting scalars from a float32 unit cloud."""
    # Follow the source mode order and mix only defined within-mode ray values.
    label_ids = np.asarray(labels, dtype=np.int64)
    fit_weights = np.asarray(weights, dtype=np.float64)
    fit_concentrations = np.asarray(concentrations, dtype=np.float64)
    global_c_ray = _historical_c_ray(unit_rows)
    assert global_c_ray is not None
    mode_rays = [
        _historical_c_ray(unit_rows[torch.from_numpy(label_ids == mode)])
        for mode in range(len(fit_weights))
    ]
    assert all(value is not None for value in mode_rays)
    mode_ray_values = np.asarray(mode_rays, dtype=np.float64)
    return {
        "delta_mix": float(np.dot(fit_weights, mode_ray_values) - global_c_ray),
        "mode_mass_min": float(np.min(fit_weights)),
        "min_mode_c_ray": float(np.min(mode_ray_values)),
        "mode_kappa_min": float(np.min(fit_concentrations)),
    }


def _old_gram_delta_mix(
    directions: torch.Tensor, labels: np.ndarray, weights: np.ndarray
) -> float:
    """Reproduce the removed float64 Gram-based delta_mix path."""
    # This is the pre-fix path: cast rows to float64, form D D^T, then reduce off-diagonals.
    rows64 = directions.to(torch.float64)
    kernel = (rows64 @ rows64.T).numpy()
    label_ids = np.asarray(labels, dtype=np.int64)
    count = len(kernel)
    global_c_ray = float((kernel.sum() - np.trace(kernel)) / (count * (count - 1)))
    rays: list[float] = []
    for mode in range(len(weights)):
        indices = np.flatnonzero(label_ids == mode)
        block = kernel[np.ix_(indices, indices)]
        rays.append(
            float((block.sum() - np.trace(block)) / (len(indices) * (len(indices) - 1)))
        )
    return float(
        np.dot(np.asarray(weights, dtype=np.float64), np.asarray(rays)) - global_c_ray
    )


class FakeSAE:
    """Small residual-post SAE with observable reconstruction and ablation."""

    release = "fake-release"
    sae_id = "fake-sae"
    n_features = 2
    metadata = SimpleNamespace(hook_name="blocks.0.hook_resid_post", hook_layer=0)

    def encode(self, residual: Tensor) -> Tensor:
        """Shift residual rows into deterministic latent codes."""
        # Ensure full reconstruction differs visibly from the captured residual.
        return residual + 1.0

    def decode(self, latent: Tensor) -> Tensor:
        """Decode latent rows with a deterministic non-identity map."""
        # Keep the feature-zeroing effect exactly inspectable.
        return latent * 2.0


class TinyModel(nn.Module):
    """Tiny hookable model used to exercise the complete intervention seam."""

    def __init__(self) -> None:
        """Create one residual layer and a diagonal output embedding."""
        # Make the pre-output residual distinguishable from returned logits.
        super().__init__()
        self.layer = nn.Identity()
        self.output_embedding = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.output_embedding.weight.copy_(torch.diag(torch.tensor([3.0, 2.0])))

    def forward(self, input_ids: Tensor, **_: object) -> Tensor:
        """Map integer tokens to two-dimensional residual rows."""
        # Expose mixed sequence lengths while preserving exact float32 arithmetic.
        values = input_ids.to(torch.float32)
        hidden = torch.stack((values, values + 1.0), dim=-1)
        return self.output_embedding(self.layer(hidden))

    def get_output_embeddings(self) -> nn.Module:
        """Return the hookable output projection used by FEGA."""
        # Match the Hugging Face causal-LM accessor.
        return self.output_embedding


class TinyTokenizer:
    """Tokenize prompt length into mixed-length integer sequences."""

    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, prompts: list[str], **_: object) -> dict[str, list[list[int]]]:
        """Return one nonempty unpadded token row per prompt."""
        # Let FEGA perform the left padding and target-position translation.
        return {"input_ids": [list(range(1, len(prompt) + 1)) for prompt in prompts]}


class TinyBackend:
    """Expose the minimal model surface required by the FEGA effect steps."""

    model_id = "tiny-model"
    n_layers = 1
    d_model = 2

    def __init__(self) -> None:
        """Create one shared raw model and tokenizer."""
        # Use the same weights for baseline and feature-zeroed passes.
        self.raw_model = TinyModel()
        self.tokenizer = TinyTokenizer()

    @property
    def unembed_weight(self) -> Tensor:
        """Return vocabulary rows in the model's canonical order."""
        # Reuse the exact output projection read during the forward pass.
        return self.raw_model.output_embedding.weight

    def raw_layer(self, index: int) -> nn.Module:
        """Return the only supported residual layer."""
        # Fail loudly if the public SAE metadata points elsewhere.
        if index != 0:
            raise IndexError(index)
        return self.raw_model.layer


def test_intervention_left_padding_float32_batch_invariance() -> None:
    """Mixed-length effects must preserve sign, order, dtype, and values by batch."""
    # Run the real preparation and intervention loops at two batch boundaries.
    stores = []
    for batch_size in (1, 3):
        backend = cast(ModelBackend, TinyBackend())
        sae = cast(SAEModel, FakeSAE())
        results = Results()
        results[murano_keys.PROMPTS] = PromptBatch(["a", "bb", "cccc"], source="unit")
        FEGADataPrep(
            backend,
            FakeSAE.release,
            FakeSAE.sae_id,
            [0],
            position=(0, 1, 2),
            batch_size=batch_size,
            sae_model=sae,
        )(results)
        FEGAComputeEffect(backend, batch_size=batch_size, sae_model=sae)(results)
        stores.append(results[keys.EFFECTS])

    # Compare the scientific population and normalized float32 effect cloud.
    left, right = stores
    assert left.features[0].context_indices == right.features[0].context_indices
    assert left.features[0].retained_mask == right.features[0].retained_mask
    assert left.features[0].directions.dtype == torch.float32
    assert left.features[0].magnitudes.dtype == torch.float32
    torch.testing.assert_close(
        left.features[0].directions,
        torch.tensor([[-1.0 / 3.0, 0.0]] * 3),
        rtol=0.0,
        atol=1.0e-7,
    )
    assert torch.equal(
        left.features[0].magnitudes.sort().values, torch.tensor([12.0, 18.0, 24.0])
    )
    assert torch.equal(left.features[0].directions, right.features[0].directions)
    assert torch.equal(left.features[0].magnitudes, right.features[0].magnitudes)


def test_explicit_preselected_contexts_preserve_order() -> None:
    """Explicit FEGA row picks must bypass activation reranking and stay ordered."""
    # Reverse the naturally increasing activations and keep that exact order.
    backend = cast(ModelBackend, TinyBackend())
    sae = cast(SAEModel, FakeSAE())
    results = Results()
    results[murano_keys.PROMPTS] = PromptBatch(["a", "bb", "cccc"], source="unit")

    FEGADataPrep(
        backend,
        FakeSAE.release,
        FakeSAE.sae_id,
        [0],
        position="last",
        batch_size=1,
        max_contexts=3,
        sae_model=sae,
        preselected_context_indices={0: (1, 2, 0)},
    )(results)
    FEGAComputeEffect(backend, batch_size=1, sae_model=sae)(results)

    assert results[keys.DATA_PREP].selected_context_indices[0] == (1, 2, 0)
    assert results[keys.EFFECTS].features[0].context_indices == (1, 2, 0)


def test_explicit_preselected_contexts_require_active_rows() -> None:
    """Explicit FEGA row picks must still satisfy the activation threshold."""
    # Reject a manually selected row whose activation is below the active cutoff.
    backend = cast(ModelBackend, TinyBackend())
    sae = cast(SAEModel, FakeSAE())
    results = Results()
    results[murano_keys.PROMPTS] = PromptBatch(["a", "bb", "cccc"], source="unit")

    with pytest.raises(ValueError, match="inactive preselected context"):
        FEGADataPrep(
            backend,
            FakeSAE.release,
            FakeSAE.sae_id,
            [0],
            position="last",
            batch_size=1,
            activation_threshold=4.0,
            max_contexts=2,
            sae_model=sae,
            preselected_context_indices={0: (0, 2)},
        )(results)


def test_seeded_synthetic_vmf_is_worker_invariant() -> None:
    """Seeded dense-vMF selection must not depend on candidate worker count."""
    # Fit the same separated directional cloud serially and in parallel.
    angles = np.r_[np.linspace(-0.1, 0.1, 12), np.linspace(1.4, 1.7, 12)]
    cloud = np.column_stack((np.cos(angles), np.sin(angles)))
    serial = select_vmf(cloud, (3, 1, 2, 1), seed=17, n_init=2, n_jobs=1)
    parallel = select_vmf(cloud, (3, 1, 2, 1), seed=17, n_init=2, n_jobs=4)

    # Selection, partition, and objective are the decision-changing outputs.
    assert feature_seed(42, 33760) == 3_535_651_082
    assert [candidate.n_components for candidate in serial.candidates] == [1, 2, 3]
    assert serial.selected.n_components == parallel.selected.n_components
    np.testing.assert_array_equal(serial.selected.labels, parallel.selected.labels)
    assert serial.selected.log_likelihood == parallel.selected.log_likelihood


def test_normalize_delta_rows_raises_on_persistent_negative_quadratic_form() -> None:
    """Persistent negative Gram quadratic forms must fail closed, not clamp to zero."""
    # Use an indefinite metric so the float64 retry remains materially negative.
    with pytest.raises(FloatingPointError, match="negative Gram quadratic form"):
        normalize_delta_rows(
            torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            torch.tensor([[-1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
        )


def test_analysis_uses_retained_source_reporting_metrics() -> None:
    """Reporting must reuse retained float32 source metrics, not recompute Gram rays."""
    # A cancellation-heavy opposite-center cloud separates source float32 and old Gram reductions.
    center = torch.ones(128, dtype=torch.float32)
    center /= torch.linalg.vector_norm(center)
    directions = torch.cat((center.repeat(27, 1), (-center).repeat(37, 1)), dim=0)
    labels = np.asarray([0] * 27 + [1] * 37, dtype=np.int64)
    weights = np.asarray([27 / 64, 37 / 64], dtype=np.float64)
    concentrations = np.asarray([11.0, 13.0], dtype=np.float64)
    source_rows = _historical_metric_rows(directions)
    expected = _historical_reporting_metrics(
        source_rows, labels, weights, concentrations
    )
    old_delta_mix = _old_gram_delta_mix(directions, labels, weights)

    assert abs(expected["delta_mix"] - old_delta_mix) > 1.0e-7
    assert _source_c_ray(source_rows.numpy()) == _historical_c_ray(source_rows)
    for mode in range(2):
        mode_rows = source_rows[torch.from_numpy(labels == mode)]
        assert _source_c_ray(mode_rows.numpy()) == _historical_c_ray(mode_rows)

    geometry = GeometryMetrics(
        n_total=64,
        n_valid=64,
        skipped_nonfinite=0,
        skipped_zero_norm=0,
        c_ray=0.81,
        r2=0.82,
        eigenvalues=np.asarray([1.0], dtype=np.float64),
        s_span={1: 0.81},
        u_span={},
        d_span={},
        b_axis=0.0,
        r_span_ent=1.0,
        r_span_pr=1.0,
        centered_eigenvalues=np.asarray([0.0], dtype=np.float64),
        e_res=0.0,
        s_res={},
        r_ctr_ent=0.0,
        r_ctr_pr=0.0,
    )
    effects = FEGAFeatureEffects(
        feature_id=33760,
        directions=directions,
        magnitudes=torch.ones(64, dtype=torch.float32),
        context_indices=tuple(range(64)),
        feature_activations=torch.ones(64, dtype=torch.float32),
        retained_mask=(True,) * 64,
    )
    selection = VMFSelection(
        selected=VMFSelectedFit(
            n_components=2,
            weights=weights,
            concentrations=concentrations,
            labels=labels,
            log_likelihood=1.0,
            bic=1.0,
            converged=True,
            n_iter=3,
        ),
        candidates=(),
        delta_mix=expected["delta_mix"],
        mode_mass_min=expected["mode_mass_min"],
        min_mode_c_ray=expected["min_mode_c_ray"],
        mode_kappa_min=expected["mode_kappa_min"],
    )
    vmf = FEGAVMFResult(
        features={33760: selection},
        assignment_stability={33760: 0.91},
        fega_config={},
    )

    record = _record(
        geometry,
        33760,
        effects.magnitudes.numpy(),
        effects.retained_mask,
        vmf,
    )

    assert record.selected_mode_count == 2
    assert record.delta_mix == expected["delta_mix"]
    assert record.mode_mass_min == expected["mode_mass_min"]
    assert record.min_mode_c_ray == expected["min_mode_c_ray"]
    assert record.mode_kappa_min == expected["mode_kappa_min"]
    assert record.delta_mix is not None
    assert abs(record.delta_mix - old_delta_mix) > 1.0e-7


def test_reporting_selected_family_fields_hide_non_selected_candidates() -> None:
    """Public selected-k fields must stay empty outside the chosen label family."""
    # Keep candidate-family evidence visible while enforcing source-selected fields.
    record = GeometryRecord(
        n_valid=8,
        zero_filter_frac=0.0,
        c_ray=0.9,
        s_span={1: 0.9, 2: 0.9},
        u_span={2: 0.08},
        d_span={2: 0.6},
        r_span_pr=1.6,
        m_cv=0.0,
        e_res=0.1,
        s_res={2: 0.8},
        r_ctr_pr=1.5,
    )
    classification = classify_geometry(record)

    assert classification.primary_label == "directed_ray"
    assert classification.selected_k is None
    assert classification.span_selected_k is None
    assert classification.residual_selected_k is None
    assert "global_2D_directional_subspace" in classification.candidate_labels
    assert "residual_lowD_k" in classification.candidate_labels


def test_reporting_reused_multimode_stability_stays_accepted() -> None:
    """Reused multimode assignment stability must not be downgraded by low-n protocol."""
    # Match the source state table for strict multimode reporting with reused assignments.
    qualified = qualify_geometry(
        classify_geometry(
            GeometryRecord(
                n_valid=9,
                zero_filter_frac=0.0,
                selected_mode_count=2,
                delta_mix=0.2,
                mode_mass_min=0.2,
                min_mode_c_ray=0.8,
                mode_kappa_min=1.0,
                assignment_stability=0.9,
            )
        ),
        {
            "decision": "stable",
            "assignment_stability": "reused",
            "protocol": {"status": "exploratory", "protocol": "leave_out_sensitivity"},
            "groups": {"status": "group_sampling_unavailable", "groups": {}},
        },
    )

    assert qualified.primary_label == "multi_mode_directional_geometry"
    assert qualified.stability_status == "stable"
    assert qualified.confidence == "accepted"


def test_subset_protocol_counts_all_crossed_required_margins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One replicate may cross several required margins but only one instability."""
    # Force one subset to flip every required gate and one subset to stay on side.
    metrics = iter(
        (
            SimpleNamespace(c_ray=0.81, s_span={1: 0.79}, b_axis=0.14),
            SimpleNamespace(c_ray=0.79, s_span={1: 0.81}, b_axis=0.16),
        )
    )
    monkeypatch.setattr(
        stability_module,
        "compute_geometry_metrics",
        lambda *args, **kwargs: next(metrics),
    )
    protocol = _subset_protocol(
        np.zeros((4, 2), dtype=np.float64),
        np.eye(2, dtype=np.float64),
        (
            np.array([0, 1], dtype=np.int64),
            np.array([2, 3], dtype=np.int64),
        ),
        "axis_or_antipodal",
        None,
        {"c_ray_lt": 0.1, "s_span_1_axis": 0.1, "b_axis": 0.1},
        FEGAConfig(seed=7),
    )

    assert protocol["status"] == "unstable"
    assert protocol["instability_count"] == 1
    assert protocol["gate_crossing_counts"] == {
        "c_ray_lt": 1,
        "s_span_1_axis": 1,
        "b_axis": 1,
    }


def test_subset_protocol_counts_selected_k_mismatch_as_instability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A derived subset-k mismatch must be counted even before projection."""
    # Keep the subset within the same family but force it down to a smaller passing k.
    metrics = iter(
        (
            SimpleNamespace(
                s_span={2: 0.95, 3: 0.89},
                r_span_pr=2.0,
                u_span={2: 0.10, 3: 0.06},
                d_span={2: 0.40, 3: 0.40},
            ),
        )
    )
    monkeypatch.setattr(
        stability_module,
        "compute_geometry_metrics",
        lambda *args, **kwargs: next(metrics),
    )
    protocol = _subset_protocol(
        np.zeros((4, 2), dtype=np.float64),
        np.eye(2, dtype=np.float64),
        (np.array([0, 1, 2], dtype=np.int64),),
        "global_kD_directional_subspace",
        3,
        {
            "s_span_2": 0.1,
            "r_span_pr_k2": 0.1,
            "u_span_2": 0.1,
            "d_span_2": 0.1,
            "s_span_3": 0.1,
            "r_span_pr_k3": 0.1,
            "u_span_3": 0.1,
            "d_span_3": 0.1,
        },
        FEGAConfig(seed=7),
    )

    assert protocol["status"] == "unstable"
    assert protocol["selected_k_mismatch_count"] == 1
    assert protocol["instability_count"] == 1


@pytest.mark.parametrize(
    ("family", "selected_k", "expected_renderer", "expected_kwargs"),
    (
        (
            "multi_mode_directional_geometry",
            None,
            "render_projection_2d",
            {"primary_axis_guide": False, "view_limit": 1.35},
        ),
        (
            "residual_lowD_k",
            2,
            "render_residual_view",
            {},
        ),
    ),
)
def test_visualize_uses_family_specific_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    selected_k: int | None,
    expected_renderer: str,
    expected_kwargs: dict[str, object],
) -> None:
    """Each geometry family must use the intended projection and assignments."""
    # Replace file rendering while keeping projection and capture logic intact.
    for name in (
        "render_sphere_surface",
        "render_projection_2d",
        "render_residual_view",
    ):
        monkeypatch.setattr(
            "notebooks.reproductions.fega.fega_method.analysis." + name,
            lambda path, *args, **kwargs: Path(path).write_text("ok"),
        )

    directions = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    store = FEGAEffectStore(
        features={
            42: FEGAFeatureEffects(
                feature_id=42,
                directions=directions,
                magnitudes=torch.ones(3),
                context_indices=(0, 1, 2),
                feature_activations=torch.ones(3),
                retained_mask=(True, True, True),
            )
        },
        gram=torch.eye(3),
        feature_ids=(42,),
    )
    vmf = (
        FEGAVMFResult(
            features={
                42: cast(
                    VMFSelection,
                    SimpleNamespace(
                        selected=SimpleNamespace(
                            labels=np.array([0, 0, 1], dtype=np.int64),
                            n_components=2,
                            weights=np.array([2.0 / 3.0, 1.0 / 3.0]),
                            concentrations=np.array([4.0, 5.0]),
                        )
                    ),
                ),
            },
            assignment_stability={42: 0.9},
            fega_config={},
        )
        if family == "multi_mode_directional_geometry"
        else FEGAVMFResult(features={}, assignment_stability={42: None}, fega_config={})
    )
    results = Results()
    results[keys.EFFECTS] = store
    results[keys.GEOMETRY] = SimpleNamespace(
        features={
            42: SimpleNamespace(
                n_valid=3,
                r2=0.7,
                c_ray=0.7,
                r_span_pr=1.0,
                b_axis=0.0,
                e_res=0.1,
                r_ctr_pr=1.1,
                u_span={2: 0.1},
                d_span={2: 0.3},
                s_span={1: 0.7, 2: 0.9, 3: 1.0, 4: None, 8: None},
            )
        }
    )
    results[keys.VMF] = vmf
    results[keys.REPORTING] = SimpleNamespace(
        features={
            42: SimpleNamespace(primary_label=family, selected_k=selected_k),
        },
        feature_ids=(42,),
    )

    FEGAVisualize(tmp_path, figures=("sphere_surface", "projection_2d"), dpi=72)(
        results
    )
    captures = results[keys.VISUALIZATION].captures[42]
    sphere_capture = captures["sphere_surface"]
    projection_capture = captures["projection_2d"]

    assert sphere_capture.coordinates.shape == (3, 3)
    assert projection_capture.renderer == expected_renderer
    assert projection_capture.selected_k == selected_k
    for key, value in expected_kwargs.items():
        assert projection_capture.kwargs[key] == value
    if family == "multi_mode_directional_geometry":
        assert projection_capture.assignments == (0, 0, 1)
        assert projection_capture.kwargs["mode_assignments"] == (0, 0, 1)
        assert projection_capture.point_colors is not None
