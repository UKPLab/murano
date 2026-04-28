"""Tests for per-module targeting (Issue #53).

Verifies that Record, Intervene, SteeringVector, and Probe correctly
handle the ``modules`` parameter — both single-module (``"residual"``)
and multi-module (e.g. ``["residual", "mlp"]``) configurations.
"""

from __future__ import annotations

import pytest
import torch

from murano.results import Results
from murano.steps.record import ActivationStore, LabeledActivationStore, Record
from murano.steps.train import SteeringVector, SteeringResult
from murano.steps.probe import Probe, ProbeResult
from murano.steps.intervene import (
    ablate_direction,
    steer_direction,
    ActivationKey,
)
from murano.dataset import MuranoDataset, LabeledDataset


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def d_model():
    return 64


@pytest.fixture
def n_layers():
    return 4


@pytest.fixture
def dummy_model(n_layers, d_model):
    """A minimal model stub that exposes the interface Record needs."""

    class DummyLayer:
        def __init__(self, idx):
            self.idx = idx
            self.output = None

    class DummyLM:
        def __init__(self):
            self.tokenizer = None

        def trace(self, tokens):
            class NullCtx:
                def __enter__(self_):
                    return self_

                def __exit__(self_, *args):
                    pass

            return NullCtx()

        def generate(self, tokens, **kwargs):
            class NullCtx:
                def __enter__(self_):
                    return self_

                def __exit__(self_, *args):
                    pass

            return NullCtx()

    class DummyModel:
        def __init__(self):
            self.n_layers = n_layers
            self.d_model = d_model
            self._lm = DummyLM()
            self.tokenizer = None

        def layer(self, idx):
            return DummyLayer(idx)

        @staticmethod
        def _resolve_module(layer_proxy, mod_str):
            if mod_str == "residual":
                return layer_proxy
            # For testing, just return the layer proxy for any submodule
            return layer_proxy

        def _layer_indices(self, layers):
            if isinstance(layers, str) and layers == "all":
                return list(range(self.n_layers))
            return list(layers)

    return DummyModel()


@pytest.fixture
def single_module_store(n_layers, d_model):
    """ActivationStore with int keys (single module)."""
    return ActivationStore(
        positive={layer: torch.randn(8, d_model) + 0.5 for layer in range(n_layers)},
        negative={layer: torch.randn(8, d_model) - 0.5 for layer in range(n_layers)},
    )


@pytest.fixture
def multi_module_store(n_layers, d_model):
    """ActivationStore with tuple[int, str] keys (multiple modules)."""
    modules = ["residual", "mlp"]
    pos = {}
    neg = {}
    for layer in range(n_layers):
        for mod in modules:
            key = (layer, mod)
            pos[key] = torch.randn(8, d_model) + 0.5
            neg[key] = torch.randn(8, d_model) - 0.5
    return ActivationStore(positive=pos, negative=neg)


@pytest.fixture
def multi_module_labeled_store(n_layers, d_model):
    """LabeledActivationStore with tuple[int, str] keys."""
    modules = ["residual", "attn"]
    acts = {}
    for layer in range(n_layers):
        for mod in modules:
            key = (layer, mod)
            acts[key] = torch.cat(
                [
                    torch.randn(10, d_model) + 2.0,
                    torch.randn(10, d_model) - 2.0,
                ]
            )
    labels = torch.tensor([0] * 10 + [1] * 10)
    return LabeledActivationStore(activations=acts, labels=labels)


# ── Record Tests ──────────────────────────────────────────────────────


class TestRecordModules:
    """Record step with modules parameter."""

    def test_default_modules_is_residual(self, dummy_model):
        """Default modules='residual' should produce int keys."""
        step = Record(dummy_model, layers=[0, 1])
        assert step.modules == ["residual"]

    def test_single_module_produces_int_keys(self, dummy_model):
        """A single module string should produce dict[int, Tensor]."""
        step = Record(dummy_model, layers=[0, 1], modules="residual")
        assert step.modules == ["residual"]

    def test_multi_module_produces_tuple_keys(self, dummy_model):
        """A list of modules should produce dict[tuple[int, str], Tensor]."""
        step = Record(dummy_model, layers=[0, 1], modules=["residual", "mlp"])
        assert step.modules == ["residual", "mlp"]

    def test_activation_store_type_hints_accept_tuple_keys(self):
        """ActivationStore should accept tuple[int, str] keys."""
        store = ActivationStore(
            positive={(0, "mlp"): torch.randn(4, 64)},
            negative={(0, "mlp"): torch.randn(4, 64)},
        )
        assert isinstance(list(store.positive.keys())[0], tuple)

    def test_labeled_store_type_hints_accept_tuple_keys(self):
        """LabeledActivationStore should accept tuple[int, str] keys."""
        store = LabeledActivationStore(
            activations={(0, "attn"): torch.randn(4, 64)},
            labels=torch.tensor([0, 0, 1, 1]),
        )
        assert isinstance(list(store.activations.keys())[0], tuple)


# ── SteeringVector Tests ──────────────────────────────────────────────


class TestSteeringVectorModules:
    """SteeringVector with multi-module keys."""

    def test_single_module_keys_unchanged(self, single_module_store):
        """With int keys, SteeringVector should produce int-keyed results."""
        r = Results()
        r["record"] = single_module_store
        results = SteeringVector()(r)
        steering = results["steering"]
        keys = list(steering.direction_per_layer.keys())
        assert all(isinstance(k, int) for k in keys)

    def test_multi_module_keys_preserved(self, multi_module_store):
        """With tuple keys, SteeringVector should preserve tuple keys."""
        r = Results()
        r["record"] = multi_module_store
        results = SteeringVector()(r)
        steering = results["steering"]
        keys = list(steering.direction_per_layer.keys())
        assert all(isinstance(k, tuple) for k in keys)
        assert all(len(k) == 2 for k in keys)

    def test_multi_module_direction_shapes(self, multi_module_store, d_model):
        """Directions should have correct shape regardless of key type."""
        r = Results()
        r["record"] = multi_module_store
        results = SteeringVector()(r)
        steering = results["steering"]
        for key, direction in steering.direction_per_layer.items():
            assert direction.shape == (d_model,), f"Key {key}: shape mismatch"

    def test_multi_module_directions_normalized(self, multi_module_store):
        """Directions should be normalized with tuple keys."""
        r = Results()
        r["record"] = multi_module_store
        results = SteeringVector(normalize=True)(r)
        for key, direction in results["steering"].direction_per_layer.items():
            norm = direction.norm().item()
            assert abs(norm - 1.0) < 1e-5, f"Key {key}: norm={norm}"

    def test_multi_module_separation_scores(self, multi_module_store, n_layers):
        """Separation scores should exist for all keys."""
        r = Results()
        r["record"] = multi_module_store
        results = SteeringVector()(r)
        scores = results["steering"].separation_scores
        expected_keys = {(layer, mod) for layer in range(n_layers) for mod in ["residual", "mlp"]}
        assert set(scores.keys()) == expected_keys

    def test_multi_module_best_layer_is_tuple(self, multi_module_store):
        """best_layer should be a tuple when keys are tuples."""
        r = Results()
        r["record"] = multi_module_store
        results = SteeringVector()(r)
        assert isinstance(results["steering"].best_layer, tuple)

    def test_single_module_best_layer_is_int(self, single_module_store):
        """best_layer should be an int when keys are ints."""
        r = Results()
        r["record"] = single_module_store
        results = SteeringVector()(r)
        assert isinstance(results["steering"].best_layer, int)

    def test_steering_result_type(self, multi_module_store):
        """Output type should be SteeringResult regardless of key type."""
        r = Results()
        r["record"] = multi_module_store
        results = SteeringVector()(r)
        assert isinstance(results["steering"], SteeringResult)


# ── Probe Tests ───────────────────────────────────────────────────────


class TestProbeModules:
    """Probe step with multi-module keys."""

    def test_multi_module_accuracy_keys(self, multi_module_labeled_store, n_layers):
        """Accuracy dict should have tuple keys."""
        ds = LabeledDataset(texts=["a"] * 20, labels=[0] * 10 + [1] * 10)
        r = Results()
        r["dataset"] = ds
        r["record"] = multi_module_labeled_store
        results = Probe(cv=2)(r)
        keys = list(results["probe"].accuracy_per_layer.keys())
        assert all(isinstance(k, tuple) for k in keys)

    def test_multi_module_best_layer_is_tuple(self, multi_module_labeled_store):
        """best_layer should be a tuple with multi-module keys."""
        ds = LabeledDataset(texts=["a"] * 20, labels=[0] * 10 + [1] * 10)
        r = Results()
        r["dataset"] = ds
        r["record"] = multi_module_labeled_store
        results = Probe(cv=2)(r)
        assert isinstance(results["probe"].best_layer, tuple)

    def test_multi_module_high_accuracy(self, multi_module_labeled_store):
        """Well-separated data should yield high accuracy with tuple keys."""
        ds = LabeledDataset(texts=["a"] * 20, labels=[0] * 10 + [1] * 10)
        r = Results()
        r["dataset"] = ds
        r["record"] = multi_module_labeled_store
        results = Probe(cv=2)(r)
        best_acc = max(results["probe"].accuracy_per_layer.values())
        assert best_acc > 0.7

    def test_multi_module_refit_classifiers(self, multi_module_labeled_store, n_layers):
        """Refit should store classifiers under tuple keys."""
        ds = LabeledDataset(texts=["a"] * 20, labels=[0] * 10 + [1] * 10)
        r = Results()
        r["dataset"] = ds
        r["record"] = multi_module_labeled_store
        results = Probe(cv=2, refit=True)(r)
        keys = list(results["probe"].classifiers.keys())
        assert all(isinstance(k, tuple) for k in keys)
        assert len(keys) == n_layers * 2  # 2 modules per layer

    def test_probe_result_type(self, multi_module_labeled_store):
        """Output type should be ProbeResult regardless of key type."""
        ds = LabeledDataset(texts=["a"] * 20, labels=[0] * 10 + [1] * 10)
        r = Results()
        r["dataset"] = ds
        r["record"] = multi_module_labeled_store
        results = Probe(cv=2)(r)
        assert isinstance(results["probe"], ProbeResult)


# ── Intervention Function Tests ───────────────────────────────────────


class TestInterventionFunctionsModules:
    """Intervention functions with ActivationKey."""

    def test_ablate_with_int_keys(self, d_model):
        """ablate_direction should work with int keys."""
        directions = {0: torch.randn(d_model)}
        fn = ablate_direction(directions)
        activation = torch.randn(1, 1, d_model)
        result = fn(activation, 0)
        assert result.shape == activation.shape

    def test_ablate_with_tuple_keys(self, d_model):
        """ablate_direction should work with tuple keys."""
        directions = {(0, "mlp"): torch.randn(d_model)}
        fn = ablate_direction(directions)
        activation = torch.randn(1, 1, d_model)
        result = fn(activation, (0, "mlp"))
        assert result.shape == activation.shape

    def test_steer_with_int_keys(self, d_model):
        """steer_direction should work with int keys."""
        directions = {0: torch.randn(d_model)}
        fn = steer_direction(directions, alpha=1.0)
        activation = torch.randn(1, 1, d_model)
        result = fn(activation, 0)
        assert result.shape == activation.shape

    def test_steer_with_tuple_keys(self, d_model):
        """steer_direction should work with tuple keys."""
        directions = {(0, "mlp"): torch.randn(d_model)}
        fn = steer_direction(directions, alpha=1.0)
        activation = torch.randn(1, 1, d_model)
        result = fn(activation, (0, "mlp"))
        assert result.shape == activation.shape

    def test_ablate_absent_key_is_identity(self, d_model):
        """Missing key should return activation unchanged."""
        directions = {0: torch.randn(d_model)}
        fn = ablate_direction(directions)
        activation = torch.randn(1, 1, d_model)
        result = fn(activation, (0, "mlp"))  # tuple key not in directions
        assert torch.equal(result, activation)

    def test_steer_absent_key_is_identity(self, d_model):
        """Missing key should return activation unchanged."""
        directions = {(0, "mlp"): torch.randn(d_model)}
        fn = steer_direction(directions, alpha=1.0)
        activation = torch.randn(1, 1, d_model)
        result = fn(activation, 0)  # int key not in directions
        assert torch.equal(result, activation)

    def test_ablate_removes_component_tuple_key(self, d_model):
        """Ablation should remove direction component with tuple keys."""
        direction = torch.randn(d_model)
        direction = direction / direction.norm()
        fn = ablate_direction({(0, "mlp"): direction})
        activation = direction.unsqueeze(0).unsqueeze(0) * 5.0
        result = fn(activation, (0, "mlp"))
        component = (result @ direction).item()
        assert abs(component) < 1e-4

    def test_steer_adds_component_tuple_key(self, d_model):
        """Steering should add direction component with tuple keys."""
        direction = torch.randn(d_model)
        direction = direction / direction.norm()
        alpha = 2.0
        fn = steer_direction({(0, "mlp"): direction}, alpha=alpha)
        activation = torch.zeros(1, 1, d_model)
        result = fn(activation, (0, "mlp"))
        expected = alpha * direction
        diff = (result.squeeze() - expected).norm().item()
        assert diff < 1e-4


# ── ActivationKey Type Tests ──────────────────────────────────────────


class TestActivationKeyType:
    """Verify ActivationKey behaves correctly as a dict key."""

    def test_int_and_tuple_keys_can_coexist(self):
        """Dict can hold both int and tuple keys."""
        d: dict[ActivationKey, str] = {
            0: "layer_0",
            (0, "mlp"): "layer_0_mlp",
        }
        assert d[0] == "layer_0"
        assert d[(0, "mlp")] == "layer_0_mlp"

    def test_tuple_key_equality(self):
        """Tuple keys should compare correctly."""
        assert (0, "mlp") == (0, "mlp")
        assert (0, "mlp") != (1, "mlp")
        assert (0, "mlp") != (0, "attn")

    def test_tuple_key_hashable(self):
        """Tuple keys should be usable in sets."""
        s = {(0, "mlp"), (1, "attn"), (0, "mlp")}
        assert len(s) == 2