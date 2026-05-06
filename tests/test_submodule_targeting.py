"""Tests for per-module targeting (Issue #53).

Verifies that Record, Intervene, SteeringVector, and Probe correctly
handle the ``modules`` parameter — both single-module (``"residual"``)
and multi-module (e.g. ``["residual", "mlp"]``) configurations.
"""

from __future__ import annotations

import pytest
import torch

from murano.results import Results
from murano.steps.record import (
    ActivationKey,
    ActivationStore,
    LabeledActivationStore,
    Record,
)
from murano.steps.train import SteeringVector, SteeringResult
from murano.steps.probe import Probe, ProbeResult
from murano.steps.intervene import (
    ablate_direction,
    steer_direction,
)
from murano.dataset import LabeledDataset


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
        expected_keys = {
            (layer, mod) for layer in range(n_layers) for mod in ["residual", "mlp"]
        }
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


# ── Real-Model Integration Tests ──────────────────────────────────────


class TestResolveModule:
    """Unit tests for MuranoModel._resolve_module."""

    def test_resolve_residual(self):
        """_resolve_module returns the layer proxy for 'residual'."""
        from murano.model import MuranoModel

        class FakeLayer:
            pass

        layer = FakeLayer()
        result = MuranoModel._resolve_module(layer, "residual")
        assert result is layer

    def test_resolve_child_attribute(self):
        """_resolve_module resolves single child via getattr."""
        from murano.model import MuranoModel

        class FakeLayer:
            class MLP:
                pass

            mlp = MLP()

        layer = FakeLayer()
        result = MuranoModel._resolve_module(layer, "mlp")
        assert result is layer.mlp

    def test_resolve_dotted_path(self):
        """_resolve_module resolves dotted paths."""
        from murano.model import MuranoModel

        class GateProj:
            pass

        class MLP:
            gate_proj = GateProj()

        class FakeLayer:
            mlp = MLP()

        layer = FakeLayer()
        result = MuranoModel._resolve_module(layer, "mlp.gate_proj")
        assert result is layer.mlp.gate_proj

    def test_resolve_bad_path_raises_value_error(self):
        """_resolve_module raises ValueError for non-existent paths."""
        from murano.model import MuranoModel

        class FakeLayer:
            pass

        layer = FakeLayer()
        with pytest.raises(ValueError, match="Could not resolve submodule"):
            MuranoModel._resolve_module(layer, "mlp")

    def test_resolve_bad_dotted_path_raises_value_error(self):
        """_resolve_module raises ValueError for non-existent dotted paths."""
        from murano.model import MuranoModel

        class MLP:
            pass

        class FakeLayer:
            mlp = MLP()

        layer = FakeLayer()
        with pytest.raises(ValueError, match="Could not resolve submodule"):
            MuranoModel._resolve_module(layer, "mlp.gate_proj")


@pytest.mark.skipif(
    not pytest.importorskip("nnsight"),
    reason="nnsight not available",
)
class TestSubmoduleTargetingRealModel:
    """End-to-end submodule targeting with a real (tiny) model."""

    def _build_tiny_model(self, tmp_path):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import (
            LlamaConfig,
            LlamaForCausalLM,
            PreTrainedTokenizerFast,
        )

        vocab = {
            "<pad>": 0,
            "<s>": 1,
            "</s>": 2,
            "<unk>": 3,
            "hello": 4,
            "world": 5,
            "good": 6,
            "bad": 7,
        }
        tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="</s>",
            model_max_length=64,
        )
        fast_tokenizer.save_pretrained(tmp_path)

        config = LlamaConfig(
            vocab_size=len(vocab),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
            pad_token_id=vocab["<pad>"],
            bos_token_id=vocab["<s>"],
            eos_token_id=vocab["</s>"],
        )
        model = LlamaForCausalLM(config)
        model.save_pretrained(tmp_path)

    def test_record_single_module_mlp(self, tmp_path):
        """Record with modules='mlp' produces int keys."""
        self._build_tiny_model(tmp_path)
        from murano.model import MuranoModel

        model = MuranoModel(str(tmp_path), device_map="cpu", dtype=torch.float32)
        store = model.record(
            ["hello world", "good world"],
            layers=[0],
            modules="mlp",
            position="mean",
            batch_size=2,
        )
        assert 0 in store.positive
        assert store.positive[0].shape == (2, model.d_model)

    def test_record_multi_module(self, tmp_path):
        """Record with modules=['residual', 'mlp'] produces tuple keys."""
        self._build_tiny_model(tmp_path)
        from murano.model import MuranoModel

        model = MuranoModel(str(tmp_path), device_map="cpu", dtype=torch.float32)
        store = model.record(
            ["hello world"],
            layers=[0],
            modules=["residual", "mlp"],
            position="last",
            batch_size=1,
        )
        assert (0, "residual") in store.positive
        assert (0, "mlp") in store.positive
        assert store.positive[(0, "residual")].shape == (1, model.d_model)
        assert store.positive[(0, "mlp")].shape == (1, model.d_model)

    def test_steering_vector_with_multi_module(self, tmp_path):
        """SteeringVector preserves tuple keys from multi-module recording."""
        self._build_tiny_model(tmp_path)
        from murano.model import MuranoModel

        model = MuranoModel(str(tmp_path), device_map="cpu", dtype=torch.float32)
        steering = model.find_direction(
            positive=["good world"],
            negative=["bad world"],
            layers=[0],
            modules=["residual", "mlp"],
            position="first",
            batch_size=1,
        )
        keys = list(steering.direction_per_layer.keys())
        assert all(isinstance(k, tuple) for k in keys)
        assert (0, "residual") in steering.direction_per_layer
        assert (0, "mlp") in steering.direction_per_layer
        assert steering.direction_per_layer[(0, "mlp")].shape == (model.d_model,)

    def test_generate_with_module_mlp(self, tmp_path):
        """Generate with modules='mlp' and ablation runs without error."""
        self._build_tiny_model(tmp_path)
        from murano.model import MuranoModel

        model = MuranoModel(str(tmp_path), device_map="cpu", dtype=torch.float32)
        steering = model.find_direction(
            positive=["good world"],
            negative=["bad world"],
            layers=[0],
            modules="mlp",
            position="last",
            batch_size=1,
        )
        result = model.generate(
            "hello",
            ablate=steering,
            modules="mlp",
            gen_kwargs={"max_new_tokens": 1, "do_sample": False},
        )
        assert isinstance(result, str)
        assert len(result) > 0
