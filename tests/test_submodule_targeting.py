"""Tests for per-module targeting (Issue #53).

Verifies that Record, Intervene, SteeringVector, and Probe correctly
handle the ``modules`` parameter: both single-module (``"residual"``)
and multi-module (e.g. ``["residual", "mlp"]``) configurations.
"""

from __future__ import annotations

import pytest
import torch

from murano.nodes import MLP, Node
from murano.results import Results
from murano.steps.record import (
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
def multi_module_store(n_layers, d_model):
    """ActivationStore built from tuple shorthand, stored as Node keys."""
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
    """LabeledActivationStore built from tuple shorthand, stored as Node keys."""
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

    def test_module_string_is_normalized_to_list(self, dummy_model):
        """A module string (default or explicit) becomes a one-element list."""
        assert Record(dummy_model, layers=[0, 1]).modules == ["residual"]
        assert Record(dummy_model, layers=[0, 1], modules="mlp").modules == ["mlp"]

    def test_multi_module_produces_node_keys(self, dummy_model):
        """A list of modules should record under a Node key per (layer, module)."""
        step = Record(dummy_model, layers=[0, 1], modules=["residual", "mlp"])
        assert step.modules == ["residual", "mlp"]

    def test_activation_store_coerces_tuple_keys_to_node(self):
        """ActivationStore accepts tuple shorthand and stores canonical Nodes."""
        store = ActivationStore(
            positive={(0, "mlp"): torch.randn(4, 64)},
            negative={(0, "mlp"): torch.randn(4, 64)},
        )
        assert isinstance(list(store.positive.keys())[0], Node)
        assert list(store.positive.keys())[0] == Node(0, MLP)

    def test_labeled_store_coerces_tuple_keys_to_node(self):
        """LabeledActivationStore accepts tuple shorthand and stores Nodes."""
        store = LabeledActivationStore(
            activations={(0, "self_attn"): torch.randn(4, 64)},
            labels=torch.tensor([0, 0, 1, 1]),
        )
        assert isinstance(list(store.activations.keys())[0], Node)


# ── SteeringVector Tests ──────────────────────────────────────────────


class TestSteeringVectorModules:
    """SteeringVector with multi-module keys."""

    def test_rejects_full_position_store(self, d_model):
        """SteeringVector rejects a full-position store instead of producing a
        wrong-shaped direction."""
        r = Results()
        r["record"] = ActivationStore(
            positive={(0, "residual"): torch.randn(2, 3, d_model)},
            negative={(0, "residual"): torch.randn(2, 3, d_model)},
            position="none",
            positive_token_mask=torch.ones(2, 3),
            negative_token_mask=torch.ones(2, 3),
        )
        with pytest.raises(ValueError, match="reduced"):
            SteeringVector()(r)

    def test_rejects_per_head_store(self, d_model):
        """SteeringVector rejects a per-head store."""
        r = Results()
        r["record"] = ActivationStore(
            positive={(0, "self_attn"): torch.randn(2, 4, d_model // 4)},
            negative={(0, "self_attn"): torch.randn(2, 4, d_model // 4)},
            per_head=True,
        )
        with pytest.raises(ValueError, match="reduced"):
            SteeringVector()(r)

    def test_multi_module_keys_preserved(self, multi_module_store):
        """SteeringVector keys its output by canonical Node addresses."""
        r = Results()
        r["record"] = multi_module_store
        results = SteeringVector()(r)
        steering = results["steering"]
        keys = list(steering.direction_per_layer.keys())
        assert all(isinstance(k, Node) for k in keys)

    def test_multi_module_direction_shapes(self, multi_module_store, d_model):
        """Directions should have correct shape regardless of key type."""
        r = Results()
        r["record"] = multi_module_store
        results = SteeringVector()(r)
        steering = results["steering"]
        for key, direction in steering.direction_per_layer.items():
            assert direction.shape == (d_model,), f"Key {key}: shape mismatch"

    def test_multi_module_directions_normalized(self, multi_module_store):
        """Directions should be normalized (keys given as tuple shorthand)."""
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
            Node(layer, mod) for layer in range(n_layers) for mod in ["residual", "mlp"]
        }
        assert set(scores.keys()) == expected_keys

    def test_multi_module_best_layer_is_node(self, multi_module_store):
        """best_layer should be a Node."""
        r = Results()
        r["record"] = multi_module_store
        results = SteeringVector()(r)
        assert isinstance(results["steering"].best_layer, Node)

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
        """Accuracy dict should be keyed by Node."""
        ds = LabeledDataset(texts=["a"] * 20, labels=[0] * 10 + [1] * 10)
        r = Results()
        r["dataset"] = ds
        r["record"] = multi_module_labeled_store
        results = Probe(cv=2)(r)
        keys = list(results["probe"].accuracy_per_layer.keys())
        assert all(isinstance(k, Node) for k in keys)

    def test_multi_module_best_layer_is_node(self, multi_module_labeled_store):
        """best_layer should be a Node with multi-module keys."""
        ds = LabeledDataset(texts=["a"] * 20, labels=[0] * 10 + [1] * 10)
        r = Results()
        r["dataset"] = ds
        r["record"] = multi_module_labeled_store
        results = Probe(cv=2)(r)
        assert isinstance(results["probe"].best_layer, Node)

    def test_multi_module_high_accuracy(self, multi_module_labeled_store):
        """Well-separated data should yield high accuracy (tuple-shorthand keys)."""
        ds = LabeledDataset(texts=["a"] * 20, labels=[0] * 10 + [1] * 10)
        r = Results()
        r["dataset"] = ds
        r["record"] = multi_module_labeled_store
        results = Probe(cv=2)(r)
        best_acc = max(results["probe"].accuracy_per_layer.values())
        assert best_acc > 0.7

    def test_multi_module_refit_classifiers(self, multi_module_labeled_store, n_layers):
        """Refit should store classifiers under Node keys."""
        ds = LabeledDataset(texts=["a"] * 20, labels=[0] * 10 + [1] * 10)
        r = Results()
        r["dataset"] = ds
        r["record"] = multi_module_labeled_store
        results = Probe(cv=2, refit=True)(r)
        keys = list(results["probe"].classifiers.keys())
        assert all(isinstance(k, Node) for k in keys)
        assert len(keys) == n_layers * 2  # 2 modules per layer

    def test_probe_result_type(self, multi_module_labeled_store):
        """Output type should be ProbeResult regardless of key type."""
        ds = LabeledDataset(texts=["a"] * 20, labels=[0] * 10 + [1] * 10)
        r = Results()
        r["dataset"] = ds
        r["record"] = multi_module_labeled_store
        results = Probe(cv=2)(r)
        assert isinstance(results["probe"], ProbeResult)

    def test_rejects_full_position_store(self, d_model):
        """Probe rejects a full-position store rather than passing rank-3 arrays
        to sklearn."""
        r = Results()
        r["dataset"] = LabeledDataset(texts=["a"] * 4, labels=[0, 0, 1, 1])
        r["record"] = LabeledActivationStore(
            activations={(0, "residual"): torch.randn(4, 3, d_model)},
            labels=torch.tensor([0, 0, 1, 1]),
            position="none",
            token_mask=torch.ones(4, 3),
        )
        with pytest.raises(ValueError, match="reduced"):
            Probe(cv=2)(r)

    def test_rejects_per_head_store(self, d_model):
        """Probe rejects a per-head store."""
        r = Results()
        r["dataset"] = LabeledDataset(texts=["a"] * 4, labels=[0, 0, 1, 1])
        r["record"] = LabeledActivationStore(
            activations={(0, "self_attn"): torch.randn(4, 4, d_model // 4)},
            labels=torch.tensor([0, 0, 1, 1]),
            per_head=True,
        )
        with pytest.raises(ValueError, match="reduced"):
            Probe(cv=2)(r)


# ── Intervention Function Tests ───────────────────────────────────────


class TestInterventionFunctionsModules:
    """Intervention functions with Node-coerced direction keys."""

    def test_ablate_with_tuple_keys(self, d_model):
        """ablate_direction should accept tuple-shorthand keys."""
        directions = {(0, "mlp"): torch.randn(d_model)}
        fn = ablate_direction(directions)
        activation = torch.randn(1, 1, d_model)
        result = fn(activation, (0, "mlp"))
        assert result.shape == activation.shape

    def test_steer_with_tuple_keys(self, d_model):
        """steer_direction should accept tuple-shorthand keys."""
        directions = {(0, "mlp"): torch.randn(d_model)}
        fn = steer_direction(directions, alpha=1.0)
        activation = torch.randn(1, 1, d_model)
        result = fn(activation, (0, "mlp"))
        assert result.shape == activation.shape

    def test_ablate_absent_key_is_identity(self, d_model):
        """Missing key should return activation unchanged."""
        directions = {(0, "residual"): torch.randn(d_model)}
        fn = ablate_direction(directions)
        activation = torch.randn(1, 1, d_model)
        result = fn(activation, (0, "mlp"))  # key not in directions
        assert torch.equal(result, activation)

    def test_steer_absent_key_is_identity(self, d_model):
        """Missing key should return activation unchanged."""
        directions = {(0, "mlp"): torch.randn(d_model)}
        fn = steer_direction(directions, alpha=1.0)
        activation = torch.randn(1, 1, d_model)
        result = fn(activation, 0)  # int key not in directions
        assert torch.equal(result, activation)

    def test_ablate_removes_component_tuple_key(self, d_model):
        """Ablation should remove the direction component (tuple-shorthand keys)."""
        direction = torch.randn(d_model)
        direction = direction / direction.norm()
        fn = ablate_direction({(0, "mlp"): direction})
        activation = direction.unsqueeze(0).unsqueeze(0) * 5.0
        result = fn(activation, (0, "mlp"))
        component = (result @ direction).item()
        assert abs(component) < 1e-4

    def test_steer_adds_component_tuple_key(self, d_model):
        """Steering should add the direction component (tuple-shorthand keys)."""
        direction = torch.randn(d_model)
        direction = direction / direction.norm()
        alpha = 2.0
        fn = steer_direction({(0, "mlp"): direction}, alpha=alpha)
        activation = torch.zeros(1, 1, d_model)
        result = fn(activation, (0, "mlp"))
        expected = alpha * direction
        diff = (result.squeeze() - expected).norm().item()
        assert diff < 1e-4


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
        """Record with modules='mlp' produces ``(layer, module)`` keys."""
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
        assert (0, "mlp") in store.positive
        assert store.positive[(0, "mlp")].shape == (2, model.d_model)
        assert store.position == "mean"

    def test_record_multi_module(self, tmp_path):
        """Record with modules=['residual', 'mlp'] produces Node keys."""
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

    def test_record_full_position(self, tmp_path):
        """position='none' keeps every token: [N, seq, d_model] + a mask."""
        self._build_tiny_model(tmp_path)
        from murano.model import MuranoModel

        model = MuranoModel(str(tmp_path), device_map="cpu", dtype=torch.float32)
        store = model.record(
            ["hello world", "good world"],
            layers=[0],
            modules="residual",
            position="none",
            batch_size=2,
        )
        acts = store.positive[(0, "residual")]
        assert acts.ndim == 3
        assert acts.shape[0] == 2
        assert acts.shape[2] == model.d_model
        assert store.position == "none"
        assert store.positive_token_mask is not None
        assert store.positive_token_mask.shape == (2, acts.shape[1])

    def test_record_per_head(self, tmp_path):
        """per_head splits attention output into [N, n_heads, head_dim]."""
        self._build_tiny_model(tmp_path)
        from murano.model import MuranoModel

        model = MuranoModel(str(tmp_path), device_map="cpu", dtype=torch.float32)
        n_heads = model._lm.config.num_attention_heads
        store = model.record(
            ["hello world", "good world"],
            layers=[0],
            modules="self_attn",
            position="last",
            per_head=True,
            batch_size=2,
        )
        acts = store.positive[(0, "self_attn")]
        assert store.per_head is True
        assert acts.shape == (2, n_heads, model.d_model // n_heads)

    def test_record_per_head_full_position(self, tmp_path):
        """per_head + position='none' gives [N, seq, n_heads, head_dim]."""
        self._build_tiny_model(tmp_path)
        from murano.model import MuranoModel

        model = MuranoModel(str(tmp_path), device_map="cpu", dtype=torch.float32)
        n_heads = model._lm.config.num_attention_heads
        store = model.record(
            ["hello world", "good world"],
            layers=[0],
            modules="self_attn",
            position="none",
            per_head=True,
            batch_size=2,
        )
        acts = store.positive[(0, "self_attn")]
        assert acts.ndim == 4
        assert acts.shape[0] == 2
        assert acts.shape[2] == n_heads
        assert acts.shape[3] == model.d_model // n_heads

    def test_per_head_non_attention_raises(self, tmp_path):
        """per_head on a non-attention module raises NotImplementedError."""
        self._build_tiny_model(tmp_path)
        from murano.model import MuranoModel

        model = MuranoModel(str(tmp_path), device_map="cpu", dtype=torch.float32)
        with pytest.raises(NotImplementedError):
            model.record(
                ["hello world"],
                layers=[0],
                modules="mlp",
                position="last",
                per_head=True,
                batch_size=1,
            )

    def test_steering_vector_with_multi_module(self, tmp_path):
        """SteeringVector keys multi-module output by Node."""
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
        assert all(isinstance(k, Node) for k in keys)
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
        # Don't assert non-empty: with max_new_tokens=1 and a tiny random-init
        # model whose vocab is half special tokens, the generated token is
        # often <pad>/<s>/</s>/<unk>, which decodes to "" via skip_special_tokens.
        # The test's stated purpose ("runs without error") is satisfied by the
        # call returning a string.
        assert isinstance(result, str)

    def test_generate_accepts_int_keyed_directions(self, tmp_path):
        """A bare-int direction key is accepted as that layer's residual stream.

        Node coercion maps int ``0`` to ``Node(0, resid_post)``, the exact site
        the default ``modules="residual"`` hook edits, so the ablation applies
        and generation runs instead of silently no-opping.
        """
        self._build_tiny_model(tmp_path)
        from murano.model import MuranoModel

        model = MuranoModel(str(tmp_path), device_map="cpu", dtype=torch.float32)
        result = model.generate(
            "hello",
            ablate={0: torch.randn(model.d_model)},
            layers=[0],
            gen_kwargs={"max_new_tokens": 1, "do_sample": False},
        )
        assert isinstance(result, str)

    def test_generate_raises_on_unreachable_directions(self, tmp_path):
        """Directions that match no hooked site fail loudly, not silently.

        A bare-int key targets ``resid_post``; hooking only ``mlp`` would make
        the intervention a no-op, so it must error instead.
        """
        self._build_tiny_model(tmp_path)
        from murano.model import MuranoModel

        model = MuranoModel(str(tmp_path), device_map="cpu", dtype=torch.float32)
        with pytest.raises(ValueError, match="would do nothing"):
            model.generate(
                "hello",
                ablate={0: torch.randn(model.d_model)},
                modules="mlp",
                layers=[0],
                gen_kwargs={"max_new_tokens": 1, "do_sample": False},
            )

    def test_canonical_module_round_trips_through_resolver(self, tmp_path):
        """A stored Node's module feeds back into record/resolve_module.

        Node canonicalizes "residual" -> "resid_post"; the resolver must accept
        the canonical name so results compose (record a site, reuse its module).
        """
        self._build_tiny_model(tmp_path)
        from murano.model import MuranoModel

        model = MuranoModel(str(tmp_path), device_map="cpu", dtype=torch.float32)
        store = model.record("hello", layers=[0], position="last")
        node = next(iter(store.positive))
        assert node.module == "resid_post"

        # Feeding the stored canonical module back must resolve to a live hook.
        store2 = model.record(
            "hello", layers=[node.layer], modules=node.module, position="last"
        )
        assert node in store2.positive

        # resolve_module accepts canonical names and aliases.
        for module in (
            "resid_post",
            "residual",
            "mlp",
            "mlp_out",
            "self_attn",
            "attn_out",
        ):
            model.resolve_module(0, module)
        # resid_pre/resid_mid have no single output hook on this path.
        for module in ("resid_pre", "resid_mid"):
            with pytest.raises(ValueError, match="no single output hook"):
                model.resolve_module(0, module)
