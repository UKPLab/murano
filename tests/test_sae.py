"""Smoke tests for the SAE steps (SAEEncode, SAETopActivations) and their I/O."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("nnsight")
pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from murano import MuranoModel
from murano.steps import Save
from murano.steps.sae import (
    SAEActivationStore,
    SAEEncode,
    SAEFeatureExamples,
    SAEModel,
    SAETopActivations,
)


VOCAB = {
    "<pad>": 0,
    "<s>": 1,
    "</s>": 2,
    "<unk>": 3,
    "hello": 4,
    "world": 5,
    "good": 6,
    "bad": 7,
}


def _build_tiny_local_model(path: Path) -> None:
    tokenizer = Tokenizer(WordLevel(vocab=VOCAB, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        model_max_length=64,
    )
    fast_tokenizer.save_pretrained(path)

    config = LlamaConfig(
        vocab_size=len(VOCAB),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        pad_token_id=VOCAB["<pad>"],
        bos_token_id=VOCAB["<s>"],
        eos_token_id=VOCAB["</s>"],
    )
    model = LlamaForCausalLM(config)
    model.save_pretrained(path)


@pytest.fixture(scope="module")
def tiny_model_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny_model_sae")
    _build_tiny_local_model(path)
    return path


@pytest.fixture
def model(tiny_model_path):
    return MuranoModel(str(tiny_model_path), device_map="cpu", dtype=torch.float32)


def _synthetic_sae_store(
    *,
    n: int = 3,
    seq: int = 4,
    n_features: int = 5,
    layer: int = 0,
    release: str = "test/synthetic-sae",
    sae_id: str = "test/sae-id",
    texts: list[str] | None = None,
    tokens: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    activations: torch.Tensor | None = None,
) -> SAEActivationStore:
    """Build a fully-formed SAEActivationStore for tests that don't run SAEEncode."""
    if texts is None:
        texts = [f"prompt {i}" for i in range(n)]
    if tokens is None:
        tokens = torch.arange(n * seq, dtype=torch.long).reshape(n, seq) % len(VOCAB)
    if attention_mask is None:
        attention_mask = torch.ones(n, seq, dtype=torch.long)
    if activations is None:
        activations = torch.zeros(n, seq, n_features)
    return SAEActivationStore(
        activations=activations,
        tokens=tokens,
        attention_mask=attention_mask,
        texts=texts,
        hook=layer,
        release=release,
        sae_id=sae_id,
        n_features=n_features,
    )


class _FakeSAE:
    """Drop-in for a sae-lens SAE in tests, no HF download required."""

    def __init__(
        self,
        d_sae: int = 16,
        hook_name: str = "blocks.0.hook_resid_post",
        hook_layer: int = 0,
    ):
        metadata = type("Metadata", (), {})()
        metadata.hook_name = hook_name
        metadata.hook_layer = hook_layer
        cfg = type("Cfg", (), {})()
        cfg.d_sae = d_sae
        cfg.metadata = metadata
        self.cfg = cfg

    def encode(self, residual: torch.Tensor) -> torch.Tensor:
        return residual.new_zeros(*residual.shape[:-1], self.cfg.d_sae)


class TestSAEModel:
    """SAEModel contract; loading is mocked to avoid HF downloads."""

    def test_init_stores_args(self):
        sae = SAEModel(release="acme/sae", sae_id="layer_0/canonical")
        assert sae.release == "acme/sae"
        assert sae.sae_id == "layer_0/canonical"
        assert sae.device == "cpu"
        assert sae._sae is None  # nothing loaded yet

    def test_n_features_delegates_to_loaded_sae(self):
        sae = SAEModel(release="acme/sae", sae_id="layer_0/canonical")
        sae._sae = _FakeSAE()
        assert sae.n_features == 16

    def test_encode_delegates_to_loaded_sae(self):
        sae = SAEModel(release="acme/sae", sae_id="layer_0/canonical")
        sae._sae = _FakeSAE()
        out = sae.encode(torch.zeros(2, 3, 32))
        assert out.shape == (2, 3, 16)

    def test_load_is_cached(self):
        # Once _sae is set, subsequent calls reuse it without re-loading.
        sae = SAEModel(release="acme/sae", sae_id="layer_0/canonical")
        sae._sae = _FakeSAE()
        before = sae._sae
        _ = sae.n_features
        _ = sae.encode(torch.zeros(1, 1, 4))
        assert sae._sae is before


class TestSAEArtifacts:
    """Dataclass construction + field invariants."""

    def test_sae_activation_store_fields(self):
        store = _synthetic_sae_store(n=2, seq=3, n_features=4)
        assert store.activations.shape == (2, 3, 4)
        assert store.tokens.shape == (2, 3)
        assert store.attention_mask.shape == (2, 3)
        assert len(store.texts) == 2
        assert store.hook.layer == 0
        assert store.release == "test/synthetic-sae"
        assert store.sae_id == "test/sae-id"
        assert store.n_features == 4

    def test_sae_feature_examples_construction(self):
        ex = SAEFeatureExamples(
            feat_ids=[0, 1],
            contexts={0: ["a", "b"], 1: ["c"]},
            tokens={0: ["x", "y"], 1: ["z"]},
            act_vals={0: [1.0, 0.5], 1: [0.7]},
            hook=3,
            release="acme/release",
            sae_id="layer_3/canonical",
            k=2,
        )
        assert ex.feat_ids == [0, 1]
        assert ex.contexts[0] == ["a", "b"]
        assert ex.act_vals[1] == [0.7]
        assert ex.hook.layer == 3


class TestSAEEncodeContract:
    """Contract-only tests for SAEEncode (encoder body unimplemented)."""

    def test_init_constructs_sae_model(self, model):
        step = SAEEncode(model, release="test/repo", sae_id="test/id")
        assert step.model is model
        assert isinstance(step.sae_model, SAEModel)
        assert step.sae_model.release == "test/repo"
        assert step.sae_model.sae_id == "test/id"

    def test_declares_correct_reads_writes(self):
        assert SAEEncode.reads == ["prompts"]
        assert SAEEncode.writes == ["sae_record"]
        assert SAEEncode.write_types == {"sae_record": SAEActivationStore}

    def test_call_runs_trace_and_encodes_via_sae_model(self, model):
        # Mock the SAE so this exercises trace + package without HF download.
        from murano.artifacts import PromptBatch
        from murano.results import Results

        results = Results()
        results["prompts"] = PromptBatch(prompts=["hello world", "good world"])
        step = SAEEncode(model, release="test/repo", sae_id="test/id")
        step.sae_model._sae = _FakeSAE(hook_layer=0)

        results = step(results)
        store: SAEActivationStore = results["sae_record"]
        assert store.activations.shape[0] == 2
        assert store.activations.shape[-1] == 16
        assert store.release == "test/repo"
        assert store.sae_id == "test/id"
        assert store.hook.layer == 0
        assert store.n_features == 16
        assert store.texts == ["hello world", "good world"]

    def test_call_validates_layer_bounds_against_sae_cfg(self, model):
        from murano.artifacts import PromptBatch
        from murano.results import Results

        results = Results()
        results["prompts"] = PromptBatch(prompts=["hello"])
        step = SAEEncode(model, release="test/repo", sae_id="test/id")
        step.sae_model._sae = _FakeSAE(hook_layer=model.n_layers)  # out of range
        with pytest.raises(ValueError, match="model has only"):
            step(results)

    def test_call_handles_resid_pre_hook(self, model):
        # SAE trained on `resid_pre` uses layer.input instead of layer.output.
        from murano.artifacts import PromptBatch
        from murano.results import Results

        results = Results()
        results["prompts"] = PromptBatch(prompts=["hello world"])
        step = SAEEncode(model, release="test/repo", sae_id="test/id")
        step.sae_model._sae = _FakeSAE(
            hook_name="blocks.0.hook_resid_pre", hook_layer=0
        )

        results = step(results)
        store: SAEActivationStore = results["sae_record"]
        assert store.hook.layer == 0
        assert store.activations.shape[-1] == 16

    def test_call_handles_mlp_out_hook(self, model):
        from murano.artifacts import PromptBatch
        from murano.results import Results

        results = Results()
        results["prompts"] = PromptBatch(prompts=["hello"])
        step = SAEEncode(model, release="test/repo", sae_id="test/id")
        step.sae_model._sae = _FakeSAE(hook_name="blocks.0.hook_mlp_out", hook_layer=0)

        results = step(results)
        assert results["sae_record"].hook.layer == 0

    def test_call_handles_attn_out_hook(self, model):
        from murano.artifacts import PromptBatch
        from murano.results import Results

        results = Results()
        results["prompts"] = PromptBatch(prompts=["hello"])
        step = SAEEncode(model, release="test/repo", sae_id="test/id")
        step.sae_model._sae = _FakeSAE(hook_name="blocks.0.hook_attn_out", hook_layer=0)

        results = step(results)
        assert results["sae_record"].hook.layer == 0

    def test_call_handles_resid_mid_hook(self, model):
        # resid_mid = resid_pre + attn_out; ensures both captures + add path runs.
        from murano.artifacts import PromptBatch
        from murano.results import Results

        results = Results()
        results["prompts"] = PromptBatch(prompts=["hello"])
        step = SAEEncode(model, release="test/repo", sae_id="test/id")
        step.sae_model._sae = _FakeSAE(
            hook_name="blocks.0.hook_resid_mid", hook_layer=0
        )

        results = step(results)
        assert results["sae_record"].hook.layer == 0

    def test_call_falls_back_when_hook_layer_is_none(self, model):
        # gemma-scope-style: hook_layer is None but sae_id encodes layer_N.
        from murano.artifacts import PromptBatch
        from murano.results import Results

        results = Results()
        results["prompts"] = PromptBatch(prompts=["hello"])
        step = SAEEncode(
            model, release="test/repo", sae_id="layer_1/width_16k/canonical"
        )
        step.sae_model._sae = _FakeSAE(hook_name=None, hook_layer=None)

        results = step(results)
        assert results["sae_record"].hook.layer == 1

    def test_call_rejects_hook_z(self, model):
        # Per-head attention SAEs need release-specific reshape; not supported.
        from murano.artifacts import PromptBatch
        from murano.results import Results

        results = Results()
        results["prompts"] = PromptBatch(prompts=["hello"])
        step = SAEEncode(model, release="test/repo", sae_id="test/id")
        step.sae_model._sae = _FakeSAE(hook_name="blocks.0.attn.hook_z", hook_layer=0)
        with pytest.raises(NotImplementedError, match="hook_z"):
            step(results)

    def test_call_rejects_truly_unknown_hook(self, model):
        from murano.artifacts import PromptBatch
        from murano.results import Results

        results = Results()
        results["prompts"] = PromptBatch(prompts=["hello"])
        step = SAEEncode(model, release="test/repo", sae_id="test/id")
        step.sae_model._sae = _FakeSAE(
            hook_name="blocks.0.something_weird", hook_layer=0
        )
        with pytest.raises(NotImplementedError, match="not recognized"):
            step(results)

    def test_max_length_truncates_when_set(self, model):
        from murano.artifacts import PromptBatch
        from murano.results import Results

        long_prompt = "hello world good bad hello world good bad"  # 8 tokens
        results = Results()
        results["prompts"] = PromptBatch(prompts=[long_prompt])
        step = SAEEncode(model, release="test/repo", sae_id="test/id", max_length=4)
        step.sae_model._sae = _FakeSAE(hook_layer=0)

        results = step(results)
        store = results["sae_record"]
        assert store.tokens.shape[1] == 4


class TestSAETopActivations:
    """Synthetic activations, deterministic top-K behavior."""

    def test_rejects_k_below_one(self, model):
        with pytest.raises(ValueError, match="k must be >= 1"):
            SAETopActivations(model, k=0)

    def test_rejects_out_of_range_feat_ids(self, model):
        from murano.results import Results

        results = Results()
        results["sae_record"] = _synthetic_sae_store(n_features=4)
        step = SAETopActivations(model, k=2, feat_ids=[0, 7])
        with pytest.raises(ValueError, match="out of range"):
            step(results)

    def test_topk_selects_largest_activations_per_feature(self, model):
        from murano.results import Results

        # 2 texts x 3 seq x 2 features. Top values per feature:
        # feat 0: 9.0 at (1, 2) then 5.0 at (0, 2)
        # feat 1: 8.0 at (0, 1) then 6.0 at (0, 2)
        acts = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 8.0], [5.0, 6.0]],
                [[2.0, 1.0], [4.0, 3.0], [9.0, 0.5]],
            ]
        )
        tokens = torch.tensor(
            [
                [VOCAB["hello"], VOCAB["world"], VOCAB["good"]],
                [VOCAB["bad"], VOCAB["hello"], VOCAB["world"]],
            ],
            dtype=torch.long,
        )
        attention_mask = torch.ones(2, 3, dtype=torch.long)
        store = _synthetic_sae_store(
            n=2,
            seq=3,
            n_features=2,
            activations=acts,
            tokens=tokens,
            attention_mask=attention_mask,
            texts=["first prompt", "second prompt"],
        )

        results = Results()
        results["sae_record"] = store
        results = SAETopActivations(model, k=2)(results)

        examples: SAEFeatureExamples = results["feature_examples"]
        assert examples.feat_ids == [0, 1]
        assert examples.k == 2
        assert examples.hook.layer == 0
        assert examples.act_vals[0] == [9.0, 5.0]
        assert examples.contexts[0] == ["second prompt", "first prompt"]
        assert examples.act_vals[1] == [8.0, 6.0]
        assert examples.contexts[1] == ["first prompt", "first prompt"]

    def test_topk_respects_attention_mask(self, model):
        from murano.results import Results

        # The largest raw value (10.0) is at a padded position and must not
        # appear in top-K.
        acts = torch.tensor([[[10.0], [1.0], [2.0]]])
        attention_mask = torch.tensor([[0, 1, 1]], dtype=torch.long)
        tokens = torch.tensor(
            [[VOCAB["<pad>"], VOCAB["hello"], VOCAB["world"]]], dtype=torch.long
        )
        store = _synthetic_sae_store(
            n=1,
            seq=3,
            n_features=1,
            activations=acts,
            tokens=tokens,
            attention_mask=attention_mask,
            texts=["only prompt"],
        )

        results = Results()
        results["sae_record"] = store
        results = SAETopActivations(model, k=2)(results)

        examples: SAEFeatureExamples = results["feature_examples"]
        assert 10.0 not in examples.act_vals[0]
        assert examples.act_vals[0] == [2.0, 1.0]

    def test_topk_subset_via_feat_ids(self, model):
        from murano.results import Results

        store = _synthetic_sae_store(n=1, seq=2, n_features=5)
        store.activations[:] = torch.arange(10, dtype=torch.float32).reshape(1, 2, 5)

        results = Results()
        results["sae_record"] = store
        step = SAETopActivations(model, k=1, feat_ids=[1, 3])
        results = step(results)

        examples: SAEFeatureExamples = results["feature_examples"]
        assert examples.feat_ids == [1, 3]
        assert set(examples.contexts.keys()) == {1, 3}

    def test_topk_propagates_release_and_sae_id(self, model):
        from murano.results import Results

        store = _synthetic_sae_store(
            n=1, seq=2, n_features=2, release="acme/release", sae_id="layer_0/canonical"
        )
        results = Results()
        results["sae_record"] = store
        results = SAETopActivations(model, k=1)(results)

        examples: SAEFeatureExamples = results["feature_examples"]
        assert examples.release == "acme/release"
        assert examples.sae_id == "layer_0/canonical"

    def test_topk_skips_bos_by_default(self, model):
        from murano.results import Results

        # Hand-craft: largest activation is at a BOS-token position.
        # With skip_bos=True (default) it must not appear.
        bos = VOCAB["<s>"]
        acts = torch.tensor([[[10.0], [3.0], [2.0]]])
        tokens = torch.tensor([[bos, VOCAB["hello"], VOCAB["world"]]], dtype=torch.long)
        store = _synthetic_sae_store(
            n=1,
            seq=3,
            n_features=1,
            activations=acts,
            tokens=tokens,
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            texts=["only prompt"],
        )

        results = Results()
        results["sae_record"] = store
        results = SAETopActivations(model, k=2)(results)

        examples: SAEFeatureExamples = results["feature_examples"]
        assert 10.0 not in examples.act_vals[0]
        assert examples.act_vals[0] == [3.0, 2.0]

    def test_topk_includes_bos_when_skip_disabled(self, model):
        from murano.results import Results

        bos = VOCAB["<s>"]
        acts = torch.tensor([[[10.0], [3.0], [2.0]]])
        tokens = torch.tensor([[bos, VOCAB["hello"], VOCAB["world"]]], dtype=torch.long)
        store = _synthetic_sae_store(
            n=1,
            seq=3,
            n_features=1,
            activations=acts,
            tokens=tokens,
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            texts=["only prompt"],
        )

        results = Results()
        results["sae_record"] = store
        results = SAETopActivations(model, k=2, skip_bos=False)(results)

        examples: SAEFeatureExamples = results["feature_examples"]
        assert examples.act_vals[0] == [10.0, 3.0]


class TestSAESave:
    """End-to-end Pipeline + Save round-trip with synthetic SAE artifacts."""

    def test_save_writes_sae_record_and_examples(self, model, tmp_path):
        from murano.results import Results

        results = Results()
        results["sae_record"] = _synthetic_sae_store(n=2, seq=3, n_features=4)
        results["sae_record"].activations[:] = torch.randn(2, 3, 4)
        SAETopActivations(model, k=2)(results)
        Save(output_dir=str(tmp_path))(results)

        sae_pt = tmp_path / "sae" / "sae_record.pt"
        examples_json = tmp_path / "sae" / "feature_examples.json"
        assert sae_pt.exists(), sorted(p.name for p in tmp_path.iterdir())
        assert examples_json.exists()

        loaded = torch.load(sae_pt, weights_only=False)
        original = results["sae_record"]
        assert torch.equal(loaded["activations"], original.activations)
        assert torch.equal(loaded["tokens"], original.tokens)
        assert torch.equal(loaded["attention_mask"], original.attention_mask)
        assert loaded["texts"] == original.texts
        assert loaded["hook"] == original.hook
        assert loaded["release"] == original.release
        assert loaded["sae_id"] == original.sae_id
        assert loaded["n_features"] == original.n_features

        examples_data = json.loads(examples_json.read_text())
        original_examples = results["feature_examples"]
        assert examples_data["feat_ids"] == original_examples.feat_ids
        # JSON serialization stringifies int keys; the load path would cast back.
        assert set(examples_data["contexts"].keys()) == {
            str(f) for f in original_examples.feat_ids
        }

    def test_save_records_sae_metadata(self, model, tmp_path):
        from murano.results import Results

        results = Results()
        results["sae_record"] = _synthetic_sae_store(
            n=2,
            seq=3,
            n_features=4,
            layer=1,
            release="acme/sae-v1",
            sae_id="layer_1/canonical",
        )
        SAETopActivations(model, k=2)(results)
        Save(output_dir=str(tmp_path))(results)

        metadata = json.loads((tmp_path / "metadata.json").read_text())
        assert "sae_record" in metadata
        assert metadata["sae_record"]["hook"] == "L1.resid_post"
        assert metadata["sae_record"]["release"] == "acme/sae-v1"
        assert metadata["sae_record"]["sae_id"] == "layer_1/canonical"
        assert metadata["sae_record"]["n_features"] == 4
        assert "feature_examples" in metadata
        assert metadata["feature_examples"]["k"] == 2
        assert metadata["feature_examples"]["n_tracked"] == 4


class TestSAELoadRoundTrip:
    """save then load returns an equivalent artifact."""

    def test_load_sae_activations_roundtrip(self, model, tmp_path):
        from murano.io import load_sae_activations
        from murano.results import Results

        results = Results()
        results["sae_record"] = _synthetic_sae_store(
            n=2,
            seq=3,
            n_features=4,
            layer=1,
            release="acme/sae-v1",
            sae_id="layer_1/canonical",
        )
        results["sae_record"].activations[:] = torch.randn(2, 3, 4)
        Save(output_dir=str(tmp_path))(results)

        loaded = load_sae_activations(tmp_path / "sae" / "sae_record.pt")
        original = results["sae_record"]
        assert torch.equal(loaded.activations, original.activations)
        assert torch.equal(loaded.tokens, original.tokens)
        assert torch.equal(loaded.attention_mask, original.attention_mask)
        assert loaded.texts == original.texts
        assert loaded.hook == original.hook
        assert loaded.release == original.release
        assert loaded.sae_id == original.sae_id
        assert loaded.n_features == original.n_features

    def test_load_sae_examples_roundtrip(self, model, tmp_path):
        from murano.io import load_sae_examples
        from murano.results import Results

        results = Results()
        results["sae_record"] = _synthetic_sae_store(n=2, seq=3, n_features=4)
        results["sae_record"].activations[:] = torch.randn(2, 3, 4)
        SAETopActivations(model, k=2)(results)
        Save(output_dir=str(tmp_path))(results)

        loaded = load_sae_examples(tmp_path / "sae" / "feature_examples.json")
        original = results["feature_examples"]
        assert loaded.feat_ids == original.feat_ids
        # Stringified JSON keys must come back as int.
        assert all(isinstance(k, int) for k in loaded.contexts)
        assert loaded.contexts == original.contexts
        assert loaded.tokens == original.tokens
        assert loaded.act_vals == original.act_vals
        assert loaded.hook == original.hook
        assert loaded.release == original.release
        assert loaded.sae_id == original.sae_id
        assert loaded.k == original.k

    def test_load_via_top_level(self, model, tmp_path):
        # Verify the lazy top-level exposure works.
        import murano
        from murano.results import Results

        results = Results()
        results["sae_record"] = _synthetic_sae_store(n=2, seq=3, n_features=4)
        Save(output_dir=str(tmp_path))(results)

        loaded = murano.load_sae_activations(tmp_path / "sae" / "sae_record.pt")
        assert loaded.hook == results["sae_record"].hook
