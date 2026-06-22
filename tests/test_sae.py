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
    SAEFeatureLabel,
    SAEFeatureLabels,
    SAEModel,
    SAETopActivations,
    _steer_site,
    sae_steer,
    top_sae_features_for_tokens,
    top_sae_features_per_prompt,
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
        d_model: int = 32,
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
        self.W_dec = torch.arange(d_sae * d_model, dtype=torch.float32).reshape(
            d_sae, d_model
        )

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

    def test_feature_direction_returns_decoder_row(self):
        sae = SAEModel(release="acme/sae", sae_id="layer_0/canonical")
        sae._sae = _FakeSAE(d_sae=4, d_model=8)
        direction = sae.feature_direction(2)
        assert direction.shape == (8,)
        assert torch.equal(direction, sae._sae.W_dec[2])
        # Must be a detached copy: mutating the returned tensor
        # leaves W_dec untouched.
        direction[0] = -999.0
        assert sae._sae.W_dec[2, 0].item() != -999.0


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


class TestTopSaeFeaturesPerPrompt:
    """Picks features at each prompt's last real token, excluding BOS sinks."""

    def test_picks_top_at_last_real_token(self):
        record = _synthetic_sae_store(n=2, seq=4, n_features=4)
        # attention_mask defaults to all 1s, so last pos = 3 for both.
        record.activations[0, 3] = torch.tensor([0.1, 5.0, 2.0, 8.0])
        record.activations[1, 3] = torch.tensor([0.5, 0.4, 9.0, 0.1])
        # sink_position=-1 disables the sink filter (no position equals -1).
        assert top_sae_features_per_prompt(record, n=1, sink_position=-1) == [[3], [2]]

    def test_respects_attention_mask(self):
        record = _synthetic_sae_store(n=1, seq=4, n_features=4)
        # Mask 1,1,0,0 -> last real position is index 1.
        record.attention_mask = torch.tensor([[1, 1, 0, 0]])
        record.activations[0, 1] = torch.tensor([5.0, 0.1, 0.1, 0.1])
        # Padded positions must be ignored.
        record.activations[0, 3] = torch.tensor([0.0, 99.0, 99.0, 99.0])
        # sink_position=-1 disables the sink filter to test attention masking
        # alone (feature 0 has peak at masked position 3 of vocab; but we want
        # to test that last_acts uses position 1 not 3).
        assert top_sae_features_per_prompt(record, n=1, sink_position=-1) == [[0]]

    def test_excludes_bos_sink_features(self):
        # 3 prompts, 3 features. Feature 0 is a BOS sink: its global peak
        # across the batch is at position 1 (the first content token, the
        # canonical sink position). Features 1 and 2 peak at semantic
        # positions and should win at the last token.
        record = _synthetic_sae_store(n=3, seq=4, n_features=3)
        # Feature 0: huge spike at position 1 of every prompt = global sink.
        record.activations[:, 1, 0] = 500.0
        # Feature 1 wins prompt 0's last token (and peaks at the last token).
        record.activations[0, 3, 1] = 50.0
        # Feature 2 wins prompt 1's last token.
        record.activations[1, 3, 2] = 50.0
        # Feature 0 still has the largest activation at the last token of
        # prompt 2 (say 10), but its global peak is at position 1, so it
        # gets filtered. Prompt 2 falls back to feature 1 or 2 (both 0 at
        # last position); not asserted here, we only check sink exclusion.
        record.activations[2, 3, 0] = 10.0
        top = top_sae_features_per_prompt(record, n=1)
        # Sink feature 0 is excluded; legitimate features win.
        assert top[0] == [1]
        assert top[1] == [2]

    def test_reduce_mean_pools_over_content_tokens(self):
        # 1 prompt, 4 tokens, 3 features. Positions 0-1 are the BOS-sink region
        # (excluded from the mean). Over content positions 2-3, feature 2 has
        # the highest average, so reduce="mean" picks it.
        record = _synthetic_sae_store(n=1, seq=4, n_features=3)
        record.attention_mask = torch.ones(1, 4, dtype=torch.long)
        record.activations.zero_()
        record.activations[0, 0] = torch.tensor([99.0, 0.0, 0.0])  # sink pos, excluded
        record.activations[0, 1] = torch.tensor([99.0, 0.0, 0.0])  # sink pos, excluded
        record.activations[0, 2] = torch.tensor([0.0, 1.0, 8.0])
        record.activations[0, 3] = torch.tensor([0.0, 1.0, 6.0])
        # max_density=1.0 disables the broad-feature filter so this isolates
        # the pooling behavior.
        assert top_sae_features_per_prompt(
            record, n=1, reduce="mean", max_density=1.0
        ) == [[2]]

    def test_reduce_mean_drops_broad_features(self):
        # Feature 0 fires on every content token (broad/junk); feature 2 fires
        # once on prompt 0. With max_density=0.4, feature 0 (density 0.5) is
        # dropped, so prompt 0 surfaces feature 2 despite feature 0's higher
        # mean.
        record = _synthetic_sae_store(n=2, seq=4, n_features=3)
        record.attention_mask = torch.ones(2, 4, dtype=torch.long)
        record.activations.zero_()
        record.activations[:, 2:, 0] = 9.0  # feature 0 on all content tokens
        record.activations[0, 2, 2] = 5.0  # feature 2 once, prompt 0
        out = top_sae_features_per_prompt(record, n=1, reduce="mean", max_density=0.4)
        assert out[0] == [2]

    def test_rejects_unknown_reduce(self):
        record = _synthetic_sae_store(n_features=5)
        with pytest.raises(ValueError, match="reduce must be"):
            top_sae_features_per_prompt(record, reduce="median")

    def test_rejects_n_below_one(self):
        record = _synthetic_sae_store(n_features=5)
        with pytest.raises(ValueError, match=r"n must be in \[1, 5\]"):
            top_sae_features_per_prompt(record, n=0)

    def test_rejects_n_above_n_features(self):
        record = _synthetic_sae_store(n_features=5)
        with pytest.raises(ValueError, match=r"n must be in \[1, 5\]"):
            top_sae_features_per_prompt(record, n=6)

    def test_rejects_all_padding_row(self):
        record = _synthetic_sae_store(n=1, seq=4, n_features=3)
        record.attention_mask = torch.tensor([[0, 0, 0, 0]])
        with pytest.raises(ValueError, match="at least one real token"):
            top_sae_features_per_prompt(record, n=1)

    def test_all_sinks_returns_empty_not_leaked_ids(self):
        # Every feature peaks at position <= sink_position, so the filter
        # masks them all. The function must return [] per prompt, never the
        # masked sink ids it was meant to exclude.
        record = _synthetic_sae_store(n=2, seq=4, n_features=3)
        record.activations.zero_()
        record.activations[:, 1, :] = 100.0  # all features spike at position 1
        assert top_sae_features_per_prompt(record, n=1) == [[], []]

    def test_drops_sink_slots_when_fewer_than_n_survive(self):
        # n=2 but only one non-sink feature survives at the last token; the
        # result must contain that one id, not a padded sink id.
        record = _synthetic_sae_store(n=1, seq=4, n_features=3)
        record.activations.zero_()
        record.activations[0, 1, 0] = 500.0  # feature 0 is a sink (peak at pos 1)
        record.activations[0, 3, 1] = 5.0  # feature 1 fires at last token
        # feature 2 never fires (peaks at flat index 0 -> position 0 -> sink)
        assert top_sae_features_per_prompt(record, n=2) == [[1]]


class TestTopSaeFeaturesForTokens:
    """Picks features by mean activation on a set of target tokens."""

    def test_ranks_features_on_target_token_positions(self, model):
        # VOCAB id 4 = "hello". Put it at position 2 of both prompts.
        record = _synthetic_sae_store(n=2, seq=4, n_features=3)
        record.tokens = torch.tensor([[1, 5, 4, 5], [1, 6, 4, 7]])
        record.attention_mask = torch.ones(2, 4, dtype=torch.long)
        record.activations.zero_()
        # Feature 1 fires on the "hello" positions; feature 0 fires elsewhere.
        record.activations[0, 2, 1] = 9.0
        record.activations[1, 2, 1] = 7.0
        record.activations[0, 1, 0] = 50.0  # not a target position, and skipped
        assert top_sae_features_for_tokens(record, model, {"hello"}, n=1) == [1]

    def test_raises_when_no_target_token_matches(self, model):
        record = _synthetic_sae_store(n=1, seq=4, n_features=3)
        record.tokens = torch.tensor([[1, 5, 5, 5]])  # no "hello"
        record.attention_mask = torch.ones(1, 4, dtype=torch.long)
        with pytest.raises(ValueError, match="no token in target_tokens"):
            top_sae_features_for_tokens(record, model, {"hello"}, n=1)

    def test_rejects_n_below_one(self, model):
        record = _synthetic_sae_store()
        with pytest.raises(ValueError, match="n must be >= 1"):
            top_sae_features_for_tokens(record, model, {"hello"}, n=0)


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

    def test_forces_right_padding_even_if_tokenizer_left_pads(self, model):
        # Gemma and many instruct tokenizers default to left padding, which
        # would shift the BOS sink and last-token positions the feature helpers
        # rely on. SAEEncode must force right padding regardless.
        from murano.artifacts import PromptBatch
        from murano.results import Results

        model.tokenizer.padding_side = "left"  # simulate a left-padding model
        results = Results()
        results["prompts"] = PromptBatch(prompts=["hello world good", "bad"])
        step = SAEEncode(model, release="test/repo", sae_id="test/id")
        step.sae_model._sae = _FakeSAE(hook_layer=0)

        results = step(results)
        mask = results["sae_record"].attention_mask
        # Right-padded: real tokens flush-left, so column 0 is always real and
        # every row's real tokens are a contiguous prefix (no leading pad).
        assert bool((mask[:, 0] == 1).all())
        for row in mask.tolist():
            ones = sum(row)
            assert row[:ones] == [1] * ones  # all real tokens at the front
        # SAEEncode must restore the tokenizer's original padding side.
        assert model.tokenizer.padding_side == "left"

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


def _make_fake_sae_model(d_sae: int, d_model: int) -> SAEModel:
    """Build an SAEModel wrapping a _FakeSAE for SAEFeatureLabel tests."""
    sae = SAEModel(release="acme/sae", sae_id="layer_0/canonical")
    sae._sae = _FakeSAE(d_sae=d_sae, d_model=d_model)
    return sae


class TestSAEFeatureLabel:
    """SAEFeatureLabel projects decoder directions through ln_final + lm_head."""

    def test_declares_correct_reads_writes(self):
        assert SAEFeatureLabel.reads == ["sae_record"]
        assert SAEFeatureLabel.writes == ["feature_labels"]

    def test_rejects_k_tokens_below_one(self, model):
        with pytest.raises(ValueError, match="k_tokens must be >= 1"):
            SAEFeatureLabel(model, feat_ids=[0], k_tokens=0)

    def test_rejects_out_of_range_feat_ids(self, model):
        from murano.results import Results

        d_model = model._lm.lm_head.in_features
        fake = _make_fake_sae_model(d_sae=4, d_model=d_model)
        results = Results()
        results["sae_record"] = _synthetic_sae_store(n_features=4)
        with pytest.raises(ValueError, match="out of range"):
            SAEFeatureLabel(model, feat_ids=[0, 9], sae_model=fake)(results)

    def test_label_top_token_matches_target_row(self, model):
        # Set W_dec[0] to amplify the lm_head row for token "hello" so the
        # projection lm_head(ln_final(W_dec[0])) returns "hello" as top-1.
        from murano.results import Results

        d_model = model._lm.lm_head.in_features
        fake = _make_fake_sae_model(d_sae=3, d_model=d_model)
        with torch.no_grad():
            target_row = model._lm.lm_head.weight[VOCAB["hello"]].detach().float()
            fake._sae.W_dec[0] = target_row * 100.0

        results = Results()
        results["sae_record"] = _synthetic_sae_store(n_features=3)
        SAEFeatureLabel(model, feat_ids=[0], k_tokens=2, sae_model=fake)(results)

        labels = results["feature_labels"]
        assert isinstance(labels, SAEFeatureLabels)
        assert labels.feat_ids == [0]
        assert labels.tokens[0][0] == "hello"
        # Logits returned in descending order.
        assert labels.logits[0][0] >= labels.logits[0][1]
        assert len(labels.tokens[0]) == 2
        assert len(labels.logits[0]) == 2

    def test_handles_w_dec_dtype_mismatch_with_model_head(self, model):
        # W_dec from a separately loaded SAE routinely differs in dtype from
        # the base model head (e.g. fp16 SAE vs fp32 model). project_on_vocab
        # must cast before the matmul; without it the projection would raise.
        from murano.results import Results

        d_model = model._lm.lm_head.in_features
        fake = _make_fake_sae_model(d_sae=3, d_model=d_model)
        with torch.no_grad():
            target_row = model._lm.lm_head.weight[VOCAB["hello"]].detach().float()
            fake._sae.W_dec[0] = (target_row * 100.0).to(torch.float16)

        results = Results()
        results["sae_record"] = _synthetic_sae_store(n_features=3)
        SAEFeatureLabel(model, feat_ids=[0], k_tokens=1, sae_model=fake)(results)
        assert results["feature_labels"].tokens[0][0] == "hello"

    def test_propagates_layer_release_sae_id(self, model):
        from murano.results import Results

        d_model = model._lm.lm_head.in_features
        fake = _make_fake_sae_model(d_sae=4, d_model=d_model)
        results = Results()
        results["sae_record"] = _synthetic_sae_store(
            n_features=4,
            layer=1,
            release="acme/sae-v1",
            sae_id="layer_1/canonical",
        )
        SAEFeatureLabel(model, feat_ids=[0, 2], k_tokens=3, sae_model=fake)(results)

        labels = results["feature_labels"]
        assert labels.layer == 1
        assert labels.release == "acme/sae-v1"
        assert labels.sae_id == "layer_1/canonical"
        assert labels.k_tokens == 3
        assert set(labels.tokens.keys()) == {0, 2}


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

    def test_save_writes_feature_labels(self, model, tmp_path):
        from murano.results import Results

        results = Results()
        results["feature_labels"] = SAEFeatureLabels(
            feat_ids=[0, 5],
            tokens={0: ["hello", "world"], 5: ["good", "bad"]},
            logits={0: [3.5, 2.1], 5: [4.0, 1.7]},
            k_tokens=2,
            layer=8,
            release="acme/sae-v1",
            sae_id="layer_8/canonical",
        )
        Save(output_dir=str(tmp_path))(results)

        labels_json = tmp_path / "sae" / "feature_labels.json"
        assert labels_json.exists()
        data = json.loads(labels_json.read_text())
        assert data["feat_ids"] == [0, 5]
        assert data["tokens"]["0"] == ["hello", "world"]
        assert data["k_tokens"] == 2

        metadata = json.loads((tmp_path / "metadata.json").read_text())
        assert "feature_labels" in metadata
        assert metadata["feature_labels"]["k_tokens"] == 2
        assert metadata["feature_labels"]["n_labeled"] == 2


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

    def test_load_sae_labels_roundtrip(self, model, tmp_path):
        from murano.io import load_sae_labels
        from murano.results import Results

        original = SAEFeatureLabels(
            feat_ids=[0, 5],
            tokens={0: ["hello", "world"], 5: ["good", "bad"]},
            logits={0: [3.5, 2.1], 5: [4.0, 1.7]},
            k_tokens=2,
            layer=1,
            release="acme/sae-v1",
            sae_id="layer_1/canonical",
        )
        results = Results()
        results["feature_labels"] = original
        Save(output_dir=str(tmp_path))(results)

        loaded = load_sae_labels(tmp_path / "sae" / "feature_labels.json")
        assert loaded.feat_ids == original.feat_ids
        # Int keys survive the JSON round-trip.
        assert loaded.tokens == original.tokens
        assert loaded.logits == original.logits
        assert loaded.k_tokens == original.k_tokens
        assert loaded.layer == original.layer
        assert loaded.release == original.release
        assert loaded.sae_id == original.sae_id


class TestSteerSite:
    """_steer_site maps each SAE hook kind to the matching intervention site."""

    def test_resid_post_maps_to_same_layer_residual(self):
        assert _steer_site(5, "resid_post") == (5, "residual")

    def test_mlp_out_maps_to_mlp(self):
        assert _steer_site(3, "mlp_out") == (3, "mlp")

    def test_attn_out_maps_to_self_attn(self):
        assert _steer_site(7, "attn_out") == (7, "self_attn")

    def test_resid_pre_maps_to_previous_layer_residual(self):
        # resid_pre of L is the resid_post of L-1.
        assert _steer_site(4, "resid_pre") == (3, "residual")

    def test_resid_pre_at_layer_zero_raises(self):
        with pytest.raises(ValueError, match="layer 0"):
            _steer_site(0, "resid_pre")

    def test_resid_mid_is_not_supported(self):
        with pytest.raises(NotImplementedError, match="resid_mid"):
            _steer_site(2, "resid_mid")

    def test_unknown_hook_kind_is_not_supported(self):
        with pytest.raises(NotImplementedError):
            _steer_site(2, "something_weird")


class TestSaeSteer:
    """sae_steer builds an Intervene step targeting the SAE's training site."""

    @staticmethod
    def _sae(hook_name: str, hook_layer: int) -> SAEModel:
        sae = SAEModel(release="acme/sae", sae_id="layer/canonical")
        sae._sae = _FakeSAE(hook_name=hook_name, hook_layer=hook_layer)
        return sae

    def test_returns_intervene_for_resid_post(self, model):
        from murano.steps.intervene import Intervene

        sae = self._sae("blocks.1.hook_resid_post", 1)
        step = sae_steer(model, sae, feature_id=2, alpha=10.0)
        assert isinstance(step, Intervene)
        assert step.model is model
        assert step.layers == [1]
        assert step.modules == ["residual"]

    def test_mlp_out_targets_mlp_module(self, model):
        sae = self._sae("blocks.1.hook_mlp_out", 1)
        step = sae_steer(model, sae, feature_id=0, alpha=5.0)
        assert step.layers == [1]
        assert step.modules == ["mlp"]

    def test_attn_out_targets_self_attn_module(self, model):
        sae = self._sae("blocks.0.hook_attn_out", 0)
        step = sae_steer(model, sae, feature_id=0, alpha=5.0)
        assert step.layers == [0]
        assert step.modules == ["self_attn"]

    def test_resid_pre_steers_at_previous_layer(self, model):
        # resid_pre of layer 1 == resid_post of layer 0.
        sae = self._sae("blocks.1.hook_resid_pre", 1)
        step = sae_steer(model, sae, feature_id=0, alpha=5.0)
        assert step.layers == [0]
        assert step.modules == ["residual"]

    def test_resid_pre_at_layer_zero_raises(self, model):
        sae = self._sae("blocks.0.hook_resid_pre", 0)
        with pytest.raises(ValueError, match="layer 0"):
            sae_steer(model, sae, feature_id=0, alpha=5.0)

    def test_resid_mid_raises_not_implemented(self, model):
        sae = self._sae("blocks.0.hook_resid_mid", 0)
        with pytest.raises(NotImplementedError, match="resid_mid"):
            sae_steer(model, sae, feature_id=0, alpha=5.0)

    def test_steer_site_out_of_bounds_raises(self, model):
        # hook_layer == n_layers is out of range for the model.
        sae = self._sae("blocks.2.hook_resid_post", model.n_layers)
        with pytest.raises(ValueError, match="out of range"):
            sae_steer(model, sae, feature_id=0, alpha=5.0)

    def test_default_gen_kwargs_when_omitted(self, model):
        sae = self._sae("blocks.0.hook_resid_post", 0)
        step = sae_steer(model, sae, feature_id=0, alpha=5.0)
        assert step.gen_kwargs == {"max_new_tokens": 256, "do_sample": False}

    def test_custom_gen_kwargs_pass_through(self, model):
        sae = self._sae("blocks.0.hook_resid_post", 0)
        gen_kwargs = {"max_new_tokens": 8, "do_sample": True}
        step = sae_steer(model, sae, feature_id=0, alpha=5.0, gen_kwargs=gen_kwargs)
        assert step.gen_kwargs == gen_kwargs

    def test_fn_adds_normalized_feature_direction_at_resolved_layer(self, model):
        from murano.nodes import Node

        sae = self._sae("blocks.1.hook_resid_post", 1)
        alpha = 3.0
        step = sae_steer(model, sae, feature_id=2, alpha=alpha)
        layer, module = step.layers[0], step.modules[0]

        direction = sae.feature_direction(2).float()
        d_hat = direction / direction.norm()
        act = torch.zeros(1, 1, direction.shape[0])

        # The fn is keyed by the Node the generation hook uses.
        steered = step.fn(act, Node(layer, module))
        assert torch.allclose(steered[0, 0], alpha * d_hat, atol=1e-5)

        # A different node is left untouched.
        untouched = step.fn(act, Node(layer + 1, module))
        assert torch.equal(untouched, act)
