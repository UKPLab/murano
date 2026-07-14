"""Smoke tests for the LogitLens step and its visualization."""

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

from murano import MuranoModel, Pipeline
from murano.nodes import RESID_POST, Node
from murano.steps import Save
from murano.steps.logit_lens import LogitLens, LogitLensResult
from murano.steps.prompts import LoadPrompts


def _build_tiny_local_model(path: Path) -> None:
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
    fast_tokenizer.save_pretrained(path)

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
    model.save_pretrained(path)


@pytest.fixture(scope="module")
def tiny_model_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny_model_logit")
    _build_tiny_local_model(path)
    return path


@pytest.fixture
def model(tiny_model_path):
    return MuranoModel(str(tiny_model_path), device_map="cpu", dtype=torch.float32)


class TestLogitLensStep:
    """Step-level smoke tests."""

    def test_pipeline_produces_logit_lens_result(self, model):
        pipe = Pipeline(
            [
                LoadPrompts(["hello world", "good world"]),
                LogitLens(model),
            ]
        )
        results = pipe.run()
        result = results["logit_lens"]

        assert isinstance(result, LogitLensResult)
        n_layers = model.n_layers
        n_inputs = 2
        # The full-vocab tensor is off by default (it OOMs on real models); the
        # reduced fields carry the standard logit-lens signal.
        assert result.all_probs is None
        assert result.max_probs.ndim == 3
        assert result.max_probs.shape[0] == n_layers
        assert result.max_probs.shape[1] == n_inputs
        assert result.predicted_tokens.shape == result.max_probs.shape
        assert result.addresses == [Node(i, RESID_POST) for i in range(n_layers)]
        assert len(result.input_words) == n_inputs
        assert len(result.predicted_words) == n_layers

    def test_store_full_probs_keeps_full_vocab_tensor(self, model):
        pipe = Pipeline(
            [
                LoadPrompts(["hello world", "good world"]),
                LogitLens(model, store_full_probs=True),
            ]
        )
        result = pipe.run()["logit_lens"]
        n_layers, n_inputs = model.n_layers, 2
        assert result.all_probs.ndim == 4
        assert result.all_probs.shape[0] == n_layers
        assert result.all_probs.shape[1] == n_inputs
        assert result.all_probs.shape[3] == model.tokenizer.vocab_size
        assert result.max_probs.shape == result.all_probs.shape[:-1]
        # The reduced fields must equal the reduction of the full tensor.
        mp, pt = result.all_probs.max(dim=-1)
        assert torch.allclose(result.max_probs, mp, atol=1e-6)
        assert torch.equal(result.predicted_tokens, pt)

    def test_layers_subset(self, model):
        pipe = Pipeline(
            [
                LoadPrompts(["hello"]),
                LogitLens(model, layers=[0]),
            ]
        )
        results = pipe.run()
        result = results["logit_lens"]
        assert result.max_probs.shape[0] == 1
        assert result.addresses == [Node(0, RESID_POST)]

    def test_invalid_layers_string_raises(self, model):
        with pytest.raises(ValueError, match="layers as string must be 'all'"):
            LogitLens(model, layers="some")

    def test_probabilities_sum_to_one(self, model):
        pipe = Pipeline(
            [
                LoadPrompts(["hello world"]),
                LogitLens(model, store_full_probs=True),
            ]
        )
        results = pipe.run()
        all_probs = results["logit_lens"].all_probs
        sums = all_probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test_last_layer_lens_matches_model_logits(self, model):
        """Ground truth: the last-layer lens is the model's real output.

        ``project_on_vocab`` applies the model's own final norm and unembedding,
        so the logit lens at the final ``resid_post`` must reproduce the model's
        true next-token distribution. This catches a wrong norm, a wrong layer,
        or a wrong projection, which a shape/sum-to-one check cannot.
        """
        prompts = ["hello world", "good world"]
        result = Pipeline([LoadPrompts(prompts), LogitLens(model)]).run()["logit_lens"]

        true_logits = model.logits(prompts)  # [B, S, V], the model's real output
        true_pred = true_logits.argmax(dim=-1)  # [B, S]
        # predicted_tokens is [n_layers, B, S]; the last layer is the model output.
        last_layer_pred = result.predicted_tokens[-1]  # [B, S]
        assert torch.equal(last_layer_pred, true_pred)


class TestLogitLensSave:
    """Round-trip persistence via Pipeline + Save."""

    def test_save_writes_logit_lens_artifact(self, model, tmp_path):
        pipe = Pipeline(
            [
                LoadPrompts(["hello world", "good world"]),
                LogitLens(model, layers=[0, 1], store_full_probs=True),
                Save(output_dir=str(tmp_path)),
            ]
        )
        results = pipe.run()

        artifact_path = tmp_path / "logit_lens" / "logit_lens.pt"
        assert artifact_path.exists(), (
            f"Save did not write LogitLens artifact. "
            f"Found: {sorted(p.name for p in tmp_path.iterdir())}"
        )

        loaded = torch.load(artifact_path, weights_only=False)
        original = results["logit_lens"]
        assert isinstance(original, LogitLensResult)
        assert torch.equal(loaded["all_probs"], original.all_probs)
        assert torch.equal(loaded["max_probs"], original.max_probs)
        assert torch.equal(loaded["predicted_tokens"], original.predicted_tokens)
        assert torch.equal(loaded["attention_mask"], original.attention_mask)
        assert loaded["predicted_words"] == original.predicted_words
        assert loaded["input_words"] == original.input_words
        assert loaded["addresses"] == original.addresses

    def test_save_records_logit_lens_in_metadata(self, model, tmp_path):
        pipe = Pipeline(
            [
                LoadPrompts(["hello world"]),
                LogitLens(model, layers=[0, 1]),
                Save(output_dir=str(tmp_path)),
            ]
        )
        pipe.run()

        metadata = json.loads((tmp_path / "metadata.json").read_text())
        assert "logit_lens" in metadata
        assert metadata["logit_lens"]["addresses"] == ["L0.resid_post", "L1.resid_post"]
        assert metadata["logit_lens"]["n_layers"] == 2
        assert metadata["logit_lens"]["n_inputs"] == 1

    def test_load_logit_lens_roundtrip(self, model, tmp_path):
        from murano.io import load_logit_lens

        results = Pipeline(
            [
                LoadPrompts(["hello world", "good world"]),
                LogitLens(model, layers=[0, 1], store_full_probs=True),
                Save(output_dir=str(tmp_path)),
            ]
        ).run()

        loaded = load_logit_lens(tmp_path / "logit_lens" / "logit_lens.pt")
        original = results["logit_lens"]
        assert isinstance(loaded, LogitLensResult)
        assert torch.equal(loaded.all_probs, original.all_probs)
        assert torch.equal(loaded.max_probs, original.max_probs)
        assert torch.equal(loaded.predicted_tokens, original.predicted_tokens)
        assert torch.equal(loaded.attention_mask, original.attention_mask)
        assert loaded.predicted_words == original.predicted_words
        assert loaded.input_words == original.input_words
        assert loaded.addresses == original.addresses

    def test_save_routes_custom_key_to_keyed_filename(self, model, tmp_path):
        from murano.results import Results

        pipe_results = Pipeline(
            [
                LoadPrompts(["hello world"]),
                LogitLens(model, layers=[0]),
            ]
        ).run()
        results = Results()
        results["lens_baseline"] = pipe_results["logit_lens"]
        Save(output_dir=str(tmp_path))(results)

        assert (tmp_path / "logit_lens" / "lens_baseline.pt").exists()
        assert not (tmp_path / "logit_lens" / "logit_lens.pt").exists()


class TestLogitLensPlot:
    """Visualization smoke tests."""

    def test_plot_returns_plotly_figure(self, model):
        pytest.importorskip("plotly")
        from murano.plotting.logit_lens import plot_logit_lens

        pipe = Pipeline(
            [
                LoadPrompts(["hello world"]),
                LogitLens(model),
            ]
        )
        results = pipe.run()
        fig = plot_logit_lens(results["logit_lens"], title="Test")

        # Validate via to_dict() (CI-safe; no headless rendering).
        data = fig.to_dict()
        assert "data" in data
        assert data["data"][0]["type"] == "heatmap"
        assert data["layout"]["title"]["text"] == "Test"
