"""Smoke tests for the LogitLens step and its visualization."""

from __future__ import annotations

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
        assert result.all_probs.ndim == 4
        assert result.all_probs.shape[0] == n_layers
        assert result.all_probs.shape[1] == n_inputs
        assert result.all_probs.shape[3] == model.tokenizer.vocab_size
        assert result.max_probs.shape == result.all_probs.shape[:-1]
        assert result.predicted_tokens.shape == result.all_probs.shape[:-1]
        assert result.layer_indices == list(range(n_layers))
        assert len(result.input_words) == n_inputs
        assert len(result.predicted_words) == n_layers

    def test_layers_subset(self, model):
        pipe = Pipeline(
            [
                LoadPrompts(["hello"]),
                LogitLens(model, layers=[0]),
            ]
        )
        results = pipe.run()
        result = results["logit_lens"]
        assert result.all_probs.shape[0] == 1
        assert result.layer_indices == [0]

    def test_invalid_layers_string_raises(self, model):
        with pytest.raises(ValueError, match="layers as string must be 'all'"):
            LogitLens(model, layers="some")

    def test_probabilities_sum_to_one(self, model):
        pipe = Pipeline(
            [
                LoadPrompts(["hello world"]),
                LogitLens(model),
            ]
        )
        results = pipe.run()
        all_probs = results["logit_lens"].all_probs
        sums = all_probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


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
