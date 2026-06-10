"""Tests for the ported example scripts.

Uses a tiny local model (2-layer Llama) so tests run without downloading.
"""

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
from murano.dataset import MuranoDataset
from murano.steps.intervene import Intervene, InterveneResult, steer_direction
from murano.steps.load import Load
from murano.steps.record import ActivationStore, Record
from murano.steps.train import SteeringResult, SteeringVector


# ── Helpers ──────────────────────────────────────────────────────────


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
        "prompt": 8,
        "response": 9,
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


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def tiny_model_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny_model")
    _build_tiny_local_model(path)
    return path


@pytest.fixture
def model(tiny_model_path):
    return MuranoModel(str(tiny_model_path), device_map="cpu", dtype=torch.float32)


@pytest.fixture
def contrastive_dataset():
    return MuranoDataset(
        positive_texts=["hello world", "good world"],
        negative_texts=["bad world", "bad prompt"],
    )


# ── Tests for federico_visualization.py logic ────────────────────────


class TestFedericoVisualization:
    """Tests the core logic of the ported federico_visualization example."""

    def test_plotter_lens_accepts_activation_store(self, model, contrastive_dataset):
        """PlotterLens.observe should accept an ActivationStore from Record."""
        from examples.federico_visualization import PlotterLens
        from sklearn.decomposition import PCA

        pipe = Pipeline(
            [
                Load(contrastive_dataset),
                Record(model, layers=[0], position="last", batch_size=2),
            ]
        )
        results = pipe.run()
        store: ActivationStore = results["record"]

        plotter = PlotterLens(
            reducer=PCA(n_components=2),
            save_path="/tmp/test_activation_pca.html",
        )
        # Should not raise
        plotter.observe(store, layer=0, title="Test PCA")

    def test_plotter_lens_with_lda(self, model, contrastive_dataset):
        """PlotterLens should work with LDA (supervised reducer)."""
        from examples.federico_visualization import PlotterLens
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        pipe = Pipeline(
            [
                Load(contrastive_dataset),
                Record(model, layers=[0], position="last", batch_size=2),
            ]
        )
        results = pipe.run()
        store: ActivationStore = results["record"]

        plotter = PlotterLens(
            reducer=LinearDiscriminantAnalysis(n_components=1),
            save_path="/tmp/test_activation_lda.html",
        )
        plotter.observe(store, layer=0, title="Test LDA")


# ── Tests for record_intervene.py logic ──────────────────────────────


class TestRecordIntervene:
    """Tests the core logic of the ported record_intervene example."""

    def test_pipeline_record_steering(self, model, contrastive_dataset):
        """Pipeline: Load → Record → SteeringVector should produce SteeringResult."""
        pipe = Pipeline(
            [
                Load(contrastive_dataset),
                Record(model, layers=[0], position="last", batch_size=2),
                SteeringVector(normalize=True),
            ]
        )
        results = pipe.run()
        steering: SteeringResult = results["steering"]

        assert (0, "residual") in steering.direction_per_layer
        assert steering.direction_per_layer[(0, "residual")].shape == (model.d_model,)
        assert (0, "residual") in steering.separation_scores
        assert steering.best_layer == (0, "residual")

    def test_pipeline_intervene(self, model, contrastive_dataset):
        """Pipeline: Load → Intervene should produce InterveneResult."""
        # First compute a steering direction
        train_pipe = Pipeline(
            [
                Load(contrastive_dataset),
                Record(model, layers=[0], position="last", batch_size=2),
                SteeringVector(normalize=True),
            ]
        )
        train_results = train_pipe.run()
        steering: SteeringResult = train_results["steering"]

        # Now intervene
        steer_fn = steer_direction(steering.direction_per_layer, alpha=1.0)
        eval_dataset = MuranoDataset(
            positive_texts=["hello", "good"],
            negative_texts=[],
        )
        eval_pipe = Pipeline(
            [
                Load(eval_dataset),
                Intervene(
                    model,
                    fn=steer_fn,
                    layers=[0],
                    gen_kwargs={"max_new_tokens": 1, "do_sample": False},
                ),
            ]
        )
        eval_results = eval_pipe.run()
        intervene_result: InterveneResult = eval_results["intervene"]

        assert len(intervene_result.clean_generations) == 2
        assert len(intervene_result.modified_generations) == 2
        assert isinstance(intervene_result.clean_generations[0], str)
        assert isinstance(intervene_result.modified_generations[0], str)

    def test_full_record_intervene_flow(self, model):
        """End-to-end: record → steer → intervene with small synthetic data."""
        dataset = MuranoDataset(
            positive_texts=["hello world", "good world"],
            negative_texts=["bad world", "bad prompt"],
        )

        # Train
        train_pipe = Pipeline(
            [
                Load(dataset),
                Record(model, layers=[0], position="last", batch_size=2),
                SteeringVector(normalize=True),
            ]
        )
        train_results = train_pipe.run()
        steering: SteeringResult = train_results["steering"]

        # Evaluate
        steer_fn = steer_direction(steering.direction_per_layer, alpha=2.0)
        eval_dataset = MuranoDataset(
            positive_texts=["hello"],
            negative_texts=[],
        )
        eval_pipe = Pipeline(
            [
                Load(eval_dataset),
                Intervene(
                    model,
                    fn=steer_fn,
                    layers=[0],
                    gen_kwargs={"max_new_tokens": 1, "do_sample": False},
                ),
            ]
        )
        eval_results = eval_pipe.run()
        intervene_result: InterveneResult = eval_results["intervene"]

        # Basic sanity: clean and steered outputs should be strings
        assert isinstance(intervene_result.clean_generations[0], str)
        assert isinstance(intervene_result.modified_generations[0], str)
