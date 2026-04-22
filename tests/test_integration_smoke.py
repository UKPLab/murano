"""Tiny local-model integration smoke tests."""

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

from murano.model import MuranoModel
from murano.steps.record import ActivationStore
from murano.steps.train import SteeringResult


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


def test_quick_api_smoke_with_local_model(tmp_path):
    _build_tiny_local_model(tmp_path)

    model = MuranoModel(str(tmp_path), device_map="cpu", dtype=torch.float32)

    record = model.record(
        ["hello world", "good world"],
        layers=[0],
        position="mean",
        batch_size=2,
    )
    assert isinstance(record, ActivationStore)
    assert record.positive[0].shape == (2, model.d_model)
    assert record.negative == {}

    steering = model.find_direction(
        ["good world"],
        ["bad world"],
        layers=[0],
        position="first",
        batch_size=1,
    )
    assert isinstance(steering, SteeringResult)
    assert steering.best_layer == 0
    assert steering.direction_per_layer[0].shape == (model.d_model,)

    generation = model.generate(
        "hello",
        gen_kwargs={"max_new_tokens": 1, "do_sample": False},
    )
    assert isinstance(generation, str)
