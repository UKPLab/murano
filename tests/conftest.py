"""Test configuration for src-layout imports."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _build_tiny_local_model(path: Path) -> None:
    """Build a tiny Llama model for test fixtures."""
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


@pytest.fixture(scope="session")
def murano_model(tmp_path_factory):
    """Fixture that builds a tiny local model and returns a MuranoModel instance."""
    from murano.model import MuranoModel

    model_dir = tmp_path_factory.mktemp("tiny_model")
    _build_tiny_local_model(model_dir)
    return MuranoModel(str(model_dir), device_map="cpu", dtype=torch.float32)
