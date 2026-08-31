"""Test configuration for src-layout imports."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

if TYPE_CHECKING:
    from murano.steps.gradients import RolloutBatch


def toy_rollouts(n: int, length: int, prompt_len: int, seed: int) -> RolloutBatch:
    """Deterministic fake rollouts over the tiny fixture vocab (ids 4..9)."""
    from murano.steps.gradients import RolloutBatch

    generator = torch.Generator().manual_seed(seed)
    ids = [torch.randint(4, 10, (length,), generator=generator) for _ in range(n)]
    return RolloutBatch(input_ids=ids, prompt_lengths=[prompt_len] * n)


_VOCAB = {
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


def _build_word_tokenizer(path: Path) -> None:
    """Save a tiny WordLevel tokenizer shared by the test model fixtures."""
    tokenizer = Tokenizer(WordLevel(vocab=dict(_VOCAB), unk_token="<unk>"))
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
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)
    model.save_pretrained(path)


@pytest.fixture(scope="session")
def murano_model(tmp_path_factory):
    """Fixture that builds a tiny local model and returns a MuranoModel instance."""
    from murano.model import MuranoModel

    model_dir = tmp_path_factory.mktemp("tiny_model")
    _build_tiny_local_model(model_dir)
    return MuranoModel(str(model_dir), device_map="cpu", dtype=torch.float32)


def _build_tiny_gpt2(path: Path) -> None:
    """Build a tiny GPT-2 model (LayerNorm + transposed Conv1D projections)."""
    _build_word_tokenizer(path)
    config = GPT2Config(
        vocab_size=len(_VOCAB),
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_positions=64,
        pad_token_id=_VOCAB["<pad>"],
        bos_token_id=_VOCAB["<s>"],
        eos_token_id=_VOCAB["</s>"],
    )
    torch.manual_seed(0)
    GPT2LMHeadModel(config).save_pretrained(path)


@pytest.fixture(scope="session")
def gpt2_model(tmp_path_factory):
    """A tiny GPT-2 MuranoModel, for exercising the LayerNorm code paths on CPU."""
    from murano.model import MuranoModel

    model_dir = tmp_path_factory.mktemp("tiny_gpt2")
    _build_tiny_gpt2(model_dir)
    return MuranoModel(str(model_dir), device_map="cpu", dtype=torch.float32)


@pytest.fixture(scope="session")
def murano_model_attn(tmp_path_factory):
    """A tiny Llama MuranoModel with attention weights enabled (eager attention)."""
    from murano.model import MuranoModel

    model_dir = tmp_path_factory.mktemp("tiny_model_attn")
    _build_tiny_local_model(model_dir)
    return MuranoModel(
        str(model_dir),
        device_map="cpu",
        dtype=torch.float32,
        enable_attention_probs=True,
    )


@pytest.fixture(scope="session")
def gpt2_model_attn(tmp_path_factory):
    """A tiny GPT-2 MuranoModel with attention weights enabled (eager attention)."""
    from murano.model import MuranoModel

    model_dir = tmp_path_factory.mktemp("tiny_gpt2_attn")
    _build_tiny_gpt2(model_dir)
    return MuranoModel(
        str(model_dir),
        device_map="cpu",
        dtype=torch.float32,
        enable_attention_probs=True,
    )


@pytest.fixture(scope="session")
def murano_model_attn_bf16(tmp_path_factory):
    """A tiny Llama with attention weights enabled, loaded in bfloat16.

    Covers the intervention's dtype handling: the attention weights are the
    model's native dtype, so a float32 replacement must be coerced before the
    blend.
    """
    from murano.model import MuranoModel

    model_dir = tmp_path_factory.mktemp("tiny_model_attn_bf16")
    _build_tiny_local_model(model_dir)
    return MuranoModel(
        str(model_dir),
        device_map="cpu",
        dtype=torch.bfloat16,
        enable_attention_probs=True,
    )


def _build_tiny_mqa_model(path: Path) -> None:
    """Build a tiny multi-query Llama (num_key_value_heads=1) for GQA coverage."""
    _build_word_tokenizer(path)
    config = LlamaConfig(
        vocab_size=len(_VOCAB),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        max_position_embeddings=64,
        pad_token_id=_VOCAB["<pad>"],
        bos_token_id=_VOCAB["<s>"],
        eos_token_id=_VOCAB["</s>"],
    )
    LlamaForCausalLM(config).save_pretrained(path)


@pytest.fixture(scope="session")
def murano_model_attn_mqa(tmp_path_factory):
    """A tiny multi-query Llama with attention weights enabled (n_kv_heads=1)."""
    from murano.model import MuranoModel

    model_dir = tmp_path_factory.mktemp("tiny_model_attn_mqa")
    _build_tiny_mqa_model(model_dir)
    return MuranoModel(
        str(model_dir),
        device_map="cpu",
        dtype=torch.float32,
        enable_attention_probs=True,
    )
