"""Value-level tests for the WeightAblation projection.

The step-level suite (test_steps.py) monkeypatches ``ablate_model_weights`` out,
so the real read/write projection (``W @ P`` for read matrices, ``P @ W`` for
write matrices), the embedding special-case, and the architecture guard are never
exercised. These tests run the real projection on a fresh tiny model and assert
the direction is annihilated on the correct side of each matrix.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("nnsight")
pytest.importorskip("transformers")

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

from murano.model import MuranoModel
from murano.steps.weight_ablation import (
    ProjectionOperator,
    ablate_model_weights,
)

_VOCAB = {
    "<pad>": 0,
    "<s>": 1,
    "</s>": 2,
    "<unk>": 3,
    "hello": 4,
    "world": 5,
    "good": 6,
    "bad": 7,
}


def _save_tokenizer(path: Path) -> None:
    tok = Tokenizer(WordLevel(vocab=dict(_VOCAB), unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        model_max_length=64,
    ).save_pretrained(path)


def _build_tiny_llama(path: Path) -> None:
    _save_tokenizer(path)
    config = LlamaConfig(
        vocab_size=len(_VOCAB),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        pad_token_id=_VOCAB["<pad>"],
        bos_token_id=_VOCAB["<s>"],
        eos_token_id=_VOCAB["</s>"],
    )
    torch.manual_seed(0)
    LlamaForCausalLM(config).save_pretrained(path)


def _build_tiny_gpt2(path: Path) -> None:
    _save_tokenizer(path)
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


@pytest.fixture
def llama(tmp_path):
    # Fresh model per test: ablate_model_weights mutates weights in place.
    path = tmp_path / "llama"
    _build_tiny_llama(path)
    return MuranoModel(str(path), device_map="cpu", dtype=torch.float32)


@pytest.fixture
def gpt2(tmp_path):
    path = tmp_path / "gpt2"
    _build_tiny_gpt2(path)
    return MuranoModel(str(path), device_map="cpu", dtype=torch.float32)


def _unit_direction(d_model: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d_model, generator=g)
    return v / v.norm()


class TestWeightAblationProjection:
    def test_n_modified_counts_every_matrix(self, llama):
        direction = _unit_direction(llama.d_model, seed=1)
        n = ablate_model_weights(llama, ProjectionOperator(direction))
        # embedding (1) + per layer q,k,v,o,gate,up,down (7).
        assert n == 1 + llama.n_layers * 7

    def test_projection_annihilates_direction_on_correct_side(self, llama):
        direction = _unit_direction(llama.d_model, seed=2)
        ablate_model_weights(llama, ProjectionOperator(direction))

        d = direction.to(torch.float32)
        hf = llama.hf_model
        for layer_idx in range(llama.n_layers):
            layer = hf.layers[layer_idx]
            attn, mlp = layer.self_attn, layer.mlp

            # Read matrices (W @ P): remove the direction from the *input* side,
            # so W @ d == 0.
            for read_w in (
                attn.q_proj.weight,
                attn.k_proj.weight,
                attn.v_proj.weight,
                mlp.gate_proj.weight,
                mlp.up_proj.weight,
            ):
                got = read_w.detach().float() @ d
                assert torch.allclose(got, torch.zeros_like(got), atol=1e-5)

            # Write matrices (P @ W): remove the direction from the *output* side,
            # so d @ W == 0.
            for write_w in (attn.o_proj.weight, mlp.down_proj.weight):
                got = d @ write_w.detach().float()
                assert torch.allclose(got, torch.zeros_like(got), atol=1e-5)

        # The embedding writes the residual (W @ P form): each row orthogonal to d.
        got = hf.embed_tokens.weight.detach().float() @ d
        assert torch.allclose(got, torch.zeros_like(got), atol=1e-5)

    def test_other_direction_is_not_annihilated(self, llama):
        """Sanity: the projection is targeted, not a global zeroing."""
        direction = _unit_direction(llama.d_model, seed=3)
        other = _unit_direction(llama.d_model, seed=99)
        ablate_model_weights(llama, ProjectionOperator(direction))

        o_proj = llama.hf_model.layers[0].self_attn.o_proj.weight.detach().float()
        got = other.float() @ o_proj
        assert not torch.allclose(got, torch.zeros_like(got), atol=1e-3)

    def test_unsupported_architecture_raises_before_mutation(self, gpt2):
        """GPT-2 (fused c_attn / Conv1D) is not a Llama layout; must raise."""
        direction = _unit_direction(gpt2.d_model, seed=4)
        # Snapshot a weight to prove nothing was mutated on the failure path.
        before = next(gpt2.hf_model.parameters()).detach().clone()
        with pytest.raises(NotImplementedError, match="Llama-family"):
            ablate_model_weights(gpt2, ProjectionOperator(direction))
        after = next(gpt2.hf_model.parameters()).detach()
        assert torch.equal(before, after)
