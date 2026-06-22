"""Sanity checks for cross-architecture model loading via nnterp."""

import pytest
import torch

from murano.nodes import Node


# Architectures confirmed to expose nnterp's standardized .mlp and .self_attn.
# OPT is excluded because nnterp warns it lacks a unified .mlp module. Models are
# built locally from their config classes rather than pulled from the Hub: the
# test checks nnterp's per-architecture standardization, which depends on the
# architecture class, not on any specific checkpoint, and building offline keeps
# CI off the rate-limited Hub. (Llama3 shares the LlamaForCausalLM class, so it
# is covered by the llama case and not duplicated here.)
_CROSS_ARCH_ATTRS = dict(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    max_position_embeddings=64,
    vocab_size=16,
)


def _build_llama(path):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(num_key_value_heads=4, **_CROSS_ARCH_ATTRS)
    LlamaForCausalLM(config).save_pretrained(path)


def _build_gpt2(path):
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(n_embd=32, n_layer=2, n_head=4, n_positions=64, vocab_size=16)
    GPT2LMHeadModel(config).save_pretrained(path)


def _build_mistral(path):
    from transformers import MistralConfig, MistralForCausalLM

    config = MistralConfig(num_key_value_heads=2, **_CROSS_ARCH_ATTRS)
    MistralForCausalLM(config).save_pretrained(path)


def _build_qwen2(path):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(num_key_value_heads=2, **_CROSS_ARCH_ATTRS)
    Qwen2ForCausalLM(config).save_pretrained(path)


_CROSS_ARCH_BUILDERS = [
    pytest.param(_build_llama, id="llama"),
    pytest.param(_build_gpt2, id="gpt2"),
    pytest.param(_build_mistral, id="mistral"),
    pytest.param(_build_qwen2, id="qwen"),
]


def _save_tiny_tokenizer(path):
    """Save a minimal tokenizer next to a locally-built model so it can load."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "hello": 4, "world": 5}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        model_max_length=64,
    ).save_pretrained(path)


def test_model_structure_standardization(murano_model):
    """Verify that the loaded model exposes nnterp's standardized structure."""
    assert hasattr(murano_model._lm, "layers")
    assert len(murano_model._lm.layers) == murano_model.n_layers

    first_layer = murano_model._lm.layers[0]
    assert hasattr(first_layer, "mlp"), "Layer 0 should have 'mlp' attribute"
    assert hasattr(first_layer, "self_attn"), (
        "Layer 0 should have 'self_attn' attribute"
    )


@pytest.mark.parametrize("build_model", _CROSS_ARCH_BUILDERS)
def test_cross_architecture_standardization(build_model, tmp_path):
    """Verify that different architectures resolve to the same standardized components."""
    from murano.model import MuranoModel

    build_model(tmp_path)
    _save_tiny_tokenizer(tmp_path)
    model = MuranoModel(str(tmp_path), device_map="cpu", dtype=torch.float32)

    assert hasattr(model._lm, "layers")
    assert len(model._lm.layers) > 0

    first_layer = model._lm.layers[0]
    assert hasattr(first_layer, "mlp"), "layer 0 should have 'mlp' attribute"
    assert hasattr(first_layer, "self_attn"), (
        "layer 0 should have 'self_attn' attribute"
    )


def test_residual_intervention_on_tuple_output_arch(tmp_path):
    """Steering/ablation generation works when ``layer.output`` is a tuple.

    GPT-2-style blocks return ``(hidden_states, ...)`` tuples, so the
    intervention must edit the hidden-states element rather than treat the whole
    tuple as a tensor. Llama-style blocks (the default test model) return a plain
    tensor, so this path is only exercised on a tuple-output architecture.
    """
    from murano.model import MuranoModel

    _build_gpt2(tmp_path)
    _save_tiny_tokenizer(tmp_path)
    model = MuranoModel(str(tmp_path), device_map="cpu", dtype=torch.float32)

    out = model.generate(
        "hello world",
        ablate={0: torch.randn(model.d_model)},
        layers=[0],
        modules="residual",
        gen_kwargs={"max_new_tokens": 2, "do_sample": False},
    )
    assert isinstance(out, str)


def test_cross_architecture_end_to_end(murano_model):
    """Ensure load -> record -> intervene works with no architecture-specific code.

    This validates that the refactored pipeline is truly architecture-agnostic
    and doesn't crash looking for Llama-specific attributes.
    """
    # 1. Record
    acts = murano_model.record("Hello world", layers=[0], position="last")
    assert len(acts.positive[(0, "residual")]) > 0

    # 2. Find direction
    direction = murano_model.find_direction(
        positive=["I am good"],
        negative=["I am bad"],
        layers=[0],
    )
    assert direction.best_layer == Node.coerce(0)

    # 3. Intervene via ablation
    ablated_text = murano_model.generate(
        "Hello world",
        ablate=direction,
        gen_kwargs={"max_new_tokens": 1, "do_sample": False},
    )
    assert isinstance(ablated_text, str)

    # 4. Intervene via steering
    steered_text = murano_model.generate(
        "Hello world",
        steer=(direction, 1.0),
        gen_kwargs={"max_new_tokens": 1, "do_sample": False},
    )
    assert isinstance(steered_text, str)


def test_input_preprocessing(murano_model):
    """Verify that the model's tokenizer works as expected."""
    assert murano_model.tokenizer is not None

    text = "Hello world"
    tokens = murano_model.tokenizer(text, return_tensors="pt")

    assert "input_ids" in tokens
    assert tokens["input_ids"].shape[1] > 0


def test_device_placement(murano_model):
    """Ensure the underlying model parameters are on an expected device."""
    param = next(murano_model._lm.parameters())
    assert isinstance(param, torch.Tensor)
    assert param.device is not None
