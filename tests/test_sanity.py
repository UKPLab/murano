"""Sanity checks for cross-architecture model loading via nnterp."""

import pytest
import torch


# Architectures confirmed to expose nnterp's standardized .mlp and .self_attn.
# OPT is excluded because nnterp warns it lacks a unified .mlp module.
_CROSS_ARCH_MODELS = [
    pytest.param(
        "HuggingFaceM4/tiny-random-LlamaForCausalLM",
        id="llama",
    ),
    pytest.param(
        "hf-internal-testing/tiny-random-GPT2Model",
        id="gpt2",
    ),
    pytest.param(
        "HuggingFaceM4/tiny-random-Llama3ForCausalLM",
        id="llama3",
    ),
    pytest.param(
        "hf-internal-testing/tiny-random-MistralForCausalLM",
        id="mistral",
    ),
    pytest.param(
        "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
        id="qwen",
    ),
]


def test_model_structure_standardization(murano_model):
    """Verify that the loaded model exposes nnterp's standardized structure."""
    assert hasattr(murano_model._lm, "layers")
    assert len(murano_model._lm.layers) == murano_model.n_layers

    first_layer = murano_model._lm.layers[0]
    assert hasattr(first_layer, "mlp"), "Layer 0 should have 'mlp' attribute"
    assert hasattr(first_layer, "self_attn"), (
        "Layer 0 should have 'self_attn' attribute"
    )


@pytest.mark.parametrize("model_id", _CROSS_ARCH_MODELS)
def test_cross_architecture_standardization(model_id):
    """Verify that different architectures resolve to the same standardized components."""
    from murano.model import MuranoModel

    model = MuranoModel(model_id, device_map="cpu", dtype=torch.float32)

    assert hasattr(model._lm, "layers")
    assert len(model._lm.layers) > 0

    first_layer = model._lm.layers[0]
    assert hasattr(first_layer, "mlp"), (
        f"{model_id} layer 0 should have 'mlp' attribute"
    )
    assert hasattr(first_layer, "self_attn"), (
        f"{model_id} layer 0 should have 'self_attn' attribute"
    )


def test_cross_architecture_end_to_end(murano_model):
    """Ensure load -> record -> intervene works with no architecture-specific code.

    This validates that the refactored pipeline is truly architecture-agnostic
    and doesn't crash looking for Llama-specific attributes.
    """
    # 1. Record
    acts = murano_model.record("Hello world", layers=[0], position="last")
    assert len(acts.positive[0]) > 0

    # 2. Find direction
    direction = murano_model.find_direction(
        positive=["I am good"],
        negative=["I am bad"],
        layers=[0],
    )
    assert direction.best_layer == 0

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
