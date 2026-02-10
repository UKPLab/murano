"""
Checks that the model loaded correctly and nnterp structure is valid.
"""

import pytest
import torch


def test_model_structure_standardization(murano_model):
    """
    Test that the loaded model (GPT2) has the expected standardized structure.
    """
    # 1. Check Layer Count
    # GPT2-small usually has 12 layers
    assert hasattr(murano_model.model, "layers")
    assert len(murano_model.model.layers) == 12, "GPT2 should have 12 layers"

    # 2. Check Standardized Component Access (MLP and Attention)
    # We check the first layer to see if .mlp and .self_attn are accessible
    first_layer = murano_model.model.layers[0]

    # Check MLP
    assert hasattr(first_layer, "mlp"), "Layer 0 should have 'mlp' attribute"

    # Check Self Attention
    assert hasattr(first_layer, "self_attn"), (
        "Layer 0 should have 'self_attn' attribute"
    )


def test_input_preprocessing(murano_model):
    """
    Test that the model's tokenizer and input processing work as expected.
    """
    # Basic check that the model instance holds a tokenizer
    assert hasattr(murano_model.model, "tokenizer")
    assert murano_model.model.tokenizer is not None

    text = "Hello world"
    tokens = murano_model.model.tokenizer(text, return_tensors="pt")

    assert "input_ids" in tokens
    assert tokens["input_ids"].shape[1] > 0


def test_device_placement(murano_model):
    """
    Ensure the model parameters are on the expected device.
    """

    param = next(murano_model.model.parameters())
    assert isinstance(param, torch.Tensor)
    # Just checking it's loaded (could be cpu or cuda)
    assert param.device is not None
