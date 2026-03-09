"""
Tests generate_intervene using mocks.
"""

import pytest
import torch
from unittest.mock import Mock
import sys
from pathlib import Path

# Add src directory to path for imports
_src_path = Path(__file__).parent.parent / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from murano.model import MuranoModel
from murano.utils import Location, ActivationDataset
from datasets import Dataset
import numpy as np

# --- Mock Helpers (Adapted from test_record_intervene.py) ---


class TensorProxy:
    """Mocks nnsight tensor output that supports item assignment."""

    def __init__(self, tensor):
        self._tensor = tensor

    @property
    def output(self):
        return self

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 1 and key[0] == 0:
            return self._tensor
        return self._tensor[key]

    def __setitem__(self, key, value):
        self._tensor[key] = value

    def clone(self):
        return TensorProxy(self._tensor.clone())


class RecordTensor:
    """Mocks nnsight tensor output that supports slicing and .save()."""

    def __init__(self, tensor, save_result=None):
        self._tensor = tensor
        self._save_result = save_result or Mock(value=torch.randn(1, 1, 768))

    def __getitem__(self, key):
        return RecordTensor(self._tensor[key], self._save_result)

    def save(self):
        return self._save_result


def create_mock_model():
    """Create a simplified mock model with basic structure."""
    model = Mock()
    model.model_name = "gpt2"
    model.model = Mock()
    model.model.tokenizer = Mock()
    model.model.tokenizer.pad_token_id = 0
    model.model.tokenizer.eos_token_id = 1

    mock_param = Mock()
    mock_param.device = torch.device("cpu")

    def parameters_generator():
        while True:
            yield mock_param

    model.model.parameters = parameters_generator

    layers = [Mock() for _ in range(12)]
    for layer in layers:
        layer.mlp = Mock()
        layer.attn = Mock()

    model.model.transformer = Mock()
    model.model.transformer.h = layers
    model.model.layers = layers

    # Base nnsight mocking
    mock_tracer = Mock()
    mock_iter_ctx = Mock()

    # Setup tracer context
    mock_tracer.__enter__ = Mock(return_value=mock_tracer)
    mock_tracer.__exit__ = Mock(return_value=None)

    mock_invoke_ctx = Mock()
    mock_invoke_ctx.__enter__ = Mock(return_value=mock_invoke_ctx)
    mock_invoke_ctx.__exit__ = Mock(return_value=None)

    # Setup iter step context (what comes out of tracer.iter[0])
    mock_iter_ctx.__enter__ = Mock(return_value=mock_iter_ctx)
    mock_iter_ctx.__exit__ = Mock(return_value=None)

    # Setup the 'iter' object itself
    mock_iter_obj = Mock()
    mock_iter_obj.__getitem__ = Mock(return_value=mock_iter_ctx)

    model.model.trace = Mock(return_value=mock_tracer)
    mock_tracer.invoke = Mock(return_value=mock_invoke_ctx)
    mock_tracer.iter = mock_iter_obj
    model.model.generate = Mock(return_value=mock_tracer)
    model.model.generator = Mock()
    model.model.generator.output = Mock()
    model.model.generator.output.save = Mock(return_value=torch.tensor([[1, 2, 3]]))

    return model


def create_model_instance(model):
    """Create a MuranoModel instance from mock model."""
    instance = MuranoModel.__new__(MuranoModel)
    instance.model = model.model
    instance.model_name = "gpt2"
    return instance


def create_activation_dataset(
    num_examples=1, layers=[5], modules=["mlp"], token_pos=[-1]
):
    """Helper to create an ActivationDataset for testing."""
    activations = np.random.randn(
        num_examples, len(layers), len(modules), len(token_pos), 768
    ).astype(np.float32)
    location = Location(layers=layers, modules=modules, token_pos=token_pos)
    return ActivationDataset(
        activations=activations,
        location=location,
        global_metadata={"model_name": "gpt2"},
        dataset=Dataset.from_list([{"text": f"test{i}"} for i in range(num_examples)]),
    )


# --- Test Case ---


class TestMuranoModelCustomIntervention:
    """Focused tests for callback-based intervention behavior."""

    def test_record_intervene_custom_function_applies_and_clones(self):
        model = create_mock_model()
        intervene_proxy = TensorProxy(torch.ones(1, 4, 8, dtype=torch.float32))
        record_proxy = RecordTensor(torch.zeros(1, 4, 8, dtype=torch.float32))
        model.model.layers[0].mlp.output = intervene_proxy
        model.model.layers[1].mlp.output = record_proxy

        instance = create_model_instance(model)
        original = intervene_proxy._tensor.clone()
        observed = {}

        def intervention_fn(activation):
            observed["received_ptr"] = activation.data_ptr()
            observed["target_view_ptr"] = intervene_proxy._tensor[:, :, :].data_ptr()
            return activation + 5.0

        result = instance.record_intervene(
            input=torch.tensor([[1, 2, 3, 4]]),
            location_intervention=Location(layers=[0], modules=["mlp"], token_pos=None),
            location_recording=Location(layers=[1], modules=["mlp"], token_pos=None),
            intervention_fn=intervention_fn,
        )

        assert "activations" in result and "input_ids" in result
        assert not torch.equal(intervene_proxy._tensor, original)
        assert torch.allclose(intervene_proxy._tensor, original + 5.0)
        assert observed["received_ptr"] != observed["target_view_ptr"]

    def test_generate_intervene_custom_function_without_activation_dataset(self):
        model = create_mock_model()
        intervene_proxy = TensorProxy(torch.ones(1, 3, 6, dtype=torch.float32))
        model.model.layers[0].mlp.output = intervene_proxy
        instance = create_model_instance(model)

        result = instance.generate_intervene(
            input={"input_ids": torch.tensor([[1, 2, 3]])},
            intervene_location=Location(layers=[0], modules=["mlp"], token_pos=[-1]),
            activation_dataset=None,
            intervention_fn=lambda activation: activation * 3.0,
            max_new_tokens=2,
        )

        assert "output_ids" in result and "input_ids" in result
        assert torch.allclose(
            intervene_proxy._tensor[:, :-1, :],
            torch.ones_like(intervene_proxy._tensor[:, :-1, :]),
        )
        assert torch.allclose(
            intervene_proxy._tensor[:, -1, :],
            torch.full_like(intervene_proxy._tensor[:, -1, :], 3.0),
        )

    def test_generate_intervene_replacement_mode_backward_compatible(self):
        model = create_mock_model()
        intervene_proxy = TensorProxy(torch.ones(1, 3, 768, dtype=torch.float32))
        location = Location(layers=[0], modules=["mlp"], token_pos=[-1])
        model.model.layers[0].mlp.output = intervene_proxy
        instance = create_model_instance(model)

        activation_dataset = create_activation_dataset(
            num_examples=1, layers=[0], modules=["mlp"], token_pos=[-1]
        )

        result = instance.generate_intervene(
            input={"input_ids": torch.tensor([[1, 2, 3]])},
            intervene_location=location,
            activation_dataset=activation_dataset,
            mode="replacement",
            max_new_tokens=2,
        )

        expected = torch.tensor(activation_dataset[location], dtype=torch.float32)
        assert "output_ids" in result and "input_ids" in result
        assert torch.allclose(intervene_proxy._tensor[:, [-1], :], expected)


class TestGenerateInterveneScoped:
    def test_intervention_scoped_to_iter_zero(self):
        """
        Test that:
        1. Intervention is scoped to tracer.iter[0] (first generation step).
        2. The values in the layer output are actually modified.
        """
        model = create_mock_model()

        # --- Setup nnsight structure ---
        mock_tracer = model.model.generate.return_value

        # Mock the 'iter' object on the tracer
        mock_iter_obj = Mock()
        mock_tracer.iter = mock_iter_obj

        # Mock the context manager returned by tracer.iter[0]
        mock_iter_ctx = Mock()
        mock_iter_ctx.__enter__ = Mock(return_value=mock_iter_ctx)
        mock_iter_ctx.__exit__ = Mock(return_value=None)

        # Configure tracer.iter.__getitem__ to return our context manager
        mock_iter_obj.__getitem__ = Mock(return_value=mock_iter_ctx)

        # Setup generator output
        mock_tracer.generator = Mock()
        mock_tracer.generator.output = torch.tensor([[1, 2, 3]])

        instance = create_model_instance(model)

        # --- Setup Data and Layers ---
        input_ids = torch.tensor([[101, 102]])
        batch_size = 1
        hidden_dim = 768
        target_layer = 5

        # Initialize target layer with zeros
        initial_layer_output = torch.zeros(batch_size, 2, hidden_dim)
        mock_proxy = TensorProxy(initial_layer_output)
        model.model.transformer.h[target_layer].mlp.output = mock_proxy

        # Prepare Intervention Data (value = 99.0)
        intervention_val = 99.0
        activations = np.full(
            (1, 1, 1, 1, hidden_dim), intervention_val, dtype=np.float32
        )
        location = Location(layers=[target_layer], modules=["mlp"], token_pos=[-1])

        dataset = ActivationDataset(
            activations=activations,
            location=location,
            global_metadata={"model_name": "gpt2"},
            dataset=Dataset.from_list([{"text": "test"}]),
        )

        # --- Run the Function ---
        instance.generate_intervene(
            input={"input_ids": input_ids},
            intervene_location=location,
            activation_dataset=dataset,
            max_new_tokens=2,
            mode="replacement",
        )

        # --- Assertions ---

        # Verify tracer.iter[0] was accessed
        mock_iter_obj.__getitem__.assert_called_with(slice(0, 1, None))

        # Check if the tensor was actually modified
        modified_tensor = mock_proxy._tensor

        # Check index 1 (should be replaced with 99.0)
        assert torch.all(modified_tensor[:, 1, :] == intervention_val), (
            f"Intervention values not applied. Expected {intervention_val}"
        )

        # Check index 2 (should be replaced with 0)
        assert torch.all(modified_tensor[:, 0, :] == 0), (
            f"Intervention values applied to wrong position. Expected {0}"
        )

        print("Success: Intervention correctly scoped to tracer.iter[0]")
