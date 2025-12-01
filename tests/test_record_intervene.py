"""
Unit tests for record_intervene.py functionality.
"""

import pytest
import torch
import numpy as np
from datasets import Dataset
from unittest.mock import Mock

# Import the classes and functions to test
import sys
from pathlib import Path

# Add examples directory to path for imports
examples_dir = Path(__file__).parent.parent / "examples"
sys.path.insert(0, str(examples_dir))

from record_intervene import BatchedMuranoModel, compute_steering_vector
from federico_visualization import Location, ActivationDataset


# Helper classes for mocking nnsight behavior
class TensorProxy:
    """Mocks nnsight tensor output that supports item assignment."""
    def __init__(self, tensor):
        self._tensor = tensor
    
    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 1 and key[0] == 0:
            return self._tensor  # Handle output[0] for tuple outputs
        return self._tensor[key]
    
    def __setitem__(self, key, value):
        self._tensor[key] = value


class RecordTensor:
    """Mocks nnsight tensor that supports slicing and .save() method."""
    def __init__(self, tensor, save_result=None):
        self._tensor = tensor
        self._save_result = save_result or Mock(value=torch.randn(1, 1, 768))
    
    def __getitem__(self, key):
        return RecordTensor(self._tensor[key], self._save_result)
    
    def unsqueeze(self, dim):
        return RecordTensor(self._tensor.unsqueeze(dim), self._save_result)
    
    def save(self):
        return self._save_result


def create_mock_model():
    """Create a simplified mock model with basic structure."""
    model = Mock()
    model.model_name = "gpt2"
    model.model = Mock()
    model.model.tokenizer = Mock()
    
    # Create 12 transformer layers
    layers = []
    for i in range(12):
        layer = Mock()
        layer.mlp = Mock()
        layer.attn = Mock()
        layers.append(layer)
    
    model.model.transformer = Mock()
    model.model.transformer.h = layers
    
    # Mock trace context manager
    mock_tracer = Mock()
    mock_invoke = Mock()
    mock_tracer.__enter__ = Mock(return_value=mock_tracer)
    mock_tracer.__exit__ = Mock(return_value=None)
    mock_invoke.__enter__ = Mock(return_value=mock_invoke)
    mock_invoke.__exit__ = Mock(return_value=None)
    model.model.trace = Mock(return_value=mock_tracer)
    mock_tracer.invoke = Mock(return_value=mock_invoke)
    
    return model


def setup_layer_outputs(model, intervene_layer, record_layer, batch_size=1, seq_len=4):
    """Helper to set up mock layer outputs for intervention and recording."""
    # Setup intervention layer
    intervene_mlp = model.model.transformer.h[intervene_layer].mlp
    intervene_mlp.output = TensorProxy(torch.randn(batch_size, seq_len, 768))
    
    # Setup recording layer
    record_mlp = model.model.transformer.h[record_layer].mlp
    record_mlp.output = RecordTensor(torch.randn(batch_size, seq_len, 768))


def create_activation_dataset(num_examples=1, layers=[5], modules=["mlp"], token_pos=[-1]):
    """Helper to create an ActivationDataset for testing."""
    activations = np.random.randn(num_examples, len(layers), len(modules), len(token_pos), 768).astype(np.float32)
    location = Location(layers=layers, modules=modules, token_pos=token_pos)
    return ActivationDataset(
        activations=activations,
        location=location,
        global_metadata={"model_name": "gpt2"},
        dataset=Dataset.from_list([{"text": f"test{i}"} for i in range(num_examples)])
    )


class TestBatchedMuranoModelRecordIntervene:
    """Test suite for BatchedMuranoModel.record_intervene method."""

    def test_with_tensor_input(self):
        """Test record_intervene accepts tensor input."""
        model = create_mock_model()
        setup_layer_outputs(model, intervene_layer=5, record_layer=6, batch_size=1, seq_len=4)
        
        model_instance = BatchedMuranoModel.__new__(BatchedMuranoModel)
        model_instance.model = model.model
        model_instance.model_name = "gpt2"
        
        input_ids = torch.tensor([[1, 2, 3, 4]])
        activation_dataset = create_activation_dataset()
        
        result = model_instance.record_intervene(
            input=input_ids,
            intervene_location=Location(layers=[5], modules=["mlp"], token_pos=[-1]),
            record_location=Location(layers=[6], modules=["mlp"], token_pos=[-1]),
            activation_dataset=activation_dataset
        )
        
        assert "activations" in result
        assert "input_ids" in result
        assert torch.equal(result["input_ids"], input_ids)

    def test_with_dict_input(self):
        """Test record_intervene accepts dict input with input_ids."""
        model = create_mock_model()
        setup_layer_outputs(model, intervene_layer=5, record_layer=6, batch_size=1, seq_len=3)
        
        model_instance = BatchedMuranoModel.__new__(BatchedMuranoModel)
        model_instance.model = model.model
        model_instance.model_name = "gpt2"
        
        input_dict = {"input_ids": torch.tensor([[1, 2, 3]])}
        activation_dataset = create_activation_dataset()
        
        result = model_instance.record_intervene(
            input=input_dict,
            intervene_location=Location(layers=[5], modules=["mlp"], token_pos=[-1]),
            record_location=Location(layers=[6], modules=["mlp"], token_pos=[-1]),
            activation_dataset=activation_dataset
        )
        
        assert "activations" in result
        assert torch.equal(result["input_ids"], input_dict["input_ids"])

    def test_invalid_input_type(self):
        """Test record_intervene raises error for invalid input."""
        model = create_mock_model()
        model_instance = BatchedMuranoModel.__new__(BatchedMuranoModel)
        model_instance.model = model.model
        model_instance.model_name = "gpt2"
        
        activation_dataset = create_activation_dataset()
        
        with pytest.raises(ValueError, match="Input must be a tensor or dict"):
            model_instance.record_intervene(
                input="invalid string input",
                intervene_location=Location(layers=[5], modules=["mlp"], token_pos=[-1]),
                record_location=Location(layers=[6], modules=["mlp"], token_pos=[-1]),
                activation_dataset=activation_dataset
            )

    def test_batch_size_mismatch(self):
        """Test record_intervene raises error when batch sizes don't match."""
        model = create_mock_model()
        model_instance = BatchedMuranoModel.__new__(BatchedMuranoModel)
        model_instance.model = model.model
        model_instance.model_name = "gpt2"
        
        # Dataset with 2 examples, but input has 1
        activation_dataset = create_activation_dataset(num_examples=2)
        input_ids = torch.tensor([[1, 2, 3]])  # Batch size 1
        
        with pytest.raises(ValueError, match="Batch size mismatch"):
            model_instance.record_intervene(
                input=input_ids,
                intervene_location=Location(layers=[5], modules=["mlp"], token_pos=[-1]),
                record_location=Location(layers=[6], modules=["mlp"], token_pos=[-1]),
                activation_dataset=activation_dataset
            )

    def test_broadcast_single_activation(self):
        """Test record_intervene broadcasts single activation to batch."""
        model = create_mock_model()
        setup_layer_outputs(model, intervene_layer=5, record_layer=6, batch_size=2, seq_len=3)
        
        model_instance = BatchedMuranoModel.__new__(BatchedMuranoModel)
        model_instance.model = model.model
        model_instance.model_name = "gpt2"
        
        # Single activation that should be broadcast to batch size 2
        activation_dataset = create_activation_dataset(num_examples=1)
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])  # Batch size 2
        
        result = model_instance.record_intervene(
            input=input_ids,
            intervene_location=Location(layers=[5], modules=["mlp"], token_pos=[-1]),
            record_location=Location(layers=[6], modules=["mlp"], token_pos=[-1]),
            activation_dataset=activation_dataset
        )
        
        assert "activations" in result
        assert result["input_ids"].shape[0] == 2


class TestComputeSteeringVector:
    """Test suite for compute_steering_vector function."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for steering vector tests."""
        model = Mock()
        model.model_name = "gpt2"
        model.model = Mock()
        model.model.tokenizer = Mock()
        model.model.tokenizer.pad_token_id = 0
        
        def mock_run_task(dataset, location, **kwargs):
            num_examples = len(dataset)
            activations = torch.randn(num_examples, 1, 1, 1, 768)
            return {
                "activations": activations,
                "global_metadata": {"model_name": "gpt2"},
                "dataset": dataset
            }
        
        model.run_task = Mock(side_effect=mock_run_task)
        return model

    @pytest.fixture
    def positive_dataset(self):
        """Positive sentiment examples."""
        return Dataset.from_list([
            {"text": "I love this!"},
            {"text": "This is amazing!"},
            {"text": "Wonderful experience!"}
        ])

    @pytest.fixture
    def negative_dataset(self):
        """Negative sentiment examples."""
        return Dataset.from_list([
            {"text": "I hate this!"},
            {"text": "This is terrible!"},
            {"text": "Awful experience!"}
        ])

    @pytest.fixture
    def test_dataset(self):
        """Test examples for generation."""
        return Dataset.from_list([
            {"text": "The food was"},
            {"text": "I think that"}
        ])

    def test_basic_computation(self, mock_model, positive_dataset, negative_dataset):
        """Test basic steering vector computation."""
        location = Location(layers=[8], modules=["mlp"], token_pos=[-1])
        
        result = compute_steering_vector(
            model=mock_model,
            positive_dataset=positive_dataset,
            negative_dataset=negative_dataset,
            location=location
        )
        
        assert "steering_vector" in result
        assert "positive_activations" in result
        assert "negative_activations" in result
        assert isinstance(result["steering_vector"], torch.Tensor)
        assert isinstance(result["positive_activations"], ActivationDataset)
        assert isinstance(result["negative_activations"], ActivationDataset)
        assert "baseline_output_ids" not in result

    def test_with_test_dataset(self, mock_model, positive_dataset, negative_dataset, test_dataset):
        """Test steering vector computation with test dataset (generation)."""
        location = Location(layers=[8], modules=["mlp"], token_pos=[-1])
        
        # Mock the raw model for generation
        mock_raw_model = Mock()
        mock_raw_model.device = torch.device("cpu")
        mock_raw_model.transformer = Mock()
        mock_raw_model.transformer.h = [Mock() for _ in range(12)]
        for layer in mock_raw_model.transformer.h:
            layer.mlp = Mock()
        
        mock_raw_model.generate = Mock(return_value=torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]))
        mock_model.model._model = mock_raw_model
        
        def mock_tokenizer(texts, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2], [3, 4]]),
                "attention_mask": torch.tensor([[1, 1], [1, 1]])
            }
        mock_model.model.tokenizer = Mock(side_effect=mock_tokenizer)
        mock_model.model.tokenizer.pad_token_id = 0
        mock_model.model.tokenizer.batch_decode = Mock(return_value=["Text 1", "Text 2"])
        
        result = compute_steering_vector(
            model=mock_model,
            positive_dataset=positive_dataset,
            negative_dataset=negative_dataset,
            location=location,
            test_dataset=test_dataset,
            max_new_tokens=10
        )
        
        assert "steering_vector" in result
        assert "baseline_output_ids" in result
        assert "steered_output_ids" in result
        assert len(result["baseline_output_text"]) == len(test_dataset)

    def test_custom_coefficient(self, mock_model, positive_dataset, negative_dataset, test_dataset):
        """Test steering vector with custom coefficient."""
        location = Location(layers=[8], modules=["mlp"], token_pos=[-1])
        
        # Mock generation
        mock_raw_model = Mock()
        mock_raw_model.device = torch.device("cpu")
        mock_raw_model.transformer = Mock()
        mock_raw_model.transformer.h = [Mock() for _ in range(12)]
        for layer in mock_raw_model.transformer.h:
            layer.mlp = Mock()
        mock_raw_model.generate = Mock(return_value=torch.tensor([[1, 2], [3, 4]]))
        mock_model.model._model = mock_raw_model
        
        def mock_tokenizer(texts, **kwargs):
            return {"input_ids": torch.tensor([[1], [2]]), "attention_mask": torch.tensor([[1], [1]])}
        mock_model.model.tokenizer = Mock(side_effect=mock_tokenizer)
        mock_model.model.tokenizer.pad_token_id = 0
        mock_model.model.tokenizer.batch_decode = Mock(return_value=["A", "B"])
        
        result = compute_steering_vector(
            model=mock_model,
            positive_dataset=positive_dataset,
            negative_dataset=negative_dataset,
            location=location,
            test_dataset=test_dataset,
            coeff=2.5
        )
        
        assert "steering_vector" in result
        assert "steered_output_ids" in result

    def test_batch_size_kwarg(self, mock_model, positive_dataset, negative_dataset):
        """Test that batch_size kwarg is passed correctly."""
        location = Location(layers=[8], modules=["mlp"], token_pos=[-1])
        
        compute_steering_vector(
            model=mock_model,
            positive_dataset=positive_dataset,
            negative_dataset=negative_dataset,
            location=location,
            batch_size=8
        )
        
        # Check that run_task was called with batch_size
        assert mock_model.run_task.call_count == 2  # Once for pos, once for neg
        for call in mock_model.run_task.call_args_list:
            assert call.kwargs.get("batch_size") == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
