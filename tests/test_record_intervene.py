"""
Unit tests for record_intervene.py functionality.
"""

import pytest
import torch
import numpy as np
from datasets import Dataset
from unittest.mock import Mock

import sys
from pathlib import Path

examples_dir = Path(__file__).parent.parent / "examples"
sys.path.insert(0, str(examples_dir))

from record_intervene import (
    BatchedMuranoModel, 
    compute_steering_vector,
    extract_steering_vector,
    apply_steering_vector,
)
from federico_visualization import Location, ActivationDataset


# ============================================================================
# Helper Classes and Functions
# ============================================================================

class TensorProxy:
    """Mocks nnsight tensor output that supports item assignment."""
    def __init__(self, tensor):
        self._tensor = tensor
    
    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 1 and key[0] == 0:
            return self._tensor
        return self._tensor[key]
    
    def __setitem__(self, key, value):
        self._tensor[key] = value
    
    def clone(self):
        return TensorProxy(self._tensor.clone())
    
    @property
    def shape(self):
        return self._tensor.shape
    
    @property
    def ndim(self):
        return self._tensor.ndim


class RecordTensor:
    """Mocks nnsight tensor that supports slicing and .save() method."""
    def __init__(self, tensor, save_result=None):
        self._tensor = tensor
        self._save_result = save_result or Mock(value=torch.randn(1, 1, 768))
    
    def __getitem__(self, key):
        return RecordTensor(self._tensor[key], self._save_result)
    
    def unsqueeze(self, dim):
        return RecordTensor(self._tensor.unsqueeze(dim), self._save_result)
    
    @property
    def shape(self):
        return self._tensor.shape
    
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
    model.model.tokenizer.batch_decode = Mock(return_value=["decoded text"])
    
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
    model.model.transformer.h[intervene_layer].mlp.output = TensorProxy(torch.randn(batch_size, seq_len, 768))
    model.model.transformer.h[record_layer].mlp.output = RecordTensor(torch.randn(batch_size, seq_len, 768))


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


def create_model_instance(model):
    """Create a BatchedMuranoModel instance from mock model."""
    instance = BatchedMuranoModel.__new__(BatchedMuranoModel)
    instance.model = model.model
    instance.model_name = "gpt2"
    return instance


# ============================================================================
# Tests for record_intervene
# ============================================================================

class TestBatchedMuranoModelRecordIntervene:
    """Test suite for BatchedMuranoModel.record_intervene method."""

    def test_with_tensor_input(self):
        """Test record_intervene accepts tensor input."""
        model = create_mock_model()
        setup_layer_outputs(model, 5, 6)
        instance = create_model_instance(model)
        
        result = instance.record_intervene(
            input=torch.tensor([[1, 2, 3, 4]]),
            intervene_location=Location(layers=[5], modules=["mlp"], token_pos=[-1]),
            record_location=Location(layers=[6], modules=["mlp"], token_pos=[-1]),
            activation_dataset=create_activation_dataset()
        )
        
        assert "activations" in result and "input_ids" in result

    def test_with_dict_input(self):
        """Test record_intervene accepts dict input."""
        model = create_mock_model()
        setup_layer_outputs(model, 5, 6)
        instance = create_model_instance(model)
        
        result = instance.record_intervene(
            input={"input_ids": torch.tensor([[1, 2, 3]])},
            intervene_location=Location(layers=[5], modules=["mlp"], token_pos=[-1]),
            record_location=Location(layers=[6], modules=["mlp"], token_pos=[-1]),
            activation_dataset=create_activation_dataset()
        )
        
        assert "activations" in result

    def test_broadcasts_single_activation(self):
        """Test that single activation is broadcast to batch."""
        model = create_mock_model()
        setup_layer_outputs(model, 5, 6, batch_size=2)
        instance = create_model_instance(model)
        
        result = instance.record_intervene(
            input=torch.tensor([[1, 2, 3], [4, 5, 6]]),
            intervene_location=Location(layers=[5], modules=["mlp"], token_pos=[-1]),
            record_location=Location(layers=[6], modules=["mlp"], token_pos=[-1]),
            activation_dataset=create_activation_dataset(num_examples=1)
        )
        
        assert result["input_ids"].shape[0] == 2

    def test_raises_on_batch_mismatch(self):
        """Test error when batch sizes don't match."""
        model = create_mock_model()
        instance = create_model_instance(model)
        
        with pytest.raises(ValueError, match="Batch size mismatch"):
            instance.record_intervene(
                input=torch.tensor([[1, 2, 3]]),
                intervene_location=Location(layers=[5], modules=["mlp"], token_pos=[-1]),
                record_location=Location(layers=[6], modules=["mlp"], token_pos=[-1]),
                activation_dataset=create_activation_dataset(num_examples=2)
            )


# ============================================================================
# Tests for generate_intervene
# ============================================================================

class TestBatchedMuranoModelGenerateIntervene:
    """Test suite for BatchedMuranoModel.generate_intervene method."""

    def test_basic_generation(self):
        """Test basic generate_intervene functionality."""
        model = create_mock_model()
        raw_model = Mock()
        raw_model.device = torch.device("cpu")
        raw_model.transformer = Mock()
        raw_model.transformer.h = model.model.transformer.h
        raw_model.generate = Mock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
        model.model._model = raw_model
        
        for layer in raw_model.transformer.h:
            layer.mlp.register_forward_hook = Mock(return_value=Mock())
        
        instance = create_model_instance(model)
        
        result = instance.generate_intervene(
            input={"input_ids": torch.tensor([[1, 2, 3]])},
            intervene_location=Location(layers=[5], modules=["mlp"], token_pos=[-1]),
            activation_dataset=create_activation_dataset(),
            max_new_tokens=5
        )
        
        assert all(k in result for k in ["output_ids", "output_text", "input_ids"])
        assert raw_model.generate.called

    def test_with_coefficient(self):
        """Test generate_intervene with custom coefficient."""
        model = create_mock_model()
        raw_model = Mock()
        raw_model.device = torch.device("cpu")
        raw_model.transformer = Mock()
        raw_model.transformer.h = model.model.transformer.h
        raw_model.generate = Mock(return_value=torch.tensor([[1, 2, 3]]))
        model.model._model = raw_model
        
        for layer in raw_model.transformer.h:
            layer.mlp.register_forward_hook = Mock(return_value=Mock())
        
        instance = create_model_instance(model)
        
        result = instance.generate_intervene(
            input={"input_ids": torch.tensor([[1, 2]])},
            intervene_location=Location(layers=[5], modules=["mlp"], token_pos=[-1]),
            activation_dataset=create_activation_dataset(),
            coeff=2.5,
            max_new_tokens=3
        )
        
        assert "output_ids" in result

    def test_removes_hooks_after_generation(self):
        """Test that hooks are properly removed."""
        model = create_mock_model()
        raw_model = Mock()
        raw_model.device = torch.device("cpu")
        raw_model.transformer = Mock()
        raw_model.transformer.h = model.model.transformer.h
        raw_model.generate = Mock(return_value=torch.tensor([[1, 2]]))
        model.model._model = raw_model
        
        # Only register hooks for the layers we'll actually use
        mock_handle = Mock()
        raw_model.transformer.h[5].mlp.register_forward_hook = Mock(return_value=mock_handle)
        
        instance = create_model_instance(model)
        instance.generate_intervene(
            input={"input_ids": torch.tensor([[1]])},
            intervene_location=Location(layers=[5], modules=["mlp"], token_pos=[-1]),
            activation_dataset=create_activation_dataset(),
            max_new_tokens=2
        )
        
        mock_handle.remove.assert_called()


# ============================================================================
# Tests for extract_steering_vector
# ============================================================================

class TestExtractSteeringVector:
    """Test suite for extract_steering_vector function."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for steering vector extraction."""
        model = Mock()
        model.model_name = "gpt2"
        model.model = Mock()
        model.model.tokenizer = Mock()
        
        mock_param = Mock()
        mock_param.device = torch.device("cpu")
        def parameters_generator():
            while True:
                yield mock_param
        model.model.parameters = parameters_generator
        
        def mock_run_task(dataset, location, **kwargs):
            return {
                "activations": torch.randn(len(dataset), 1, 1, 1, 768),
                "global_metadata": {"model_name": "gpt2"},
                "dataset": dataset
            }
        
        model.run_task = Mock(side_effect=mock_run_task)
        return model

    def test_basic_extraction(self, mock_model):
        """Test basic steering vector extraction."""
        pos_dataset = Dataset.from_list([{"text": "I love this!"}, {"text": "Amazing!"}])
        neg_dataset = Dataset.from_list([{"text": "I hate this!"}, {"text": "Terrible!"}])
        
        result = extract_steering_vector(
            model=mock_model,
            positive_dataset=pos_dataset,
            negative_dataset=neg_dataset,
            location=Location(layers=[8], modules=["mlp"], token_pos=[-1])
        )
        
        assert all(k in result for k in ["steering_vector", "positive_activations", "negative_activations", "location"])
        assert isinstance(result["steering_vector"], torch.Tensor)
        assert mock_model.run_task.call_count == 2

    def test_passes_batch_size(self, mock_model):
        """Test that batch_size kwarg is passed to run_task."""
        pos_dataset = Dataset.from_list([{"text": "positive"}])
        neg_dataset = Dataset.from_list([{"text": "negative"}])
        
        extract_steering_vector(
            model=mock_model,
            positive_dataset=pos_dataset,
            negative_dataset=neg_dataset,
            location=Location(layers=[8], modules=["mlp"], token_pos=[-1]),
            batch_size=4
        )
        
        for call in mock_model.run_task.call_args_list:
            assert call.kwargs.get("batch_size") == 4


# ============================================================================
# Tests for apply_steering_vector
# ============================================================================

class TestApplySteeringVector:
    """Test suite for apply_steering_vector function."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for apply_steering_vector tests."""
        model = create_mock_model()
        raw_model = Mock()
        raw_model.device = torch.device("cpu")
        raw_model.transformer = Mock()
        raw_model.transformer.h = model.model.transformer.h
        raw_model.generate = Mock(return_value=torch.tensor([[1, 2, 3, 4]]))
        model.model._model = raw_model
        
        for layer in raw_model.transformer.h:
            layer.mlp.register_forward_hook = Mock(return_value=Mock())
        
        instance = create_model_instance(model)
        mock_param = Mock()
        mock_param.device = torch.device("cpu")
        def parameters_generator():
            while True:
                yield mock_param
        instance.model.parameters = parameters_generator
        
        return instance, model

    def test_generate_only(self, mock_model):
        """apply_steering_vector should generate text (no mode flag)."""
        instance, _ = mock_model
        steering_vector = torch.randn(1, 1, 1, 768)
        intervene_location = Location(layers=[8], modules=["mlp"], token_pos=[-1])
        
        result = apply_steering_vector(
            model=instance,
            input={"input_ids": torch.tensor([[1, 2, 3]])},
            steering_vector=steering_vector,
            intervene_location=intervene_location,
            max_new_tokens=3,
        )
        
        assert isinstance(result, dict)
        for key in ["output_ids", "output_text", "input_ids"]:
            assert key in result


    def test_with_dataset_input(self, mock_model):
        """Test apply_steering_vector processes Dataset input."""
        instance, _ = mock_model
        test_dataset = Dataset.from_list([{"text": "First"}, {"text": "Second"}])
        
        # Mock generate_intervene to return proper dict for each call
        call_count = [0]
        def mock_generate_intervene(*args, **kwargs):
            call_count[0] += 1
            return {
                "output_ids": torch.tensor([[1, 2, 3]]),
                "output_text": [f"Generated {call_count[0]}"],
                "input_ids": torch.tensor([[1]])
            }
        instance.generate_intervene = Mock(side_effect=mock_generate_intervene)
        
        # Mock tokenizer to handle string inputs
        def mock_tokenizer_func(text, **kwargs):
            return {
                "input_ids": torch.tensor([[1]]),
                "attention_mask": torch.tensor([[1]])
            }
        instance.model.tokenizer = Mock(side_effect=mock_tokenizer_func)
        instance.model.tokenizer.pad_token_id = 0
        instance.model.tokenizer.eos_token_id = 1
        
        results = apply_steering_vector(
            model=instance,
            input=test_dataset,
            steering_vector=torch.randn(1, 1, 1, 768),
            intervene_location=Location(layers=[8], modules=["mlp"], token_pos=[-1]),
            max_new_tokens=3
        )
        
        assert isinstance(results, list) and len(results) == 2
        assert all("output_ids" in r for r in results)


# ============================================================================
# Tests for compute_steering_vector
# ============================================================================

class TestComputeSteeringVector:
    """Test suite for compute_steering_vector convenience function."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for compute_steering_vector tests."""
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
        
        def mock_run_task(dataset, location, **kwargs):
            return {
                "activations": torch.randn(len(dataset), 1, 1, 1, 768),
                "global_metadata": {"model_name": "gpt2"},
                "dataset": dataset
            }
        
        model.run_task = Mock(side_effect=mock_run_task)
        return model

    @pytest.fixture
    def datasets(self):
        """Create test datasets."""
        return {
            "positive": Dataset.from_list([{"text": "I love this!"}, {"text": "Amazing!"}]),
            "negative": Dataset.from_list([{"text": "I hate this!"}, {"text": "Terrible!"}]),
            "test": Dataset.from_list([{"text": "The food was"}, {"text": "I think that"}])
        }

    def test_basic_computation(self, mock_model, datasets):
        """Test basic steering vector computation without test dataset."""
        result = compute_steering_vector(
            model=mock_model,
            positive_dataset=datasets["positive"],
            negative_dataset=datasets["negative"],
            location=Location(layers=[8], modules=["mlp"], token_pos=[-1])
        )
        
        assert all(k in result for k in ["steering_vector", "positive_activations", "negative_activations"])
        assert "baseline_output_ids" not in result

    def test_with_test_dataset(self, mock_model, datasets):
        """Test steering vector computation with test dataset."""
        mock_raw_model = Mock()
        mock_raw_model.device = torch.device("cpu")
        mock_raw_model.transformer = Mock()
        mock_raw_model.transformer.h = [Mock() for _ in range(12)]
        for layer in mock_raw_model.transformer.h:
            layer.mlp = Mock()
            layer.mlp.register_forward_hook = Mock(return_value=Mock())
        mock_raw_model.generate = Mock(return_value=torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]))
        mock_model.model._model = mock_raw_model
        
        def mock_tokenizer(texts, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2], [3, 4]]),
                "attention_mask": torch.tensor([[1, 1], [1, 1]])
            }
        mock_model.model.tokenizer = Mock(side_effect=mock_tokenizer)
        mock_model.model.tokenizer.pad_token_id = 0
        mock_model.model.tokenizer.eos_token_id = 1
        mock_model.model.tokenizer.batch_decode = Mock(return_value=["Text 1", "Text 2"])
        
        # Create a BatchedMuranoModel instance with mocked methods
        instance = BatchedMuranoModel.__new__(BatchedMuranoModel)
        instance.model = mock_model.model
        instance.model_name = "gpt2"
        instance.generate_intervene = Mock(return_value={
            "output_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
            "output_text": ["Text 1", "Text 2"],
            "input_ids": torch.tensor([[1, 2], [3, 4]])
        })
        
        # Mock extract_steering_vector to avoid calling run_task which uses trace()
        from unittest.mock import patch
        with patch('record_intervene.extract_steering_vector') as mock_extract:
            mock_extract.return_value = {
                "steering_vector": torch.randn(1, 1, 1, 768),
                "positive_activations": ActivationDataset(
                    activations=np.random.randn(2, 1, 1, 1, 768).astype(np.float32),
                    location=Location(layers=[8], modules=["mlp"], token_pos=[-1]),
                    global_metadata={},
                    dataset=None
                ),
                "negative_activations": ActivationDataset(
                    activations=np.random.randn(2, 1, 1, 1, 768).astype(np.float32),
                    location=Location(layers=[8], modules=["mlp"], token_pos=[-1]),
                    global_metadata={},
                    dataset=None
                ),
                "location": Location(layers=[8], modules=["mlp"], token_pos=[-1])
            }
            
            result = compute_steering_vector(
                model=instance,
                positive_dataset=datasets["positive"],
                negative_dataset=datasets["negative"],
                location=Location(layers=[8], modules=["mlp"], token_pos=[-1]),
                test_dataset=datasets["test"],
                max_new_tokens=10
            )
        
        # Verify result is a dict and has required keys
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        required_keys = ["steering_vector", "baseline_output_ids", "steered_output_ids"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_passes_batch_size(self, mock_model, datasets):
        """Test that batch_size kwarg is passed correctly."""
        compute_steering_vector(
            model=mock_model,
            positive_dataset=datasets["positive"],
            negative_dataset=datasets["negative"],
            location=Location(layers=[8], modules=["mlp"], token_pos=[-1]),
            batch_size=8
        )
        
        assert mock_model.run_task.call_count == 2
        for call in mock_model.run_task.call_args_list:
            assert call.kwargs.get("batch_size") == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
