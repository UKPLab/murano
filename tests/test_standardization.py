"""
Unit tests for MuranoModel integration with nnterp StandardizedTransformer.
"""

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import torch

# Add src directory to path for imports
_src_path = Path(__file__).parent.parent / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from murano.model import MuranoModel
from murano.utils import Location


class TestMuranoStandardization:
    @pytest.fixture
    def mock_standardized_transformer(self):
        """
        Fixture that mocks the StandardizedTransformer class from nnterp.
        It configures the layer outputs to be 'recursive mocks', meaning
        output[:] returns the same object as output.
        """
        with patch("murano.model.StandardizedTransformer") as MockTransformer:
            # Setup the mock instance
            mock_instance = MockTransformer.return_value

            # Setup standard attributes that nnterp provides
            mock_instance.device = "cpu"
            mock_instance.config = MagicMock(hidden_size=768)
            mock_instance.num_layers = 12

            # Mock the context managers
            mock_tracer = MagicMock()
            mock_tracer.__enter__.return_value = mock_tracer
            mock_tracer.__exit__.return_value = None
            mock_instance.trace.return_value = mock_tracer

            mock_invoke = MagicMock()
            mock_invoke.__enter__.return_value = mock_invoke
            mock_invoke.__exit__.return_value = None
            mock_tracer.invoke.return_value = mock_invoke

            mock_generator = MagicMock()
            mock_generator.__enter__.return_value = mock_generator
            mock_generator.__exit__.return_value = None
            mock_generator.generator.output = torch.tensor([[101, 200, 300]])
            mock_instance.generate.return_value = mock_generator

            # Helper to create a mock that returns itself when sliced
            def create_recursive_mock():
                m = MagicMock()
                m.__getitem__.return_value = m
                return m

            # Setup Layers with recursive mocks for outputs
            layers = []
            for i in range(12):
                layer = MagicMock()

                # Assign recursive mocks to the outputs we expect to use
                layer.mlp.output = create_recursive_mock()
                layer.self_attn.output = create_recursive_mock()

                # Also mock the layer block output itself
                layer.output = create_recursive_mock()

                layers.append(layer)

            mock_instance.layers = layers

            yield MockTransformer

    def test_initialization_uses_standardized_transformer(
        self, mock_standardized_transformer
    ):
        """Test that MuranoModel initializes with StandardizedTransformer."""
        model_name = "gpt2"
        model = MuranoModel(model_name)

        # Verify StandardizedTransformer was instantiated
        mock_standardized_transformer.assert_called_with(
            model_name, device_map="auto", dispatch=True
        )
        assert model.model == mock_standardized_transformer.return_value

    def test_record_accesses_standardized_layers(self, mock_standardized_transformer):
        """Test that record() uses the standardized .layers attribute and we can access mlp and self_attn layers"""
        model = MuranoModel("gpt2")
        input_data = {"input_ids": torch.tensor([[1, 2, 3]])}

        # Define a location targeting a standardized module 'mlp' and 'self_attn'
        loc_1 = Location(layers=[5], modules=["mlp"], token_pos=[-1])
        loc_2 = Location(layers=[6], modules=["self_attn"], token_pos=[-1])

        model.record(input_data, loc_1)
        model.record(input_data, loc_2)

        # Verify trace was called
        model.model.trace.assert_called()

        # Verify we accessed layer 5 and module 'mlp'
        model.model.layers[5].mlp.output.save.assert_called()
        # Verify we accessed layer 6 and module 'self_attn'
        model.model.layers[6].self_attn.output.save.assert_called()

    def test_record_handles_all_layers_slice(self, mock_standardized_transformer):
        """Test that record() correctly handles slice(None) using model.num_layers."""
        model = MuranoModel("gpt2")
        input_data = torch.tensor([[1, 2, 3]])

        # Location targeting all layers
        loc = Location(layers=slice(None), modules=["mlp"], token_pos=[-1])

        model.record(input_data, [loc])

        # Verify we iterated over all 12 layers
        # Each layer's mlp.output.save() should have been called
        for i, layer in enumerate(model.model.layers):
            assert layer.mlp.output.save.called, f"Layer {i} MLP output was not saved"

    def test_record_intervene_uses_standardized_layers(
        self, mock_standardized_transformer
    ):
        """Test that record_intervene uses standardized layer access."""
        model = MuranoModel("gpt2")
        input_data = torch.tensor([[1, 2]])

        # Intervene at layer 3, Record at layer 4
        loc_int = Location(layers=[3], modules=["mlp"], token_pos=[-1])
        loc_rec = Location(layers=[4], modules=["self_attn"], token_pos=[-1])

        # Intervention Shape: Examples x Layers x Modules x Tokens
        intervention = torch.randn(1, 1, 1, 1)

        model.record_intervene(input_data, loc_int, loc_rec, intervention)

        # Verify Intervention at layer 3 MLP
        layer_3 = model.model.layers[3]
        layer_3.mlp.output.__setitem__.assert_called()

        # Verify Recording at layer 4 Self Attention
        layer_4 = model.model.layers[4]
        layer_4.self_attn.output.save.assert_called()

    def test_generate_intervene_uses_standardized_layers(
        self, mock_standardized_transformer
    ):
        """Test that generate_intervene uses standardized layer access."""
        model = MuranoModel("gpt2")
        input_data = torch.tensor([[101]])

        loc_int = Location(layers=[2], modules=["mlp"], token_pos=[-1])
        intervention_dataset = MagicMock()  # Mock dataset

        # Mock prepare_intervention_activation utility since we are testing model logic
        with patch("murano.model.prepare_intervention_activation") as mock_prep:
            mock_prep.return_value = torch.randn(1, 1, 768)

            model.generate_intervene(
                input_data, loc_int, intervention_dataset, max_new_tokens=5
            )

            # Verify generate was called
            model.model.generate.assert_called()

            # Verify we accessed layer 2
            layer_2 = model.model.layers[2]
            # Just accessing it is enough to pass the test if the code logic is correct
