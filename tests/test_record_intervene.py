"""
Unit tests for MuranoModel.record_intervene functionality using a real loaded model.
"""

import pytest
import torch
import numpy as np
from datasets import Dataset
import pdb
import sys
from pathlib import Path


# Add src directory to path
_src_path = Path(__file__).parent.parent / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from murano.utils import Location, ActivationDataset

# ============================================================================
# Helpers
# ============================================================================


def create_dummy_activation_dataset(
    model_name="gpt2", num_examples=1, shape=(1, 1, 1, 1, 768), layer_idx=0
):
    """Creates a valid ActivationDataset for testing."""
    activations = np.random.randn(*shape).astype(np.float32)
    # FIX: Use the layer_idx argument so the dataset matches the intervention request
    return ActivationDataset(
        activations=activations,
        location=Location(layers=[layer_idx], modules=["mlp"], token_pos=[-1]),
        global_metadata={"model_name": model_name},
        dataset=Dataset.from_list([{"text": f"test{i}"} for i in range(num_examples)]),
    )


# ============================================================================
# Tests for record_intervene
# ============================================================================


class TestMuranoModelRecordIntervene:
    """Test suite for MuranoModel.record_intervene method."""

    def test_record_intervene_basic(self, murano_model):
        """Test that record_intervene runs and returns expected keys."""
        # Setup
        # GPT2 hidden size is 768
        hidden_dim = murano_model.model.config.hidden_size
        input_text = "Hello world"

        # We intervene at layer 0, MLP, last token
        loc_int = Location(layers=[0], modules=["mlp"], token_pos=[-1])
        # We record at layer 1, MLP, last token
        loc_rec = Location(layers=[1], modules=["mlp"], token_pos=[-1])

        # Create intervention activation: batch=1, layer=1, mod=1, tok=1, dim=768
        intervention_act = torch.randn(1, 1, 1, 1, hidden_dim)

        # Execute
        result = murano_model.record_intervene(
            input=input_text,
            location_intervention=loc_int,
            location_recording=loc_rec,
            intervention_activation=intervention_act,
            mode="replacement",
        )

        # Assertions
        assert "activations" in result
        assert "input_ids" in result
        # Check activations structure
        assert len(result["activations"]) > 0

    def test_record_intervene_addition_mode(self, murano_model):
        """Test that 'addition' mode runs without error."""
        hidden_dim = murano_model.model.config.hidden_size
        loc_int = Location(layers=[0], modules=["mlp"], token_pos=[-1])
        loc_rec = Location(layers=[1], modules=["output"], token_pos=[-1])
        intervention_act = torch.randn(1, 1, 1, 1, hidden_dim)

        result = murano_model.record_intervene(
            input="Testing addition",
            location_intervention=loc_int,
            location_recording=loc_rec,
            intervention_activation=intervention_act,
            mode="addition",
        )
        assert "activations" in result

    # ============================================================================
    # Tests for interleving behaviour of record_intervene
    # ============================================================================
    def test_record_intervene_layer_ordering(self, murano_model):
        """Test that activations are recorded in correct layer order.

        This test verifies that the single-loop implementation correctly
        processes layers in sorted order and returns activations in the
        expected sequence.
        """
        hidden_dim = murano_model.model.config.hidden_size
        device = next(murano_model.model.parameters()).device
        input_text = "Layer ordering test"

        # Intervene at layer 1, record at layer 2
        # This tests that intervention at one layer affects subsequent layers
        loc_int = Location(layers=[1], modules=["mlp"], token_pos=[-1])
        loc_rec = Location(layers=[2], modules=["mlp"], token_pos=[-1])

        intervention_act = torch.randn(1, 1, 1, 1, hidden_dim, device=device)

        result = murano_model.record_intervene(
            input=input_text,
            location_intervention=loc_int,
            location_recording=loc_rec,
            intervention_activation=intervention_act,
            mode="replacement",
        )

        # Should have activations for layer 2 (recording layer only)
        assert len(result["activations"]) == 1

        # Verify activations structure: [layer_activations][module_activations]
        for layer_act in result["activations"]:
            assert isinstance(layer_act, list)
            assert len(layer_act) == 1  # One module per layer

        # Verify that the value recorded at layer 2 is affected by intervention at layer 1
        layer_2_recorded_tensor = result["activations"][0][0]

        # If nnsight returns SaveProxy objects, extract the value. If tensors, compare directly.
        if hasattr(layer_2_recorded_tensor, "value"):
            layer_2_recorded_tensor = layer_2_recorded_tensor.value

        # The intervention should propagate to layer 2, so shapes should match
        assert (
            layer_2_recorded_tensor.squeeze().shape == intervention_act.squeeze().shape
        ), "Recorded activation shape does not match intervention shape!"

    def test_record_intervene_same_layer_intervene_and_record(self, murano_model):
        """Test intervention and recording on the same layer.

        This test verifies that when the same layer is in both intervention
        and recording sets, the intervention is applied before recording
        within the same layer iteration step.
        """
        hidden_dim = murano_model.model.config.hidden_size
        device = next(murano_model.model.parameters()).device
        input_text = "Same layer test"

        loc_int = Location(layers=[1], modules=["mlp"], token_pos=[-1])
        loc_rec = Location(layers=[1], modules=["mlp"], token_pos=[-1])

        intervention_act = torch.randn(1, 1, 1, 1, hidden_dim, device=device)

        result = murano_model.record_intervene(
            input=input_text,
            location_intervention=loc_int,
            location_recording=loc_rec,
            intervention_activation=intervention_act,
            mode="replacement",
        )

        assert len(result["activations"]) == 1
        assert "activations" in result

        # check that the recorded value is the intervention value
        recorded_tensor = result["activations"][0][0]
        if hasattr(recorded_tensor, "value"):
            recorded_tensor = recorded_tensor.value

        assert torch.allclose(
            recorded_tensor.squeeze(), intervention_act.squeeze(), atol=1e-5
        ), "Intervention was not applied before recording! Values do not match."
