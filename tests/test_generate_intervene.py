"""
Unit tests for MuranoModel.generate_intervene functionality using a real loaded model.
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
# Tests for generate_intervene
# ============================================================================


class TestMuranoModelGenerateIntervene:
    """Test suite for MuranoModel.generate_intervene method."""

    def test_generate_intervene_basic(self, murano_model):
        """Test basic generation with intervention."""
        # Setup
        target_layer = 5
        hidden_dim = murano_model.model.config.hidden_size
        loc_int = Location(layers=[target_layer], modules=["mlp"], token_pos=[-1])

        # Create a dummy dataset
        # Shape: (1 example, 1 layer, 1 module, 1 token, hidden_dim)
        act_dataset = create_dummy_activation_dataset(
            shape=(1, 1, 1, 1, hidden_dim), layer_idx=target_layer
        )

        # Execute
        result = murano_model.generate_intervene(
            input="The quick brown fox",
            intervene_location=loc_int,
            activation_dataset=act_dataset,
            max_new_tokens=5,
            mode="replacement",
        )

        # Assertions
        assert "output_ids" in result
        assert "input_ids" in result

        # Check that we actually generated tokens
        input_len = result["input_ids"].shape[1]
        output_len = result["output_ids"].shape[1]
        assert output_len > input_len

    def test_generate_intervene_steered_output(self, murano_model):
        """
        Functional test: Verify that intervention changes the output.
        We force a massive value into the MLP to guarantee output degradation/change.
        """
        target_layer = 8
        prompt = "I enjoy walking in the"
        loc_int = Location(layers=[target_layer], modules=["mlp"], token_pos=[-1])

        # 1. Baseline generation (no intervention logic helper, just basic check)
        hidden_dim = murano_model.model.config.hidden_size

        # Zero intervention (should be close to baseline)
        act_dataset_zero = create_dummy_activation_dataset(
            shape=(1, 1, 1, 1, hidden_dim), layer_idx=target_layer
        )
        act_dataset_zero.activations[:] = 0

        res_zero = murano_model.generate_intervene(
            input=prompt,
            intervene_location=loc_int,
            activation_dataset=act_dataset_zero,
            max_new_tokens=5,
            mode="addition",  # Adding 0 should do nothing
        )

        # Massive intervention
        act_dataset_huge = create_dummy_activation_dataset(
            shape=(1, 1, 1, 1, hidden_dim), layer_idx=target_layer
        )
        rng = np.random.default_rng(42)
        act_dataset_huge.activations[:] = (
            rng.standard_normal(act_dataset_huge.activations.shape) * 100.0
        )

        res_huge = murano_model.generate_intervene(
            input=prompt,
            intervene_location=loc_int,
            activation_dataset=act_dataset_huge,
            max_new_tokens=5,
            mode="addition",
        )

        # Outputs should differ

        ids_zero = res_zero["output_ids"][0].tolist()
        ids_huge = res_huge["output_ids"][0].tolist()

        # pdb.set_trace()
        assert ids_zero != ids_huge, (
            "Intervention with massive vector did not change output"
        )
