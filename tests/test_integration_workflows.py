"""
Integration tests for MuranoModel functionality.
End-to-end pipelines (Extract -> Calculate Vector -> Intervene).
"""

import pytest
import sys
import torch
from pathlib import Path
from datasets import Dataset

# Add src directory to path
_src_path = Path(__file__).parent.parent / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from murano.utils import Location, steering_vector_to_activation_dataset


def test_full_steering_pipeline(murano_model):
    """
    Test the complete flow:
    1. Record activations (extract)
    2. Compute steering vector
    3. Generate with intervention
    """
    print("\n=== Integration Test: Steering Pipeline ===")

    # Define locations
    # We will extract from the last token of layer 6 MLP
    loc_extract = Location(layers=[6], modules=["mlp"], token_pos=[-1])

    # 1. Extract Activations
    pos_input = "I love this movie, it is amazing!"
    neg_input = "I hate this movie, it is terrible!"

    # Record positive
    pos_artifact = murano_model.record(pos_input, loc_extract)
    pos_act = pos_artifact["activations"][0][0]  # Layer 0 , Module 0
    if hasattr(pos_act, "value"):
        pos_act = pos_act.value

    # Record negative
    neg_artifact = murano_model.record(neg_input, loc_extract)
    neg_act = neg_artifact["activations"][0][0]
    if hasattr(neg_act, "value"):
        neg_act = neg_act.value

    # 2. Compute Steering Vector (Pos - Neg)
    # Ensure shapes match for subtraction
    assert pos_act.shape == neg_act.shape
    steering_vector = pos_act - neg_act

    # 3. Prepare for Intervention
    # We will intervene at the same layer
    loc_intervene = Location(layers=[6], modules=["mlp"], token_pos=[-1])

    if steering_vector.ndim == 3:
        # (1, 1, hidden) -> (1, 1, 1, 1, hidden)
        sv_reshaped = steering_vector.unsqueeze(1).unsqueeze(1)
    else:
        # Fallback reshape if dimensions differ
        sv_reshaped = steering_vector.view(1, 1, 1, 1, -1)

    sv_dataset = steering_vector_to_activation_dataset(sv_reshaped, loc_intervene)

    # 4. Generate with Intervention
    test_prompt = "The movie was"

    result = murano_model.generate_intervene(
        input=test_prompt,
        intervene_location=loc_intervene,
        activation_dataset=sv_dataset,
        max_new_tokens=10,
        mode="addition",
    )

    # Verify we got output
    assert "output_ids" in result
    generated_text = murano_model.model.tokenizer.decode(
        result["output_ids"][0], skip_special_tokens=True
    )

    print(f"Prompt: {test_prompt}")
    print(f"Generated: {generated_text}")
    assert len(generated_text) > len(test_prompt)
