"""
Unit tests for MuranoModel functionality.
"""

import pytest
import sys
import os
from pathlib import Path
import torch
from datasets import Dataset

# Add src directory to path for imports
_src_path = Path(__file__).parent.parent / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from murano.model import MuranoModel
from murano.utils import Location, ActivationDataset, steering_vector_to_activation_dataset


def test_always_true():
    """Placeholder test."""
    assert True


def test_all():
    """Test the complete flow: extract steering vector and apply it during generation."""
    # Add examples directory to path for sentiment_data
    _examples_path = Path(__file__).parent.parent / "examples"
    if str(_examples_path) not in sys.path:
        sys.path.insert(0, str(_examples_path))
    
    try:
        from sentiment_data import POSITIVE_SAMPLES, NEGATIVE_SAMPLES
    except ImportError:
        pytest.skip("sentiment_data not available")
    
    model = MuranoModel.from_pretrained("gpt2")
    
    test_data = [
        "I think that the food was",
        "The food was",
        "My friend is a"
    ]
    
    # Test case 1: Extract steering vector and apply to last token only
    loc_extract = Location(layers=[8], modules=["mlp"], token_pos=[-1])
    loc_intervene = Location(layers=[8], modules=["mlp"], token_pos=[-1])
    
    print("=== Test 1: Steering vector applied to last token only ===")
    
    # Extract steering vector
    pos_dataset = Dataset.from_list([{"text": t} for t in POSITIVE_SAMPLES])
    neg_dataset = Dataset.from_list([{"text": t} for t in NEGATIVE_SAMPLES])
    
    pos_artifact = model.run_task(pos_dataset, loc_extract, batch_size=1)
    neg_artifact = model.run_task(neg_dataset, loc_extract, batch_size=1)
    
    # Convert to ActivationDataset if needed
    if hasattr(pos_artifact, 'activations') and hasattr(pos_artifact, 'location'):
        pos_acts = pos_artifact
    else:
        # pos_artifact is a dict
        pos_acts = ActivationDataset(
            activations=pos_artifact["activations"],
            location=loc_extract,
            global_metadata=pos_artifact.get("global_metadata", {}),
            dataset=pos_artifact.get("dataset"),
        )
    
    if hasattr(neg_artifact, 'activations') and hasattr(neg_artifact, 'location'):
        neg_acts = neg_artifact
    else:
        # neg_artifact is a dict
        neg_acts = ActivationDataset(
            activations=neg_artifact["activations"],
            location=loc_extract,
            global_metadata=neg_artifact.get("global_metadata", {}),
            dataset=neg_artifact.get("dataset"),
        )
    
    pos_mean = torch.tensor(pos_acts.activations).mean(0)
    neg_mean = torch.tensor(neg_acts.activations).mean(0)
    steering_vector = pos_mean - neg_mean
    
    # Utilities already imported at top
    scaled_sv = steering_vector * 6.0
    sv_dataset = steering_vector_to_activation_dataset(scaled_sv, loc_intervene)
    
    # Generate with intervention
    test_dataset = Dataset.from_list([{"text": t} for t in test_data])
    tokenizer = model.model.tokenizer
    raw_model = model.model._model if hasattr(model.model, "_model") else model.model
    test_texts = [ex["text"] for ex in test_dataset]
    inputs = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(raw_model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(raw_model.device)
    
    # Baseline generation
    with torch.no_grad():
        baseline_ids = raw_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=15,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
    baseline_text = tokenizer.batch_decode(baseline_ids, skip_special_tokens=True)
    
    # Steered generation
    input_dict = {"input_ids": input_ids}
    if attention_mask is not None:
        input_dict["attention_mask"] = attention_mask
    
    steered_result = model.generate_intervene(
        input=input_dict,
        intervene_location=loc_intervene,
        activation_dataset=sv_dataset,
        max_new_tokens=15,
    )
    
    steered_output_text = tokenizer.batch_decode(steered_result["output_ids"], skip_special_tokens=True)
    
    print("\n=== Generation Comparison ===")
    for i, prompt in enumerate(test_texts):
        print(f"\nPrompt: \"{prompt}\"")
        print(f"  Baseline: \"{baseline_text[i]}\"")
        print(f"  Steered:  \"{steered_output_text[i]}\"")
    
    # Test case 2: Apply to all tokens
    print("\n=== Test 2: Steering vector applied to ALL tokens ===")
    loc_intervene_all = Location(layers=[8], modules=["mlp"], token_pos=None)
    sv_dataset_all = steering_vector_to_activation_dataset(scaled_sv, loc_intervene_all)
    
    steered_result_all = model.generate_intervene(
        input=input_dict,
        intervene_location=loc_intervene_all,
        activation_dataset=sv_dataset_all,
        max_new_tokens=15,
    )
    
    steered_output_text_all = tokenizer.batch_decode(steered_result_all["output_ids"], skip_special_tokens=True)
    
    print("\n=== Generation Comparison (All Tokens) ===")
    for i, prompt in enumerate(test_texts):
        print(f"\nPrompt: \"{prompt}\"")
        print(f"  Baseline: \"{baseline_text[i]}\"")
        print(f"  Steered:  \"{steered_output_text_all[i]}\"")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
