"""
Record and Intervene functionality for MuranoModel.

This module provides a standalone function to record activations at specified locations
and intervene by inserting custom activations during a forward pass.
"""

from typing import Union
from murano import MuranoModel
import torch
import sys
sys.path.append('.')
from federico_visualization import Location, ActivationDataset, BatchedMuranoModel
from datasets import Dataset


def record_intervene(
    model: MuranoModel,
    input: Union[str, torch.Tensor, dict], 
    intervene_location: Location,
    record_location: Location,
    activation_dataset: ActivationDataset,
    example_idx: int = 0,
    **kwargs
) -> dict:
    """
    Run the model with tracing enabled to intervene at one location and record 
    activations at another location after the intervention.
    Computes a single forward pass for a batch of inputs with intervention.
    
    Args:
        model: MuranoModel instance to run
        input: Input to the model (string, tensor, or dict with 'input_ids')
        intervene_location: Location where intervention should occur
        record_location: Location where to record activations after intervention
        activation_dataset: ActivationDataset containing activations to use for intervention
        example_idx: Index of the example in activation_dataset to use for intervention
        **kwargs: Additional arguments to pass to the model
        
    Returns:
        dict: Artifact containing recorded activations, input_ids, and output
    """
    activations = []
    if isinstance(input, torch.Tensor):
        input_ids = input
    elif isinstance(input, dict):
        input_ids = input["input_ids"]
    else:
        raise ValueError("Input must be a string, tensor, or dictionary with 'input_ids'.")

    # Extract the activation to use for intervention from the dataset
    intervention_activation = torch.tensor(activation_dataset[intervene_location][example_idx])

    with model.model.trace() as tracer:
        with tracer.invoke(input_ids, max_length=10, **kwargs) as invoker:
            
            layers_list = list(model.model.transformer.h)

            # First, perform intervention at intervene_location
            for layer_idx, layer in enumerate(intervene_location.layers):
                for module_idx, module in enumerate(intervene_location.modules):
                    layer_module = getattr(layers_list[layer], module)
                    output = layer_module.output
                    
                    # Intervene by inserting the activation from the dataset
                    if intervene_location.token_pos[0] is not None:
                        if isinstance(output, tuple):
                            output[0][:, intervene_location.token_pos[0], :] = intervention_activation
                        else:
                            layer_module.output[:, intervene_location.token_pos[0], :] = intervention_activation
                    else:
                        if isinstance(output, tuple):
                            output[0][:] = intervention_activation
                        else:
                            layer_module.output[:] = intervention_activation
            
            # Then, record activations at record_location
            for layer_idx, layer in enumerate(record_location.layers):
                layer_activation = []
                for module_idx, module in enumerate(record_location.modules):
                    layer_module = getattr(layers_list[layer], module)
                    output = layer_module.output
                    
                    # Record activation after intervention
                    if isinstance(output, tuple):
                        hidden_states = output[0][:, record_location.token_pos[0], :] if record_location.token_pos[0] is not None else output[0]
                    else:
                        hidden_states = output[:, record_location.token_pos[0], :] if record_location.token_pos[0] is not None else output

                    module_activation = hidden_states.save()
                    layer_activation.append(module_activation)
                            
                activations.append(layer_activation)

    artifact = {
        "activations": activations,  # nested list of shape: (num_layers, num_modules, batch_size, seq_len, hidden_dim)
        "input_ids": input_ids,  # type: ignore
    }

    return artifact


def test_record_intervene():
    """Simple test function to verify record_intervene works correctly."""
    print("Testing record_intervene functionality...")
    
    # Import needed for dataset creation

    
    # Load model
    model = BatchedMuranoModel.from_pretrained("gpt2")
    tokenizer = model.model.tokenizer
    
    # Create a simple dataset to get activations from
    data = [
        {"text": f"Sample text number {i}.", "label": i % 2}
        for i in range(8)
    ]
    toy_dataset = Dataset.from_list(data)
    
    # First, create an ActivationDataset by recording activations
    print("Step 1: Recording activations from dataset...")
    source_location = Location(layers=[2], modules=["mlp"], token_pos=-1)
    activation_dataset = model.run_task(toy_dataset, source_location, batch_size=4)
    print(f"  Recorded activations shape: {activation_dataset.activations.shape}")
    
    # Now test record_intervene
    print("\nStep 2: Testing intervention...")
    text = "The quick brown fox"
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    
    # Define two locations:
    # 1. Where to intervene (layer 2, mlp)
    intervene_location = Location(layers=[2], modules=["mlp"], token_pos=-1)
    # 2. Where to record after intervention (layer 6, mlp)
    record_location = Location(layers=[6], modules=["mlp"], token_pos=-1)
    
    # Run record_intervene
    artifact = record_intervene(
        model, 
        input_ids, 
        intervene_location,
        record_location,
        activation_dataset,
        example_idx=0  # Use activation from first example
    )
    
    # Print results
    print(f"  Input text: {text}")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Intervention location: {intervene_location}")
    print(f"  Recording location: {record_location}")
    print(f"  Recorded activations shape: {artifact['activations'][0][0].value.shape}")
    
    print("\n✓ Test passed!")


if __name__ == "__main__":
    test_record_intervene()

