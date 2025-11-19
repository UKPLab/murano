"""
Record and Intervene functionality for MuranoModel.

This module mirrors the structure of federico_location.py but focuses on
recording before/after activations when intervening in a model.
"""

from typing import Union

import torch
from datasets import Dataset

from federico_location import (
    BatchedMuranoModel as BaseBatchedMuranoModel,
)
from federico_visualization import ActivationDataset, Location

class BatchedMuranoModel(BaseBatchedMuranoModel):
    """
    Extension of the standard BatchedMuranoModel with intervention helpers.
    """

    def record_intervene(
        self,
        input: Union[str, torch.Tensor, dict],
        intervene_location: Location,
        record_location: Location,
        activation_dataset: ActivationDataset,
        **kwargs,
    ) -> dict:
        """
        Run a single forward pass where we intervene at `intervene_location`
        using activations pulled from `activation_dataset`, then record the 
        resulting activations at `record_location`. Mirrors `run_recording` 
        but includes the intervention step.
        
        All elements in the `activation_dataset` will be applied as the 
        intervention activation. The batch dimension of the activation must 
        match the batch size of the input.
        
        Args:
            input: Input sequence(s) to feed the model.
            intervene_location: Where to inject the stored activation.
            record_location: Where to record activations after the injection.
            activation_dataset: Pre-recorded activations to source interventions from.
                The batch dimension must match the input batch size.
            **kwargs: Extra arguments forwarded to `tracer.invoke`.
                
        Returns:
            dict matching `run_recording`'s structure:
                {
                    "activations": nested list of saved activations,
                    "input_ids": input tensor used in the pass,
                }
        """
        if isinstance(input, torch.Tensor):
            input_ids = input
        elif isinstance(input, dict):
            input_ids = input["input_ids"]
        else:
            raise ValueError("Input must be a tensor or dict with 'input_ids'.")

        # Get intervention activations from the dataset
        intervention_activation = torch.tensor(
            activation_dataset[intervene_location]
        )
        
        # Make sure it's at least 2D: [batch, hidden]
        while intervention_activation.ndim < 2:
            intervention_activation = intervention_activation.unsqueeze(0)
        
        # Check batch size matches input
        batch_size = input_ids.shape[0]
        if intervention_activation.shape[0] == 1 and batch_size > 1:
            # Broadcast single activation to all examples in batch
            intervention_activation = intervention_activation.expand(batch_size, -1)
        elif intervention_activation.shape[0] != batch_size:
            raise ValueError(
                f"Batch size mismatch: activation has {intervention_activation.shape[0]} "
                f"examples but input has {batch_size}"
            )

        activations = []
        with self.model.trace() as tracer:
            with tracer.invoke(input_ids, max_length=10, **kwargs):
                layers_list = list(self.model.transformer.h)

                # Intervene: inject the stored activation
                for layer in intervene_location.layers:
                    for module in intervene_location.modules:
                        layer_module = getattr(layers_list[layer], module)
                        output = layer_module.output
                        
                        # Handle tuple outputs (some modules return (hidden_states, ...))
                        if isinstance(output, tuple):
                            output = output[0]
                        
                        # Inject activation at specific token position or everywhere
                        if intervene_location.token_pos[0] is not None:
                            output[:, intervene_location.token_pos[0], :] = intervention_activation
                        else:
                            output[:] = intervention_activation

                # Record after intervention
                for layer in record_location.layers:
                    layer_activation = []
                    for module in record_location.modules:
                        layer_module = getattr(layers_list[layer], module)
                        output = layer_module.output
                        if isinstance(output, tuple):
                            hidden_states = (
                                output[0][:, record_location.token_pos[0], :]
                                if record_location.token_pos[0] is not None
                                else output[0]
                            )
                        else:
                            hidden_states = (
                                output[:, record_location.token_pos[0], :]
                                if record_location.token_pos[0] is not None
                                else output
                            )
                        module_activation = hidden_states.save()
                        layer_activation.append(module_activation)
                    activations.append(layer_activation)

        return {
            "activations": activations,
            "input_ids": input_ids,
        }

def test_record_intervene():
    """Simple test function to verify record_intervene works correctly."""
    print("Testing record_intervene functionality...")
    
    model = BatchedMuranoModel.from_pretrained("gpt2")
    tokenizer = model.model.tokenizer
    
    data = [
            "Aisha studies late to prepare for her final philosophy exam.",
            "Jamal spends his weekend hiking with friends in the redwood forest.",
            "Lucia bakes fresh sourdough before opening her neighborhood café.",
            "Rafi fixes vintage radios in his garage while listening to smooth jazz.",
            "Mira drafts grant proposals to fund clean water projects abroad.",
            "Theo volunteers at the animal shelter every Saturday morning.",
            "Hana mentors high-school robotics teams after work.",
            "Carlos restores classic bicycles and rides them along the coast.",
        ]
    
    toy_dataset = Dataset.from_list([{"text": text} for text in data])
    
    # First, create an ActivationDataset by recording activations
    print("Step 1: Recording activations from dataset...")
    source_location = Location(layers=[2], modules=["mlp"], token_pos=[-1])
    artifact = model.run_task(toy_dataset, source_location)
    activation_dataset = ActivationDataset(
        activations=artifact["activations"],
        location=source_location,
        global_metadata=artifact.get("global_metadata", {}),
        dataset=artifact.get("dataset"),
    )
    print(f"  Recorded activations shape: {activation_dataset.activations.shape}")
    
    # Step 2: test intervention vs baseline using all activations
    print("\nStep 2: Testing intervention with batch of 8 inputs...")
    
    # Create a batch of test inputs matching the number of activations
    test_texts = [
        "The quick brown fox",
        "Machine learning is fascinating",
        "Climate change affects everyone",
        "Music brings people together",
        "Technology advances rapidly",
        "Ocean waves crash gently",
        "Books open new worlds",
        "Coffee fuels morning productivity"
    ]
    
    # Tokenize all texts into a batch
    tokenized = tokenizer(test_texts, return_tensors="pt", padding=True)
    input_ids = tokenized["input_ids"]
    
    intervene_location = Location(layers=[2], modules=["mlp"], token_pos=[-1])
    record_location = Location(layers=[6], modules=["mlp"], token_pos=[-1])
    
    baseline = model.run_recording(input_ids, record_location)
    intervened = model.record_intervene(
        input_ids,
        intervene_location,
        record_location,
        activation_dataset,  # Use all 8 activations
    )
    
    # Print results
    print(f"  Number of test inputs: {len(test_texts)}")
    print(f"  Input batch shape: {input_ids.shape}")
    print(f"  Intervention location: {intervene_location}")
    print(f"  Recording location: {record_location}")
    print(f"  Baseline activations shape:   {baseline['activations'][0][0].value.shape}")
    print(f"  Intervened activations shape: {intervened['activations'][0][0].value.shape}")
    
    before = baseline["activations"][0][0].value
    after = intervened["activations"][0][0].value
    diff = torch.abs(after - before).mean().item()
    print(f"  Mean absolute difference across all examples: {diff:.6f}")
    
    # Show per-example differences
    print("\n  Per-example differences:")
    for i in range(len(test_texts)):
        example_diff = torch.abs(after[i] - before[i]).mean().item()
        print(f"    Example {i+1}: {example_diff:.6f}")


if __name__ == "__main__":
    test_record_intervene()