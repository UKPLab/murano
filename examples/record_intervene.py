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
        example_idx: int = 0,
        **kwargs,
    ) -> dict:
        """
        Run a single forward pass where we intervene at `intervene_location`
        using activations pulled from `activation_dataset` (taken from the
        example indexed by `example_idx`), then record the resulting activations
        at `record_location`. Mirrors `run_recording` but includes the
        intervention step.
        
        Args:
            input: Input sequence(s) to feed the model.
            intervene_location: Where to inject the stored activation.
            record_location: Where to record activations after the injection.
            activation_dataset: Pre-recorded activations to source interventions from.
            example_idx: Which example inside `activation_dataset` to use for the
                injected activation (0-indexed).
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

        intervention_activation = torch.tensor(
            activation_dataset[intervene_location][example_idx]
        )

        activations = []
        with self.model.trace() as tracer:
            with tracer.invoke(input_ids, max_length=10, **kwargs):
                layers_list = list(self.model.transformer.h)

                # Intervene
                for layer in intervene_location.layers:
                    for module in intervene_location.modules:
                        layer_module = getattr(layers_list[layer], module)
                        output = layer_module.output
                        if intervene_location.token_pos[0] is not None:
                            if isinstance(output, tuple):
                                output[0][:, intervene_location.token_pos[0], :] = (
                                    intervention_activation
                                )
                            else:
                                layer_module.output[
                                    :, intervene_location.token_pos[0], :
                                ] = intervention_activation
                        else:
                            if isinstance(output, tuple):
                                output[0][:] = intervention_activation
                            else:
                                layer_module.output[:] = intervention_activation

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
    
    # Step 2: test intervention vs baseline
    print("\nStep 2: Testing intervention...")
    text = "The quick brown fox"
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    
    intervene_location = Location(layers=[2], modules=["mlp"], token_pos=[-1])
    record_location = Location(layers=[6], modules=["mlp"], token_pos=[-1])
    
    baseline = model.run_recording(input_ids, record_location)
    intervened = model.record_intervene(
        input_ids,
        intervene_location,
        record_location,
        activation_dataset,
        example_idx=0,
    )
    
    # Print results
    print(f"  Input text: {text}")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Intervention location: {intervene_location}")
    print(f"  Recording location: {record_location}")
    print(f"  Baseline activations shape:   {baseline['activations'][0][0].value.shape}")
    print(f"  Intervened activations shape: {intervened['activations'][0][0].value.shape}")
    
    before = baseline["activations"][0][0].value
    after = intervened["activations"][0][0].value
    diff = torch.abs(after - before).mean().item()
    print(f"  Mean absolute difference: {diff:.6f}")


if __name__ == "__main__":
    test_record_intervene()