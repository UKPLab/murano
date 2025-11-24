"""
Record and Intervene functionality for MuranoModel.
"""

from typing import Union
import torch
from datasets import Dataset

from federico_location import BatchedMuranoModel as BaseBatchedMuranoModel
from federico_visualization import ActivationDataset, Location

class BatchedMuranoModel(BaseBatchedMuranoModel):
    """Extension of BatchedMuranoModel with intervention capabilities."""

    def record_intervene(
        self,
        input: Union[str, torch.Tensor, dict],
        intervene_location: Location,
        record_location: Location,
        activation_dataset: ActivationDataset,
    ) -> dict:
        """
        Perform a forward pass with causal intervention and record the resulting activations.

        This method intervenes at `intervene_location` by injecting activations from 
        `activation_dataset`, then records the activations at `record_location`.

        Args:
            input: The input to the model. Can be a string, tensor of input_ids, or 
                a dictionary containing "input_ids".
            intervene_location: The `Location` object specifying where to intervene 
                (layer, module, token position).
            record_location: The `Location` object specifying where to record activations 
                after the intervention.
            activation_dataset: An `ActivationDataset` containing the activations to 
                inject. If batch size > 1, the dataset should contain matching batch 
                activations, or a single activation will be broadcasted.

        Returns:
            dict: A dictionary containing:
                - "activations": A nested list structure containing recorded activations 
                  from `nnsight`.
                - "input_ids": The input tensor used for the forward pass.
        """
        if isinstance(input, torch.Tensor):
            input_ids = input
        elif isinstance(input, dict):
            input_ids = input["input_ids"]
        else:
            raise ValueError("Input must be a tensor or dict with 'input_ids'.")

        # Prepare intervention activation
        intervention_activation = torch.tensor(activation_dataset[intervene_location])
        batch_size = input_ids.shape[0]

        # Ensure activation is [batch, tokens, hidden] (or compatible)
        while intervention_activation.ndim < 2:
            intervention_activation = intervention_activation.unsqueeze(0)
        
        if intervention_activation.shape[0] == 1 and batch_size > 1:
            intervention_activation = intervention_activation.expand(batch_size, -1)
        elif intervention_activation.shape[0] != batch_size:
            raise ValueError(f"Batch size mismatch: {intervention_activation.shape[0]} vs {batch_size}")

        activations = []
        
        with self.model.trace() as tracer:
            with tracer.invoke(input_ids, max_length=10):
                layers_list = list(self.model.transformer.h)

                # Intervene
                for layer in intervene_location.layers:
                    for module in intervene_location.modules:
                        layer_module = getattr(layers_list[layer], module)
                        output = layer_module.output
                        
                        # Handle tuple outputs (e.g. (hidden_states, attentions))
                        target = output[0] if isinstance(output, tuple) else output
                        
                        if intervene_location.token_pos[0] is not None:
                            target[:, intervene_location.token_pos[0], :] = intervention_activation
                        else:
                            target[:] = intervention_activation

                # Record
                for layer in record_location.layers:
                    layer_activation = []
                    for module in record_location.modules:
                        layer_module = getattr(layers_list[layer], module)
                        output = layer_module.output
                        
                        # Handle tuple outputs and slicing
                        source = output[0] if isinstance(output, tuple) else output
                        
                        if record_location.token_pos[0] is not None:
                            # Preserve dimension: [batch, hidden] -> [batch, 1, hidden]
                            hidden_states = source[:, record_location.token_pos[0], :].unsqueeze(1)
                        else:
                            hidden_states = source
                            
                        layer_activation.append(hidden_states.save())
                    activations.append(layer_activation)

        return {"activations": activations, "input_ids": input_ids}

def compute_steering_vector(
    model: BatchedMuranoModel,
    positive_dataset: Dataset,
    negative_dataset: Dataset,
    location: Location,
    test_dataset: Dataset = None,
    intervene_location: Location = None,
    record_location: Location = None,
    **kwargs,
) -> dict:
    """
    Compute a steering vector and optionally apply it to a test dataset.

    The steering vector is computed as the difference between the mean activations 
    of the positive dataset and the mean activations of the negative dataset at 
    the specified `location`.

    Args:
        model: The `BatchedMuranoModel` instance to use for recording and intervention.
        positive_dataset: A `Dataset` of examples representing the target behavior.
        negative_dataset: A `Dataset` of examples representing the opposing behavior.
        location: The `Location` (layer, module, token) where the steering vector 
            should be computed.
        test_dataset: (Optional) A `Dataset` to test the computed steering vector on. 
            If provided, intervention is performed.
        intervene_location: (Optional) The `Location` to inject the steering vector. 
            Defaults to `location`.
        record_location: (Optional) The `Location` to record activations after 
            intervention. Defaults to `location`.
        **kwargs: Additional arguments passed to `run_task` and `record_intervene` 
            (e.g., `batch_size`, `max_length`).

    Returns:
        dict: A dictionary containing:
            - "steering_vector": The computed steering vector (torch.Tensor).
            - "positive_activations": `ActivationDataset` for the positive examples.
            - "negative_activations": `ActivationDataset` for the negative examples.
            - "baseline_activations": (If test_dataset provided) `ActivationDataset` 
              for test examples before intervention.
            - "intervened_activations": (If test_dataset provided) `ActivationDataset` 
              for test examples after intervention.
    """
    kwargs.setdefault("batch_size", 1)
    
    def record_dataset(dataset, loc):
        artifact = model.run_task(dataset, loc, **kwargs)
        return ActivationDataset(
            activations=artifact["activations"],
            location=loc,
            global_metadata=artifact.get("global_metadata", {}),
            dataset=artifact.get("dataset"),
        )

    pos_acts = record_dataset(positive_dataset, location)
    neg_acts = record_dataset(negative_dataset, location)
    
    steering_vector = torch.tensor(pos_acts.activations).mean(0) - torch.tensor(neg_acts.activations).mean(0)
    
    result = {
        "steering_vector": steering_vector,
        "positive_activations": pos_acts,
        "negative_activations": neg_acts,
    }
    
    # Test on a different dataset
    if test_dataset:
        intervene_location = intervene_location or location
        record_location = record_location or location
        
        # Create dataset for steering vector (batch dim 1 for broadcasting)
        steering_ds = ActivationDataset(
            activations=steering_vector.unsqueeze(0).numpy(),
            location=location,
            global_metadata={"type": "steering_vector"},
            dataset=None,
        )
        
        # Baseline
        baseline_ds = record_dataset(test_dataset, record_location)
        
        # Intervention
        tokenizer = model.model.tokenizer
        test_texts = [ex["text"] for ex in test_dataset]
        tokenized = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True, max_length=kwargs.get("max_length", 10))
        
        intervened = model.record_intervene(
            tokenized["input_ids"], intervene_location, record_location, steering_ds, **kwargs
        )
        
        stacked = model._stack_activations([intervened['activations']])
        intervened_ds = ActivationDataset(
            activations=stacked,
            location=record_location,
            global_metadata={"intervention": "steering_vector"},
            dataset=test_dataset,
        )
        
        result["baseline_activations"] = baseline_ds
        result["intervened_activations"] = intervened_ds
        
        effect = (torch.tensor(intervened_ds.activations).mean() - torch.tensor(baseline_ds.activations).mean()).abs().item()
        
    return result


def test_all():
    model = BatchedMuranoModel.from_pretrained("gpt2")
    
    pos_data = ["I feel happy.", "This is great.", "I am joyful."]
    neg_data = ["I feel sad.", "This is terrible.", "I am miserable."]
    test_data = ["The weather is nice.", "I went to the store."]
    
    # Compute steering vector at layer 6 MLP
    loc = Location(layers=[6], modules=["mlp"], token_pos=[-1])
    rec_loc = Location(layers=[8], modules=["mlp"], token_pos=[-1])
    
    compute_steering_vector(
        model=model,
        positive_dataset=Dataset.from_list([{"text": t} for t in pos_data]),
        negative_dataset=Dataset.from_list([{"text": t} for t in neg_data]),
        location=loc,
        test_dataset=Dataset.from_list([{"text": t} for t in test_data]),
        record_location=rec_loc,
    )

if __name__ == "__main__":
    test_all()
