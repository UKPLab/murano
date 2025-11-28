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

        intervention_activation = torch.tensor(activation_dataset[intervene_location])
        batch_size = input_ids.shape[0]

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
    coeff: float = 1.0,
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
        coeff: (Optional) Multiplier for the steering vector strength. Defaults to 1.0.
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
    
    # Test on a different dataset: generate text with and without steering
    if test_dataset:
        intervene_location = intervene_location or location
        tokenizer = model.model.tokenizer
        
        # Get raw HuggingFace model and prepare inputs
        raw_model = model.model._model if hasattr(model.model, "_model") else model.model
        device = raw_model.device
        
        test_texts = [ex["text"] for ex in test_dataset]
        inputs = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device) if "attention_mask" in inputs else None
        
        max_new = kwargs.get("max_new_tokens", 20)
        pad_token_id = tokenizer.pad_token_id
        
        # 1. Baseline Generation (using the model.generate method of HF)
        with torch.no_grad():
            baseline_ids = raw_model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                max_new_tokens=max_new, 
                pad_token_id=pad_token_id
            )
        baseline_text = tokenizer.batch_decode(baseline_ids, skip_special_tokens=True)
        
        # 2. Steered Generation (HF + Hook)
        # Prepare steering vector [1, 1, hidden]
        sv = steering_vector
        while sv.ndim > 1: sv = sv.mean(0)
        sv_tensor = (sv * coeff).to(device).view(1, 1, -1)  # Apply coefficient here
        
        def hook_fn(module, args, output):
            # Add steering vector to output
            if isinstance(output, tuple):
                return (output[0] + sv_tensor,) + output[1:]
            return output + sv_tensor

        handles = []
        layers = raw_model.transformer.h
        for l in intervene_location.layers:
            for m in intervene_location.modules:
                if hasattr(layers[l], m):
                    handles.append(getattr(layers[l], m).register_forward_hook(hook_fn))

        try:
            with torch.no_grad():
                steered_ids = raw_model.generate(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    max_new_tokens=max_new, 
                    pad_token_id=pad_token_id
                )
        finally:
            for h in handles: h.remove()
            
        steered_text = tokenizer.batch_decode(steered_ids, skip_special_tokens=True)
        
        result["baseline_output_ids"] = baseline_ids
        result["baseline_output_text"] = baseline_text
        result["steered_output_ids"] = steered_ids
        result["steered_output_text"] = steered_text
        
        print("\n=== Generation Comparison ===")
        for i, prompt in enumerate(test_texts):
            print(f"\nPrompt: \"{prompt}\"")
            print(f"  Baseline: \"{baseline_text[i]}\"")
            print(f"  Steered:  \"{steered_text[i]}\"")
        
    return result


def test_all():
    model = BatchedMuranoModel.from_pretrained("gpt2")
    
    pos_data = [
        "I absolutely love this!", "You are the best friend ever.", "This is wonderful and amazing.",
        "I am so happy to see you.", "What a beautiful day it is.", "I really admire your work.",
        "You are a fantastic person.", "I love you so much.", "This is the best thing ever.",
        "I am grateful for your kindness."
    ]
    
    neg_data = [
        "I absolutely hate this!", "You are the worst enemy ever.", "This is terrible and awful.",
        "I am so angry to see you.", "What a horrible day it is.", "I really despise your work.",
        "You are a terrible person.", "I hate you so much.", "This is the worst thing ever.",
        "I am disgusted by your behavior."
    ]
    
    # Test prompts: Neutral starts that can go either way
    test_data = [
        "You are a",
        "I think that",
        "The food was",
        "My friend is"
    ]
    
    # Compute steering vector at layer 8 MLP (not 6 because 8 is a deeper layer for abstract sentiment)
    loc = Location(layers=[8], modules=["mlp"], token_pos=[-1])
    rec_loc = Location(layers=[10], modules=["mlp"], token_pos=[-1])
    
    print("Computing Love - Hate steering vector...")
    compute_steering_vector(
        model=model,
        positive_dataset=Dataset.from_list([{"text": t} for t in pos_data]),
        negative_dataset=Dataset.from_list([{"text": t} for t in neg_data]),
        location=loc,
        test_dataset=Dataset.from_list([{"text": t} for t in test_data]),
        record_location=rec_loc,
        max_new_tokens=15,
        coeff=3.0
    )

if __name__ == "__main__":
    test_all()
