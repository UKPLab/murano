from typing import Union
import torch
import numpy as np
from datasets import Dataset

from federico_location import BatchedMuranoModel as BaseBatchedMuranoModel
from federico_visualization import ActivationDataset, Location
from utils import (
    prepare_input_ids,
    prepare_intervention_activation,
    steering_vector_to_activation_dataset,
)

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
        input_ids, _ = prepare_input_ids(input, self.model.tokenizer, next(self.model.parameters()).device)
        batch_size = input_ids.shape[0]
        intervention_activation = prepare_intervention_activation(
            activation_dataset, intervene_location, batch_size, 
            next(self.model.parameters()).device, coeff=1.0
        )

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
                        
                        if intervene_location.token_pos and len(intervene_location.token_pos) > 0 and intervene_location.token_pos[0] is not None:
                            pos = intervene_location.token_pos[0]
                            if pos < 0:
                                pos = target.shape[1] + pos
                            pos = max(0, min(pos, target.shape[1] - 1))
                            # ADD steering vector to original activation (not replace)
                            target[:, pos, :] = target[:, pos, :] + intervention_activation
                        else:
                            # ADD steering vector to all positions
                            target[:] = target[:] + intervention_activation

                # Record
                for layer in record_location.layers:
                    layer_activation = []
                    for module in record_location.modules:
                        layer_module = getattr(layers_list[layer], module)
                        output = layer_module.output
                        source = output[0] if isinstance(output, tuple) else output
                        
                        if record_location.token_pos and len(record_location.token_pos) > 0 and record_location.token_pos[0] is not None:
                            pos = record_location.token_pos[0]
                            if pos < 0:
                                pos = source.shape[1] + pos
                            pos = max(0, min(pos, source.shape[1] - 1))
                            # Preserve dimension: [batch, hidden] -> [batch, 1, hidden]
                            hidden_states = source[:, pos, :].unsqueeze(1)
                        else:
                            hidden_states = source
                            
                        layer_activation.append(hidden_states.save())
                    activations.append(layer_activation)

        return {"activations": activations, "input_ids": input_ids}

    def generate_intervene(
        self,
        input: Union[str, torch.Tensor, dict],
        intervene_location: Location,
        activation_dataset: ActivationDataset,
        max_new_tokens: int = 20,
        coeff: float = 1.0,
        **generation_kwargs,
    ) -> dict:
        """
        Generate text with intervention applied at specified location.
        
        This method applies intervention during text generation by setting up hooks
        that inject activations at each forward pass step. It uses HuggingFace's
        generate() method internally, overriding the default behavior when intervention
        arguments are provided.
        
        Args:
            input: The input to the model. Can be a string, tensor of input_ids, or 
                a dictionary containing "input_ids".
            intervene_location: The `Location` object specifying where to intervene 
                (layer, module, token position).
            activation_dataset: An `ActivationDataset` containing the activations to 
                inject. If batch size > 1, the dataset should contain matching batch 
                activations, or a single activation will be broadcasted.
            max_new_tokens: Maximum number of new tokens to generate. Defaults to 20.
            coeff: Multiplier for the intervention activations. Defaults to 1.0.
            **generation_kwargs: Additional arguments passed to HuggingFace's generate() 
                method (e.g., temperature, top_p, top_k, do_sample, etc.).
        
        Returns:
            dict: A dictionary containing:
                - "output_ids": The generated token IDs (torch.Tensor).
                - "output_text": The decoded generated text (List[str]).
                - "input_ids": The input tensor used for generation.
        """
        tokenizer = self.model.tokenizer
        raw_model = self.model._model if hasattr(self.model, "_model") else self.model
        device = raw_model.device
        
        input_ids, attention_mask = prepare_input_ids(input, tokenizer, device)
        batch_size = input_ids.shape[0]
        intervention_activation = prepare_intervention_activation(
            activation_dataset, intervene_location, batch_size, device, coeff
        )
        
        # Debug: Check intervention activation shape and magnitude
        # print(f"DEBUG: Intervention activation shape: {intervention_activation.shape}")
        # print(f"DEBUG: Intervention activation norm: {intervention_activation.norm().item():.4f}")
        
        # Set up hooks for intervention
        handles = []
        layers = raw_model.transformer.h
        
        hook_call_count = [0]  # Track if hook is being called
        
        def create_hook(intervention_act, token_pos):
            """Create a hook function that applies intervention at each forward pass."""
            def hook_fn(module, args, output):
                hook_call_count[0] += 1
                output_tensor = (output[0] if isinstance(output, tuple) else output).clone()
                seq_len = output_tensor.shape[1]
                
                # Store original norm for debugging
                orig_norm = output_tensor.norm().item()
                
                # Normalize intervention activation shape to [batch, seq, hidden]
                act = intervention_act
                while act.ndim > 3:
                    act = act.squeeze(0)
                
                if act.ndim == 2:
                    # [batch, hidden] -> broadcast over sequence
                    act_full = act.unsqueeze(1).expand(-1, seq_len, -1)
                elif act.ndim == 3:
                    # [batch, seq, hidden]
                    if act.shape[1] == seq_len:
                        act_full = act
                    elif act.shape[1] == 1:
                        act_full = act.expand(-1, seq_len, -1)
                    else:
                        # Fallback: use first position and broadcast
                        act_full = act[:, :1, :].expand(-1, seq_len, -1)
                else:
                    raise ValueError(f"Unexpected intervention activation shape: {intervention_act.shape}")
                
                # Apply intervention - ADD steering vector to original activation (not replace)
                # Apply at the last token position (where steering vector was extracted from)
                # This ensures the intervention affects the generation of new tokens
                if token_pos and len(token_pos) > 0 and token_pos[0] is not None:
                    pos = token_pos[0] if token_pos[0] >= 0 else seq_len + token_pos[0]
                    pos = max(0, min(pos, seq_len - 1))
                    # Apply steering vector at the last token position
                    output_tensor[:, pos, :] = output_tensor[:, pos, :] + act_full[:, pos, :]
                else:
                    # If no token_pos specified, apply to all positions
                    output_tensor[:] = output_tensor[:] + act_full
                
                # Debug: Check if intervention changed the activation
                # new_norm = output_tensor.norm().item()
                # if hook_call_count[0] <= 3:  # Only print first few calls
                #     print(f"DEBUG: Hook call {hook_call_count[0]}, seq_len={seq_len}, orig_norm={orig_norm:.4f}, new_norm={new_norm:.4f}, change={new_norm-orig_norm:.4f}")
                
                return (output_tensor,) + output[1:] if isinstance(output, tuple) else output_tensor
            return hook_fn
        
        # Register hooks for specified layers and modules
        hook_fn = create_hook(intervention_activation, intervene_location.token_pos)
        for layer_idx in intervene_location.layers:
            for module_name in intervene_location.modules:
                if hasattr(layers[layer_idx], module_name):
                    module = getattr(layers[layer_idx], module_name)
                    handles.append(module.register_forward_hook(hook_fn))
        
        # Generate with intervention
        try:
            with torch.no_grad():
                # Prepare generation kwargs
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    **generation_kwargs
                }
                
                if attention_mask is not None:
                    gen_kwargs["attention_mask"] = attention_mask
                
                # Call HF's generate() - hooks will apply intervention at each step
                output_ids = raw_model.generate(input_ids, **gen_kwargs)
                
                # Debug: Check if hooks were called
                # print(f"DEBUG: Hook was called {hook_call_count[0]} times during generation")
        finally:
            # Remove all hooks
            for handle in handles:
                handle.remove()
        
        # Decode generated text
        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
        return {
            "output_ids": output_ids,
            "output_text": output_text,
            "input_ids": input_ids,
        }


def extract_steering_vector(
    model: BatchedMuranoModel,
    positive_dataset: Dataset,
    negative_dataset: Dataset,
    location: Location,
    **kwargs,
) -> dict:
    """
    Extract a steering vector from positive and negative datasets.
    
    The steering vector is computed as the difference between the mean activations 
    of the positive dataset and the mean activations of the negative dataset at 
    the specified `location`.
    
    Args:
        model: The `BatchedMuranoModel` instance to use for recording.
        positive_dataset: A `Dataset` of examples representing the target behavior.
        negative_dataset: A `Dataset` of examples representing the opposing behavior.
        location: The `Location` (layer, module, token) where the steering vector 
            should be computed.
        **kwargs: Additional arguments passed to `run_task` (e.g., `batch_size`).
    
    Returns:
        dict: A dictionary containing:
            - "steering_vector": The computed steering vector (torch.Tensor).
            - "positive_activations": `ActivationDataset` for the positive examples.
            - "negative_activations": `ActivationDataset` for the negative examples.
            - "location": The `Location` where the steering vector was extracted.
    """
    kwargs.setdefault("batch_size", 1)
    
    # Record activations from both datasets
    pos_artifact = model.run_task(positive_dataset, location, **kwargs)
    neg_artifact = model.run_task(negative_dataset, location, **kwargs)
    
    # Convert to ActivationDataset if needed
    def to_activation_dataset(artifact, loc):
        if isinstance(artifact, ActivationDataset):
            return artifact
        return ActivationDataset(
            activations=artifact["activations"],
            location=loc,
            global_metadata=artifact.get("global_metadata", {}),
            dataset=artifact.get("dataset"),
        )
    
    pos_acts = to_activation_dataset(pos_artifact, location)
    neg_acts = to_activation_dataset(neg_artifact, location)
    
    # Compute steering vector as difference of means
    pos_mean = torch.tensor(pos_acts.activations).mean(0)
    neg_mean = torch.tensor(neg_acts.activations).mean(0)
    steering_vector = pos_mean - neg_mean
    
    # Debug: Check steering vector magnitude
    # print(f"DEBUG: Steering vector shape: {steering_vector.shape}")
    # print(f"DEBUG: Steering vector norm: {steering_vector.norm().item():.4f}")
    # print(f"DEBUG: Positive mean norm: {pos_mean.norm().item():.4f}")
    # print(f"DEBUG: Negative mean norm: {neg_mean.norm().item():.4f}")
    
    return {
        "steering_vector": steering_vector,
        "positive_activations": pos_acts,
        "negative_activations": neg_acts,
        "location": location,
    }


def apply_steering_vector(
    model: BatchedMuranoModel,
    input: Union[str, torch.Tensor, dict, Dataset],
    steering_vector: torch.Tensor,
    intervene_location: Location,
    mode: str = "generate",
    record_location: Location = None,
    coeff: float = 1.0,
    **kwargs,
) -> dict:
    """
    Apply a pre-computed steering vector to input(s).
    
    Args:
        model: The `BatchedMuranoModel` instance to use for intervention.
        input: The input(s) to apply the steering vector to. Can be a string, tensor, 
            dict, or Dataset.
        steering_vector: The steering vector tensor to apply.
        intervene_location: The `Location` to inject the steering vector.
        mode: The mode of operation. Options:
            - "generate": Generate text with intervention (default)
            - "record": Record activations with intervention
            - "both": Both generate and record
        record_location: The `Location` to record activations after intervention. 
            Required if mode is "record" or "both".
        coeff: Multiplier for the steering vector strength. Defaults to 1.0.
        **kwargs: Additional arguments passed to `generate_intervene` or `record_intervene` 
            (e.g., `max_new_tokens`, `batch_size`).
    
    Returns:
        dict: A dictionary containing results based on the mode:
            - If mode="generate": {"output_ids", "output_text", "input_ids"}
            - If mode="record": {"activations", "input_ids"}
            - If mode="both": All of the above
    """
    # Convert steering vector to ActivationDataset
    sv_dataset = steering_vector_to_activation_dataset(steering_vector, intervene_location)
    
    # Handle different input types
    if isinstance(input, Dataset):
        # For Dataset, process each example
        results = []
        for ex in input:
            result = apply_steering_vector(
                model=model,
                input=ex["text"],
                steering_vector=steering_vector,
                intervene_location=intervene_location,
                mode=mode,
                record_location=record_location,
                coeff=coeff,
                **kwargs
            )
            results.append(result)
        return results
    
    # Prepare input_ids for single input
    tokenizer = model.model.tokenizer
    device = next(model.model.parameters()).device
    input_ids, attention_mask = prepare_input_ids(input, tokenizer, device)
    
    result = {}
    
    # Generate with intervention
    if mode in ["generate", "both"]:
        # Prepare input dict with attention_mask if available
        input_dict = {"input_ids": input_ids}
        if attention_mask is not None:
            input_dict["attention_mask"] = attention_mask
        
        gen_result = model.generate_intervene(
            input=input_dict,
            intervene_location=intervene_location,
            activation_dataset=sv_dataset,
            coeff=coeff,
            **{k: v for k, v in kwargs.items() if k not in ["batch_size", "max_length", "attention_mask"]}
        )
        result.update(gen_result)
    
    # Record with intervention
    if mode in ["record", "both"]:
        if record_location is None:
            raise ValueError("record_location must be provided when mode is 'record' or 'both'")
        rec_result = model.record_intervene(
            input=input_ids,
            intervene_location=intervene_location,
            record_location=record_location,
            activation_dataset=sv_dataset,
        )
        result.update(rec_result)
    
    return result


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
    
    This is a convenience function that calls `extract_steering_vector` and optionally
    `apply_steering_vector`. For more control, use those functions directly.
    
    Args:
        model: The `BatchedMuranoModel` instance to use.
        positive_dataset: A `Dataset` of examples representing the target behavior.
        negative_dataset: A `Dataset` of examples representing the opposing behavior.
        location: The `Location` where the steering vector should be computed.
        test_dataset: (Optional) A `Dataset` to test the computed steering vector on. 
            If provided, generation is performed with and without steering.
        intervene_location: (Optional) The `Location` to inject the steering vector. 
            Defaults to `location`.
        record_location: (Optional) The `Location` to record activations after 
            intervention. Defaults to `location`.
        coeff: (Optional) Multiplier for the steering vector strength. Defaults to 1.0.
        **kwargs: Additional arguments passed to underlying functions.
    
    Returns:
        dict: A dictionary containing:
            - "steering_vector": The computed steering vector (torch.Tensor).
            - "positive_activations": `ActivationDataset` for the positive examples.
            - "negative_activations": `ActivationDataset` for the negative examples.
            - "baseline_output_ids/text": (If test_dataset provided) Baseline generation results.
            - "steered_output_ids/text": (If test_dataset provided) Steered generation results.
    """
    # Extract steering vector
    sv_result = extract_steering_vector(
        model=model,
        positive_dataset=positive_dataset,
        negative_dataset=negative_dataset,
        location=location,
        **{k: v for k, v in kwargs.items() if k != "max_new_tokens"}
    )
    
    result = {
        "steering_vector": sv_result["steering_vector"],
        "positive_activations": sv_result["positive_activations"],
        "negative_activations": sv_result["negative_activations"],
    }
    
    # Apply to test dataset if provided
    if test_dataset:
        intervene_location = intervene_location or location
        
        # Get baseline generation (without intervention)
        tokenizer = model.model.tokenizer
        raw_model = model.model._model if hasattr(model.model, "_model") else model.model
        test_texts = [ex["text"] for ex in test_dataset]
        inputs = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(raw_model.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(raw_model.device)
        
        max_new = kwargs.get("max_new_tokens", 20)
        
        with torch.no_grad():
            baseline_ids = raw_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )
        baseline_text = tokenizer.batch_decode(baseline_ids, skip_special_tokens=True)
        
        # Apply steering vector with generation
        # Prepare input dict with attention_mask
        input_dict = {"input_ids": input_ids}
        if attention_mask is not None:
            input_dict["attention_mask"] = attention_mask
        
        steered_result = apply_steering_vector(
            model=model,
            input=input_dict,
            steering_vector=sv_result["steering_vector"],
            intervene_location=intervene_location,
            mode="generate",
            coeff=coeff,
            max_new_tokens=max_new,
        )
        
        result["baseline_output_ids"] = baseline_ids
        result["baseline_output_text"] = baseline_text
        result["steered_output_ids"] = steered_result["output_ids"]
        result["steered_output_text"] = steered_result["output_text"]
        
        print("\n=== Generation Comparison ===")
        for i, prompt in enumerate(test_texts):
            print(f"\nPrompt: \"{prompt}\"")
            print(f"  Baseline: \"{baseline_text[i]}\"")
            print(f"  Steered:  \"{steered_result['output_text'][i]}\"")
    
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
        "I absolutely hate this!", "You are the worst person ever.", "This is terrible and awful.",
        "I am so angry to see you.", "What a horrible day it is.", "I really despise your work.",
        "You are a terrible person.", "I hate you so much.", "This is the worst thing ever.",
        "I am disgusted by your behavior."
    ]
    
    # Test prompts: Negative starts to see if steering makes them more positive
    test_data = [
        "I think that the food was",
        "The food was",
        "My friend is a"
    ]
    
    # Compute steering vector at layer 8 MLP (not 6 because 8 is a deeper layer for abstract sentiment)
    # Extract and apply at same layer to avoid shape issues
    loc = Location(layers=[8], modules=["mlp"], token_pos=[-1])
    rec_loc = Location(layers=[10], modules=["mlp"], token_pos=[-1])
    
    print("Computing Love - Hate steering vector...")
    compute_steering_vector(
        model=model,
        positive_dataset=Dataset.from_list([{"text": t} for t in pos_data]),
        negative_dataset=Dataset.from_list([{"text": t} for t in neg_data]),
        location=loc,
        test_dataset=Dataset.from_list([{"text": t} for t in test_data]),
        intervene_location=loc,  # Apply at same layer where extracted
        record_location=rec_loc,
        max_new_tokens=20,
        coeff=6.0  # Slightly increased - activation changes of 37-52 are good, but can try stronger
    )

if __name__ == "__main__":
    test_all()
