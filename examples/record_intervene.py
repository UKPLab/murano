from typing import Union
import torch
from datasets import Dataset

from federico_location import BatchedMuranoModel as BaseBatchedMuranoModel
from federico_visualization import ActivationDataset
from murano import Location
from murano.utils import (
    prepare_input_ids,
    prepare_intervention_activation,
    steering_vector_to_activation_dataset,
    create_intervention_hook,
)
from sentiment_data import POSITIVE_SAMPLES, NEGATIVE_SAMPLES


class BatchedMuranoModel(BaseBatchedMuranoModel):
    """Extension of BatchedMuranoModel with intervention capabilities."""

    def record_intervene(
        self,
        input: Union[str, torch.Tensor, dict],
        intervene_location: Location,
        record_location: Location,
        activation_dataset: ActivationDataset,
        mode: str = "replacement",
    ) -> dict:
        """
        Replace or add activations at intervene_location with activation_dataset, then record at record_location.

        Args:
            mode: "replacement" (default) to replace activations, "addition" to add activations.

        Returns: {"activations": [...], "input_ids": tensor}
        """
        input_ids, _ = prepare_input_ids(
            input, self.model.tokenizer, next(self.model.parameters()).device
        )
        batch_size = input_ids.shape[0]
        intervention_activation = prepare_intervention_activation(
            activation_dataset,
            intervene_location,
            batch_size,
            next(self.model.parameters()).device,
            coeff=1.0,
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

                        if (
                            intervene_location.token_pos
                            and len(intervene_location.token_pos) > 0
                            and intervene_location.token_pos[0] is not None
                        ):
                            pos = intervene_location.token_pos[0]
                            if pos < 0:
                                pos = target.shape[1] + pos
                            pos = max(0, min(pos, target.shape[1] - 1))
                            # Apply intervention based on mode
                            if mode == "replacement":
                                target[:, pos, :] = intervention_activation
                            elif mode == "addition":
                                target[:, pos, :] = (
                                    target[:, pos, :] + intervention_activation
                                )
                            else:
                                raise ValueError(
                                    f"Invalid mode: {mode}. Must be 'replacement' or 'addition'."
                                )
                        else:
                            # Apply to all positions based on mode
                            if mode == "replacement":
                                target[:] = intervention_activation
                            elif mode == "addition":
                                target[:] = target[:] + intervention_activation
                            else:
                                raise ValueError(
                                    f"Invalid mode: {mode}. Must be 'replacement' or 'addition'."
                                )

                # Record
                for layer in record_location.layers:
                    layer_activation = []
                    for module in record_location.modules:
                        layer_module = getattr(layers_list[layer], module)
                        output = layer_module.output
                        source = output[0] if isinstance(output, tuple) else output

                        if (
                            record_location.token_pos
                            and len(record_location.token_pos) > 0
                            and record_location.token_pos[0] is not None
                        ):
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
        **generation_kwargs,
    ) -> dict:
        """
        Generate text with intervention applied via hooks at each forward pass.

        Uses HF's generate() with hooks that add activation_dataset at intervene_location.
        Returns: {"output_ids": tensor, "input_ids": tensor}
        """
        tokenizer = self.model.tokenizer
        raw_model = self.model._model if hasattr(self.model, "_model") else self.model
        device = raw_model.device

        input_ids, attention_mask = prepare_input_ids(input, tokenizer, device)
        batch_size = input_ids.shape[0]
        intervention_activation = prepare_intervention_activation(
            activation_dataset, intervene_location, batch_size, device, coeff=1.0
        )

        # Set up hooks for intervention
        handles = []
        layers = raw_model.transformer.h

        # Create hook function using utility
        hook_fn = create_intervention_hook(intervention_activation, intervene_location)
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
                    "pad_token_id": tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id,
                    **generation_kwargs,
                }

                if attention_mask is not None:
                    gen_kwargs["attention_mask"] = attention_mask

                # Call HF's generate() - hooks will apply intervention at each step
                output_ids = raw_model.generate(input_ids, **gen_kwargs)
        except Exception as e:
            raise e
        finally:
            # Remove all hooks
            for handle in handles:
                handle.remove()

        return {
            "output_ids": output_ids,
            "input_ids": input_ids,
        }


def extract_steering_vector(
    model: BatchedMuranoModel,
    positive_dataset: Dataset,
    negative_dataset: Dataset,
    extract_location: Location,
    **kwargs,
) -> dict:
    """
    Compute steering vector as mean(positive_activations) - mean(negative_activations) at extract_location.

    Note: Steering vector is location-specific. For full semantic difference, extract multiple
    vectors at different locations or use a different approach.

    Returns: {"steering_vector": tensor, "positive_activations": ActivationDataset,
             "negative_activations": ActivationDataset, "location": Location}
    """
    kwargs.setdefault("batch_size", 1)

    # Record activations from both datasets
    pos_artifact = model.run_task(positive_dataset, extract_location, **kwargs)
    neg_artifact = model.run_task(negative_dataset, extract_location, **kwargs)

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

    pos_acts = to_activation_dataset(pos_artifact, extract_location)
    neg_acts = to_activation_dataset(neg_artifact, extract_location)

    pos_mean = torch.tensor(pos_acts.activations).mean(0)
    neg_mean = torch.tensor(neg_acts.activations).mean(0)
    steering_vector = pos_mean - neg_mean

    return {
        "steering_vector": steering_vector,
        "positive_activations": pos_acts,
        "negative_activations": neg_acts,
        "location": extract_location,
    }


def apply_steering_vector(
    model: BatchedMuranoModel,
    input: Union[str, torch.Tensor, dict, Dataset],
    steering_vector: torch.Tensor,
    intervene_location: Location,
    coeff: float = 1.0,
    **kwargs,
) -> dict:
    """
    Apply steering vector for text generation (calls generate_intervene).

    For recording, use record_intervene pattern but add steering vector instead of replace.
    """
    # Apply coeff to steering vector before converting to activation dataset
    scaled_steering_vector = steering_vector * coeff
    sv_dataset = steering_vector_to_activation_dataset(
        scaled_steering_vector, intervene_location
    )

    if isinstance(input, Dataset):
        return [
            apply_steering_vector(
                model=model,
                input=ex["text"],
                steering_vector=steering_vector,
                intervene_location=intervene_location,
                coeff=coeff,
                **kwargs,
            )
            for ex in input
        ]

    tokenizer = model.model.tokenizer
    device = next(model.model.parameters()).device
    input_ids, attention_mask = prepare_input_ids(input, tokenizer, device)

    input_dict = {"input_ids": input_ids}
    if attention_mask is not None:
        input_dict["attention_mask"] = attention_mask

    return model.generate_intervene(
        input=input_dict,
        intervene_location=intervene_location,
        activation_dataset=sv_dataset,
        **{
            k: v
            for k, v in kwargs.items()
            if k not in ["batch_size", "max_length", "attention_mask"]
        },
    )


def compute_steering_vector(
    model: BatchedMuranoModel,
    positive_dataset: Dataset,
    negative_dataset: Dataset,
    extract_location: Location,
    test_dataset: Dataset = None,
    intervene_location: Location = None,
    coeff: float = 1.0,
    **kwargs,
) -> dict:
    """
    Convenience function: extract steering vector and optionally apply to test_dataset.

    Calls extract_steering_vector and optionally apply_steering_vector.
    Returns steering vector + baseline/steered outputs if test_dataset provided.
    """
    sv_result = extract_steering_vector(
        model=model,
        positive_dataset=positive_dataset,
        negative_dataset=negative_dataset,
        extract_location=extract_location,
        **{k: v for k, v in kwargs.items() if k != "max_new_tokens"},
    )

    result = {
        "steering_vector": sv_result["steering_vector"],
        "positive_activations": sv_result["positive_activations"],
        "negative_activations": sv_result["negative_activations"],
    }

    if test_dataset:
        if intervene_location is None:
            intervene_location = extract_location

        # Get baseline generation (without intervention)
        tokenizer = model.model.tokenizer
        raw_model = (
            model.model._model if hasattr(model.model, "_model") else model.model
        )
        test_texts = [ex["text"] for ex in test_dataset]
        inputs = tokenizer(
            test_texts, return_tensors="pt", padding=True, truncation=True
        )
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
                pad_token_id=tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id,
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
            coeff=coeff,
            max_new_tokens=max_new,
        )

        # Decode steered output text
        steered_output_text = tokenizer.batch_decode(
            steered_result["output_ids"], skip_special_tokens=True
        )

        result["baseline_output_ids"] = baseline_ids
        result["baseline_output_text"] = baseline_text
        result["steered_output_ids"] = steered_result["output_ids"]
        result["steered_output_text"] = steered_output_text

        print("\n=== Generation Comparison ===")
        for i, prompt in enumerate(test_texts):
            print(f'\nPrompt: "{prompt}"')
            print(f'  Baseline: "{baseline_text[i]}"')
            print(f'  Steered:  "{steered_output_text[i]}"')

    return result


def test_all():
    model = BatchedMuranoModel.from_pretrained("gpt2")

    pos_data = POSITIVE_SAMPLES
    neg_data = NEGATIVE_SAMPLES

    test_data = ["I think that the food was", "The food was", "My friend is a"]

    # Compute steering vector at layer 8 MLP (not 6 because 8 is a deeper layer for abstract sentiment)
    # Test case 1: Apply to last token only
    loc_extract = Location(
        layers=[8], modules=["mlp"], token_pos=[-1]
    )  # Extract at last token
    loc_intervene = Location(
        layers=[8], modules=["mlp"], token_pos=[-1]
    )  # Apply at last token

    print("=== Test 1: Steering vector applied to last token only ===")
    compute_steering_vector(
        model=model,
        positive_dataset=Dataset.from_list([{"text": t} for t in pos_data]),
        negative_dataset=Dataset.from_list([{"text": t} for t in neg_data]),
        extract_location=loc_extract,  # Extract requires concrete token_pos
        test_dataset=Dataset.from_list([{"text": t} for t in test_data]),
        intervene_location=loc_intervene,  # Apply at last token
        max_new_tokens=15,
        coeff=6.0,
    )

    print("\n=== Test 2: Steering vector applied to ALL tokens ===")
    compute_steering_vector(
        model=model,
        positive_dataset=Dataset.from_list([{"text": t} for t in pos_data]),
        negative_dataset=Dataset.from_list([{"text": t} for t in neg_data]),
        extract_location=Location(
            layers=[8], modules=["mlp"], token_pos=[-1]
        ),  # Extract at last token
        test_dataset=Dataset.from_list([{"text": t} for t in test_data]),
        intervene_location=Location(
            layers=[8], modules=["mlp"], token_pos=None
        ),  # Apply to all tokens during generation
        max_new_tokens=15,
        coeff=6.0,
    )


if __name__ == "__main__":
    test_all()
