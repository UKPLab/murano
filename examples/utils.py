from typing import Tuple, Union

import numpy as np
import torch

from federico_visualization import ActivationDataset, Location


def prepare_input_ids(
    input: Union[str, torch.Tensor, dict],
    tokenizer,
    device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare input_ids and attention_mask from string, tensor, or dict inputs."""
    if isinstance(input, str):
        inputs = tokenizer(input, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        return input_ids, attention_mask

    if isinstance(input, torch.Tensor):
        return input.to(device), None

    if isinstance(input, dict):
        input_ids = input["input_ids"].to(device)
        attention_mask = input.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        return input_ids, attention_mask

def prepare_intervention_activation(
    activation_dataset: ActivationDataset,
    intervene_location: Location,
    batch_size: int,
    device,
    coeff: float = 1.0,
) -> torch.Tensor:
    """Prepare the intervention activation tensor with correct shape and device."""
    intervention_activation = torch.tensor(activation_dataset[intervene_location])

    while intervention_activation.ndim < 2:
        intervention_activation = intervention_activation.unsqueeze(0)

    if intervention_activation.shape[0] == 1 and batch_size > 1:
        intervention_activation = intervention_activation.expand(batch_size, -1)
    elif intervention_activation.shape[0] != batch_size:
        raise ValueError(
            f"Batch size mismatch: {intervention_activation.shape[0]} vs {batch_size}"
        )

    return (intervention_activation * coeff).to(device)


def steering_vector_to_activation_dataset(
    steering_vector: torch.Tensor, location: Location
) -> ActivationDataset:
    """Convert a steering vector tensor to an ActivationDataset."""
    sv_array = steering_vector.detach().cpu().numpy()

    # Expand to [1, num_layers, num_modules, num_token_pos, hidden_dim]
    while sv_array.ndim < 5:
        sv_array = np.expand_dims(sv_array, axis=0)

    if sv_array.shape[0] != 1:
        sv_array = sv_array[np.newaxis, ...]

    dims = [
        len(location.layers) if location.layers else 1,
        len(location.modules) if location.modules else 1,
        len(location.token_pos) if location.token_pos else 1,
    ]
    for i, dim in enumerate(dims, 1):
        if sv_array.shape[i] != dim:
            sv_array = np.repeat(sv_array, dim, axis=i)

    return ActivationDataset(
        activations=sv_array,
        location=location,
        global_metadata={"type": "steering_vector"},
        dataset=None,
    )

