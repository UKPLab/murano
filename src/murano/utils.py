from typing import List, Union, Tuple
import numpy as np
import torch


class Location:
    """
    Location specifies where to extract or intervene in model activations.
    
    Args:
        layers: Layer index/indices (int, list of ints, or slice)
        modules: Module name(s) within layers (e.g., "mlp", "attn", "output")
        token_pos: Token position(s) to extract (int, list of ints, or None for all tokens)
    """
    def __init__(
        self, 
        layers: Union[int, List[int], slice], 
        modules: Union[str, List[str]] = "mlp",
        token_pos: Union[int, List[int], None] = None
    ):
        # Normalize layers to list (or keep as slice)
        if isinstance(layers, slice):
            self.layers = layers
        else:
            self.layers = layers if isinstance(layers, list) else [layers]
        
        # Normalize modules to list
        self.modules = modules if isinstance(modules, list) else [modules]
        
        # Keep token_pos as is (can be int, list, or None)
        self.token_pos = token_pos

    def __repr__(self):
        return f"Location(layers={self.layers}, modules={self.modules}, token_pos={self.token_pos})"


# Backward compatibility alias
LayerLocation = Location


# Intervention utility functions
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
    activation_dataset,  # ActivationDataset from federico_visualization
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
):
    """Convert a steering vector tensor to an ActivationDataset."""
    # Import ActivationDataset when needed
    try:
        from federico_visualization import ActivationDataset
    except ImportError:
        import sys
        import os
        examples_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'examples')
        if examples_path not in sys.path:
            sys.path.insert(0, examples_path)
        from federico_visualization import ActivationDataset
    
    sv_array = steering_vector.detach().cpu().numpy()

    # Expand to [1, num_layers, num_modules, num_token_pos, hidden_dim]
    while sv_array.ndim < 5:
        sv_array = np.expand_dims(sv_array, axis=0)

    if sv_array.shape[0] != 1:
        sv_array = sv_array[np.newaxis, ...]

    # Calculate dimensions - handle slice, list, or single value for layers
    if isinstance(location.layers, slice):
        layers_dim = 1  # Will be expanded by ActivationDataset if needed
    elif isinstance(location.layers, list):
        layers_dim = len(location.layers)
    else:
        layers_dim = 1
    
    modules_dim = len(location.modules) if location.modules else 1
    
    if isinstance(location.token_pos, list):
        token_pos_dim = len(location.token_pos)
    elif location.token_pos is not None:
        token_pos_dim = 1
    else:
        token_pos_dim = 1
    
    dims = [layers_dim, modules_dim, token_pos_dim]
    for i, dim in enumerate(dims, 1):
        if sv_array.shape[i] != dim:
            sv_array = np.repeat(sv_array, dim, axis=i)

    return ActivationDataset(
        activations=sv_array,
        location=location,
        global_metadata={"type": "steering_vector"},
        dataset=None,
    )


def create_intervention_hook(intervention_activation: torch.Tensor, location: Location):
    """
    Create a forward hook function that applies intervention at each forward pass.
    
    The hook fires at each generation step. At each step:
    - seq_len = current sequence length (prompt + tokens generated so far)
    - User controls where intervention is applied via Location.token_pos:
      * None or [None] → apply to all tokens in current sequence
      * [-1] → apply to last token (changes as tokens are generated)
      * [0] → apply to first token
      * [0, -1] → apply to first and last tokens
    
    Args:
        intervention_activation: The activation tensor to add (shape: [batch, hidden] or [batch, seq, hidden])
        location: Location object specifying where to apply the intervention
    
    Returns:
        A hook function that can be registered with register_forward_hook()
    """
    def hook_fn(module, args, output):
        output_tensor = (output[0] if isinstance(output, tuple) else output).clone()
        seq_len = output_tensor.shape[1]
        
        # Normalize intervention activation shape to [batch, seq, hidden]
        act = intervention_activation
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
            raise ValueError(f"Unexpected intervention activation shape: {intervention_activation.shape}")
        
        # Apply intervention based on Location.token_pos
        token_pos = location.token_pos
        
        # Check if applying to all tokens
        if token_pos is None or (isinstance(token_pos, list) and len(token_pos) == 1 and token_pos[0] is None):
            # Apply to all tokens in current sequence
            output_tensor[:] = output_tensor[:] + act_full
        elif isinstance(token_pos, (list, tuple)) and len(token_pos) > 0:
            # Apply to specified token positions (relative to current sequence)
            for pos in token_pos:
                if pos is not None:
                    # Convert relative index: -1 means last token, 0 means first
                    pos_idx = pos if pos >= 0 else seq_len + pos
                    pos_idx = max(0, min(pos_idx, seq_len - 1))
                    # Apply intervention at this position
                    output_tensor[:, pos_idx, :] = output_tensor[:, pos_idx, :] + act_full[:, pos_idx, :]
        else:
            raise ValueError(f"Invalid token_pos in Location: {token_pos}. Must be None, [None], or a list of integers.")
        
        return (output_tensor,) + output[1:] if isinstance(output, tuple) else output_tensor
    return hook_fn
