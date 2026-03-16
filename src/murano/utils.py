from typing import List, Union, Tuple, Optional
import numpy as np
import torch
from datasets import Dataset


# TODO: make Location sortable
class Location:
    """
    Location specifies a slice of model activations to extract or analyze.
    """

    VALID_KEYWORDS = {"prompt", "generation", "all", "last"}

    def __init__(
        self,
        layers: Union[int, List[int]],
        modules: Union[str, List[str]] = "mlp",
        token_pos: Union[int, List[int], str] = None,
    ):
        # FIX: Allow slice to pass through without being wrapped in a list
        if isinstance(layers, (list, slice)):
            self.layers = layers
        else:
            self.layers = [layers]

        self.modules = modules if isinstance(modules, list) else [modules]

        # Handle token_pos
        if isinstance(token_pos, str):
            if token_pos not in self.VALID_KEYWORDS:
                raise ValueError(
                    f"Invalid token_pos keyword: '{token_pos}'. "
                    f"Must be one of {self.VALID_KEYWORDS}."
                )
            self.token_pos = token_pos
        elif isinstance(token_pos, list) or token_pos is None:
            self.token_pos = token_pos
        else:
            self.token_pos = [token_pos]

    def __repr__(self):
        return f"(layers={self.layers}, modules={self.modules}, token_pos={self.token_pos})"

    def resolve_indices(self, total_seq_len: int, prompt_len: int) -> List[int]:
        """
        Resolve the token_pos (keyword or list) into concrete indices
        based on the provided sequence dimensions.

        Args:
            total_seq_len: The total length of the sequence (prompt + generation).
            prompt_len: The length of the prompt.

        Returns:
            A list of integers representing the token positions.
        """
        if isinstance(self.token_pos, list):
            # Resolve negative indices in the list
            return [t if t >= 0 else total_seq_len + t for t in self.token_pos]

        if self.token_pos is None:
            # Default to all
            return list(range(0, total_seq_len))

        if self.token_pos == "prompt":
            return list(range(0, prompt_len))
        elif self.token_pos == "generation":
            return list(range(prompt_len, total_seq_len))
        elif self.token_pos == "all":
            return list(range(0, total_seq_len))
        elif self.token_pos == "last":
            return list(range(prompt_len - 1, prompt_len))

        # Should be unreachable due to init validation, but fallback
        return list(range(0, total_seq_len))

    def applies_to_prompt_phase(self, prompt_len: int) -> bool:
        """
        Check if this location includes tokens from the prompt phase.
        """
        if isinstance(self.token_pos, str):
            if self.token_pos in ["prompt", "last", "all"]:
                return True
            if self.token_pos in ["generation"]:
                return False
            return False

        if self.token_pos is None:
            return True

        # Check list indices
        # We assume indices < prompt_len are in prompt phase
        # Negative indices are ambiguous without total len, but usually imply end (generation)
        return any(0 <= t < prompt_len for t in self.token_pos)

    def applies_to_generation_phase(self, prompt_len: int) -> bool:
        """
        Check if this location includes tokens from the generation phase.
        """
        if isinstance(self.token_pos, str):
            if self.token_pos in ["generation", "all"]:
                return True
            if self.token_pos == "prompt":
                return False
            return False

        if self.token_pos is None:
            return True

        # Check list indices
        # We assume indices >= prompt_len are in generation phase
        # Negative indices usually imply end (generation)
        return any(t >= prompt_len or t < 0 for t in self.token_pos)

    def resolve_time_steps(
        self, num_steps: Optional[int] = None
    ) -> Union[slice, List[int]]:
        """
        Resolve token_pos into a slice or list suitable for nnsight's tracer.iter[].

        Mappings:
            "prompt"     -> slice(0, 1)    (Step 0)
            "last"       -> slice(0, 1)    (Step 0)
            "generation" -> slice(1, None) (Steps 1 to End)
            "all"        -> slice(None)    (All Steps)
        """
        if isinstance(self.token_pos, list):
            # FIX: Force explicit lists to run ONLY during the Prompt Phase (Step 0)
            return slice(0, 1)

        if self.token_pos is None:
            return slice(None)

        if self.token_pos == "prompt" or "last":
            return slice(0, 1)
        elif self.token_pos == "generation":
            return slice(1, None)
        elif self.token_pos == "all":
            return slice(None)

        return slice(None)

    def get_slice_idx(self, _slice):
        """
        Convert a subset of a Location (slice) into indices for numpy array slicing.
        """
        if not self.is_valid_slice(_slice):
            raise ValueError(f"Slice {_slice} is not included within location {self}.")

        # Get indices for slicing activations based on slice Location
        # All Layers case
        if isinstance(self.layers, slice):
            layer_idx = _slice.layers
        else:
            # Original logic for list
            layer_idx = [self.layers.index(l) for l in _slice.layers]

        module_idx = [self.modules.index(m) for m in _slice.modules]

        # Handle string token_pos in get_slice_idx if possible, else raise
        if isinstance(self.token_pos, str):
            if isinstance(_slice.token_pos, str) and _slice.token_pos == self.token_pos:
                # If requesting the exact same keyword, return full slice
                token_idx = slice(None)
            else:
                raise NotImplementedError(
                    "Slicing by index into a keyword-based Location is not yet supported."
                )
        else:
            token_idx = [self.token_pos.index(p) for p in _slice.token_pos]

        return (slice(None), layer_idx, module_idx, token_idx, slice(None))

    def is_valid_slice(self, _slice) -> bool:
        """
        Check if the slice is contained within this location.
        """
        if isinstance(self.layers, slice):
            # If we have all layers, assume any subset request is valid
            layers_valid = True
        else:
            layers_valid = all(l in self.layers for l in _slice.layers)

        # Handle token_pos validation with keywords
        if isinstance(self.token_pos, str):
            if self.token_pos == "all":
                token_valid = True

            elif self.token_pos == "generation":
                token_valid = _slice.token_pos == "generation"

            else:
                token_valid = self.token_pos == _slice.token_pos
        else:
            token_valid = all(p in self.token_pos for p in _slice.token_pos)

        return (
            layers_valid
            and all(m in self.modules for m in _slice.modules)
            and token_valid
        )


# TODO: remove
# Backward compatibility alias
LayerLocation = Location


class ActivationDataset:
    """
    ActivationDataset stores model activations and associated metadata.
    Internally uses NumPy for storage.
    """

    def __init__(
        self,
        activations: Union[np.ndarray, torch.Tensor],
        location: Location,
        global_metadata: dict,
        dataset: Dataset,
    ):
        """
        Initializes the ActivationDataset with activations and metadata.
        Accepts activations as either numpy arrays or torch tensors (auto-converted to numpy).
        """
        self.activations = self._to_numpy(activations)
        self.location = location
        self.global_metadata = global_metadata
        self.dataset = dataset

        # Infer key activation dimensions
        ref_activation = self.activations.shape
        self.num_examples = ref_activation[0]
        self.num_layers = ref_activation[1]
        self.num_modules = ref_activation[2]
        self.seq_len = ref_activation[3]
        self.hidden_dim = ref_activation[4]

    def iloc(self, *idx):
        return self.activations[idx]

    def __getitem__(self, key):
        """
        Supports both Location-based slicing and dictionary-style string access.

        Expected indexing order for Location:
            [examples, layers, modules, tokens, hidden_dim]
        """
        if isinstance(key, str):
            # Dictionary-style access for attributes
            if key == "activations":
                return self.activations
            elif key == "location":
                return self.location
            elif key == "global_metadata":
                return self.global_metadata
            elif key == "dataset":
                return self.dataset
            else:
                raise KeyError(f"Key '{key}' not found")
        else:
            # Location-based slicing
            idx = self.location.get_slice_idx(key)
            return self.activations[idx].squeeze()

    def _to_numpy(self, activations: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(activations, torch.Tensor):
            return activations.detach().cpu().numpy()
        elif isinstance(activations, np.ndarray):
            return activations
        else:
            raise TypeError(
                f"Activations must be a numpy array or torch tensor, got {type(activations)}."
            )


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
    activation_dataset,
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

    # Update: Handle keyword token_pos for dimension calculation
    if isinstance(location.token_pos, str):
        # Assuming steering vectors are often 1-token vectors broadcasted.
        token_pos_dim = 1
        # If the input steering vector has more tokens, we trust its shape
        if sv_array.shape[3] > 1:
            token_pos_dim = sv_array.shape[3]
    elif isinstance(location.token_pos, list):
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
            raise ValueError(
                f"Unexpected intervention activation shape: {intervention_activation.shape}"
            )

        # Apply intervention based on Location.token_pos
        token_pos = location.token_pos

        # New: Handle keyword token_pos in hook
        if isinstance(token_pos, str):
            # For "last", we target the last token of the current chunk
            if token_pos == "last":
                output_tensor[:, -1:, :] = (
                    output_tensor[:, -1:, :] + act_full[:, -1:, :]
                )
            else:
                # For "prompt", "generation", "all", we typically apply to everything
                # that made it into this hook call (since nnsight filtered the steps).
                output_tensor[:] = output_tensor[:] + act_full

        # Existing logic for list/None
        elif token_pos is None or (
            isinstance(token_pos, list) and len(token_pos) == 1 and token_pos[0] is None
        ):
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
                    output_tensor[:, pos_idx, :] = (
                        output_tensor[:, pos_idx, :] + act_full[:, pos_idx, :]
                    )
        else:
            raise ValueError(
                f"Invalid token_pos in Location: {token_pos}. Must be None, [None], a list of integers, or a valid keyword."
            )

        return (
            (output_tensor,) + output[1:]
            if isinstance(output, tuple)
            else output_tensor
        )

    return hook_fn
