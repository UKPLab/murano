from typing import List, Union, Optional
import sys
import os
import torch
from torch.utils.data import DataLoader
from datasets import Dataset

from nnterp import StandardizedTransformer

from .lenses.base_lens import BaseLens
from .utils import (
    Location,
    ActivationDataset,
    prepare_input_ids,
    prepare_intervention_activation,
    steering_vector_to_activation_dataset,
    create_intervention_hook,
)


class MuranoModel:
    def __init__(self, model_name: str):
        self.model = StandardizedTransformer(
            model_name, device_map="auto", dispatch=True
        )
        self.model_name = model_name

    @classmethod
    def from_pretrained(cls, model_name: str):
        return cls(model_name)

    # TODO: integrate with record_intervene
    def record(
        self,
        input: Union[str, torch.Tensor, dict],
        location: Union[Location, List[Location]],
        **kwargs,
    ) -> dict:
        """
        Run the model with tracing enabled to record activations at specified locations.
        Computes a single forward pass for a batch of inputs without generation.

        Args:
            input: Input tensor, dict with input_ids, or string
            location: Location or list of Location objects specifying where to record
            **kwargs: Additional arguments passed to tracer.invoke
        """
        activations = []

        # Handle input format
        if isinstance(input, torch.Tensor):
            input_ids = input
        elif isinstance(input, dict):
            input_ids = input["input_ids"]
        else:
            raise ValueError(
                "Input must be a string, tensor, or dictionary with 'input_ids'."
            )

        with self.model.trace() as tracer:
            with tracer.invoke(input_ids, max_new_tokens=1, **kwargs):
                layers_list = self.model.layers

                # Handle single Location object
                if isinstance(location, Location) and not isinstance(location, list):
                    for layer in location.layers:
                        layer_activation = []
                        for module in location.modules:
                            # Special handling for "output" module
                            if module == "output":
                                layer_module = layers_list[layer]
                                # hidden_states = layer_module.output[0][:, location.token_pos, :] if location.token_pos is not None else layer_module.output[0]
                                hidden_states = (
                                    layer_module.output[:, location.token_pos, :]
                                    if location.token_pos is not None
                                    else layer_module.output
                                )

                            else:
                                # Access standardized modules (e.g., 'mlp', 'self_attn')
                                layer_module = getattr(layers_list[layer], module)
                                output = layer_module.output
                                if isinstance(output, tuple):
                                    hidden_states = (
                                        output[0][:, location.token_pos, :]
                                        if location.token_pos is not None
                                        else output[0]
                                    )
                                else:
                                    hidden_states = (
                                        output[:, location.token_pos, :]
                                        if location.token_pos is not None
                                        else output
                                    )

                            module_activation = hidden_states.save()
                            layer_activation.append(module_activation)
                        activations.append(layer_activation)

                # Handle List[Location] format
                elif isinstance(location, list) and all(
                    isinstance(loc, Location) for loc in location
                ):
                    layer_indices = []
                    for loc in location:
                        if isinstance(loc.layers, slice):
                            # Use self.model.num_layers from StandardizedTransformer
                            selected_layers = list(range(self.model.num_layers))
                        else:
                            selected_layers = (
                                loc.layers
                                if isinstance(loc.layers, list)
                                else [loc.layers]
                            )

                        for layer_idx in selected_layers:
                            # Universal layers list
                            layer = layers_list[layer_idx]

                            # Iterate over the modules specified in the Location
                            for module in loc.modules:
                                if module == "output":
                                    layer_module = layer
                                    output = layer_module.output
                                    if isinstance(output, tuple):
                                        hidden_states = (
                                            output[0][:, loc.token_pos, :]
                                            if loc.token_pos is not None
                                            else output[0]
                                        )
                                    else:
                                        hidden_states = (
                                            output[:, loc.token_pos, :]
                                            if loc.token_pos is not None
                                            else output
                                        )

                                else:
                                    # Standardized access (e.g., layer.mlp, layer.self_attn)
                                    layer_module = getattr(layer, module)
                                    output = layer_module.output
                                    if isinstance(output, tuple):
                                        hidden_states = (
                                            output[0][:, loc.token_pos, :]
                                            if loc.token_pos is not None
                                            else output[0]
                                        )
                                    else:
                                        hidden_states = (
                                            output[:, loc.token_pos, :]
                                            if loc.token_pos is not None
                                            else output
                                        )

                                saved_output = hidden_states.save()
                                activations.append(saved_output)
                                layer_indices.append(layer_idx)

                    # Handle tuple outputs
                    if activations and isinstance(activations[0].value, tuple):
                        activations = [act.value[0] for act in activations]
                    else:
                        activations = [act.value for act in activations]
                else:
                    raise ValueError(f"Unsupported location type: {type(location)}")

        artifact = {
            "activations": activations,
            "input_ids": input_ids,
        }

        return artifact

    def _stack_activations(
        self, obj, location: Optional[Union[Location, List[Location]]] = None
    ) -> torch.Tensor:
        """
        Utility function that reshapes activations to appropriate format.
        Necessary because activations are returned in heterogeneous nested structures.

        If location is provided (single Location with modules), uses 5D format.
        Otherwise, uses 4D format.
        """
        obj = self._stack_activations_recursive(obj)
        # (n_batches, n_layers, n_modules, batch_size, seq_len, hidden_dim)
        # Check if location has modules attribute (single Location vs list)
        has_modules = (
            location is not None
            and hasattr(location, "modules")
            and not isinstance(location, list)
        )
        if has_modules:
            obj = obj.permute(0, 3, 1, 2, 4, 5)
            obj = obj.reshape(
                -1, obj.shape[2], obj.shape[3], obj.shape[4], obj.shape[5]
            )
            # (num_examples, num_layers, num_modules, seq_len, hidden_dim)
            if hasattr(self.model, "config") and hasattr(
                self.model.config, "hidden_size"
            ):
                assert obj.shape[4] == self.model.config.hidden_size, (
                    f"Expected {self.model.config.hidden_size} hidden size, got {obj.shape[4]}"
                )
            assert obj.dim() == 5, f"Expected 5 dimensions, got {obj.dim()}"
        else:
            obj = obj.permute(0, 2, 1, 3, 4)
            obj = obj.reshape(
                obj.shape[0] * obj.shape[1], obj.shape[2], obj.shape[3], obj.shape[4]
            )

        return obj

    def _stack_activations_recursive(self, obj):
        """
        Recursively stack activations from a nested structure.
        """
        if isinstance(obj, dict):
            # Only recurse into the "activations" field
            if "activations" not in obj:
                raise KeyError(
                    f"Expected 'activations' key in dict, got keys: {list(obj.keys())}"
                )
            return self._stack_activations_recursive(obj["activations"])

        elif isinstance(obj, (list, tuple)):
            # Recurse and stack along new dimension
            return torch.stack(
                [self._stack_activations_recursive(item) for item in obj], dim=0
            )

        elif isinstance(obj, torch.Tensor):
            return obj

        elif hasattr(obj, "value") and isinstance(obj.value, torch.Tensor):
            return obj.value

        else:
            raise TypeError(f"Unsupported type in structure: {type(obj)}")

    def record_intervene(
        self,
        input: Union[str, torch.Tensor, dict],
        location_intervention: Location,
        location_recording: Location,
        intervention_activation: torch.Tensor,
        mode: str = "replacement",
    ) -> dict:
        """
        Replace or add activations at intervene_location with activation_dataset, then record at record_location.

        Args:
            mode: "replacement" (default) to replace activations, "addition" to add activations.

        Returns: {"activations": [...], "input_ids": tensor}

        Requires: ActivationDataset from examples/federico_visualization.py to be available.
        """
        # Handle input format
        if isinstance(input, torch.Tensor):
            input_ids = input
        elif isinstance(input, dict):
            input_ids = input["input_ids"]
        else:
            raise ValueError(
                "Input must be a string, tensor, or dictionary with 'input_ids'."
            )

        batch_size = input_ids.shape[0]
        intrv_batch_size = intervention_activation.shape[0]
        # TODO: add check on hidden dim from model
        intrv_shape = tuple(intervention_activation.shape[1:])
        expected_shape = (
            len(location_intervention.layers),
            len(location_intervention.modules),
            len(location_intervention.token_pos),
        )

        # Check intervention activations size
        # Expected shape: (n examples, n layers, n modules, n tokens, hidden_dim)
        if intrv_batch_size != batch_size and intrv_batch_size != 1:
            raise ValueError(
                f"Intervention activation must have batch size 1 or {batch_size}. \
                             Found: {intrv_batch_size} "
            )
        if intrv_shape != expected_shape:
            raise ValueError(
                f"Shape mismatch: expected (n layers, n modules, n tokens) = {expected_shape}, \
                              found {intrv_shape}"
            )

        if location_intervention.token_pos and len(location_intervention.token_pos) > 0:
            intrv_pos = location_intervention.token_pos
        else:
            # Acts like : in indexing
            intrv_pos = slice(None)

        if location_recording.token_pos and len(location_recording.token_pos) > 0:
            record_pos = location_recording.token_pos
        else:
            record_pos = slice(None)

        activations = []
        with self.model.trace() as tracer:
            with tracer.invoke(input_ids, max_length=10):
                # Universal layer access
                layers_list = self.model.layers

                # Intervene
                for layer in location_intervention.layers:
                    for module in location_intervention.modules:
                        layer_module = getattr(layers_list[layer], module)
                        output = layer_module.output
                        target = output[0] if isinstance(output, tuple) else output

                        # TODO: implement passing function here
                        # Apply intervention based on mode
                        if mode == "replacement":
                            target[:, intrv_pos, :] = intervention_activation
                        elif mode == "addition":
                            target[:, intrv_pos, :] = (
                                target[:, intrv_pos, :] + intervention_activation
                            )
                        else:
                            raise ValueError(
                                f"Invalid mode: {mode}. \
                                             Must be 'replacement' or 'addition'."
                            )

                # Record
                for layer in location_recording.layers:
                    layer_activation = []
                    for module in location_recording.modules:
                        layer_module = getattr(layers_list[layer], module)
                        output = layer_module.output
                        source = output[0] if isinstance(output, tuple) else output

                        hidden_states = source[:, record_pos, :]

                        layer_activation.append(hidden_states.save())
                    activations.append(layer_activation)

        # TODO: return some class from HuggingFace (possibly CausalLMOutputWithPast, shared by Llama, Qwen, Gemma)
        return {"activations": activations, "input_ids": input_ids}

    # TODO: merge functionality with record_intervene and pick function based on args
    # TODO: change intervention input to tensor
    def generate_intervene(
        self,
        input: Union[str, torch.Tensor, dict],
        intervene_location: Location,
        activation_dataset,  # Assuming this is the source for intervention_activation
        max_new_tokens: int = 20,
        mode: str = "replacement",
        **generation_kwargs,
    ) -> dict:
        """
        Generate text with intervention applied via nnsight at each forward pass step.
        """
        if isinstance(input, torch.Tensor):
            input_ids = input
        elif isinstance(input, dict):
            input_ids = input["input_ids"]
        else:
            input_ids = input

        batch_size = 1  # Default or extract from input_ids if tensor
        if isinstance(input_ids, torch.Tensor):
            batch_size = input_ids.shape[0]

        device = next(self.model.parameters()).device
        intervention_activation = prepare_intervention_activation(
            activation_dataset, intervene_location, batch_size, device, coeff=1.0
        )

        # Determine token positions
        if intervene_location.token_pos and len(intervene_location.token_pos) > 0:
            intrv_pos = intervene_location.token_pos
        else:
            intrv_pos = slice(None)

        with self.model.generate(
            input_ids, max_new_tokens=max_new_tokens, **generation_kwargs
        ) as tracer:
            with tracer.iter[0]:  # Intervene at the first generation step only
                # Universal layer access
                layers_list = self.model.layers

                for layer_idx in intervene_location.layers:
                    for module_name in intervene_location.modules:
                        # Access the module
                        layer_module = getattr(layers_list[layer_idx], module_name)
                        output = layer_module.output

                        # Handle tuple vs tensor (consistent with your record_intervene)
                        target = output[0] if isinstance(output, tuple) else output

                        # Apply intervention (modification of the proxy)
                        if mode == "replacement":
                            target[:, intrv_pos, :] = intervention_activation
                        elif mode == "addition":
                            target[:, intrv_pos, :] = (
                                target[:, intrv_pos, :] + intervention_activation
                            )

        # tracer.generator stores the output_ids
        return {
            "output_ids": tracer.generator.output,
            "input_ids": input_ids,
        }
