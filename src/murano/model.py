from typing import List, Union, Optional
import sys
import os
import torch
from torch.utils.data import DataLoader
from datasets import Dataset

from nnsight import LanguageModel

from .lenses.base_lens import BaseLens
from .utils import Location

# Add examples directory to path if available (for ActivationDataset from federico_visualization)
_EXAMPLES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'examples')
if os.path.exists(_EXAMPLES_PATH) and _EXAMPLES_PATH not in sys.path:
    sys.path.insert(0, _EXAMPLES_PATH)


class MuranoModel:
    def __init__(self, model_name: str):
        self.model = LanguageModel(model_name, device_map="auto", dispatch=True)
        self.model_name = model_name

    @classmethod
    def from_pretrained(cls, model_name: str):
        return cls(model_name)

    def run_with_lens(
        self, prompt: str, lens: BaseLens, locations: List[Location]
    ) -> dict:
        activations = []
        layer_indices = []

        tokenized = self.model.tokenizer(prompt, return_tensors="pt")
        input_ids = tokenized["input_ids"]

        with self.model.trace() as tracer:
            with tracer.invoke(prompt):
                layers_list = list(self.model.transformer.h)

                for location in locations:
                    if isinstance(location.layers, slice):
                        selected_layers = list(range(len(layers_list)))
                    else:
                        selected_layers = (
                            location.layers
                            if isinstance(location.layers, list)
                            else [location.layers]
                        )

                    for layer_idx in selected_layers:
                        layer = layers_list[layer_idx]

                        output = layer.output
                        if isinstance(output, tuple):
                            hidden_states = output[0]
                        else:
                            hidden_states = output

                        saved_output = hidden_states.save()
                        activations.append(saved_output)
                        layer_indices.append(layer_idx)

        artifact = {
            "prompt": prompt,
            "activations": [act.value for act in activations],
            "layer_indices": layer_indices,
            "input_ids": input_ids[0],  # type: ignore
            "model": self.model,
            "tokenizer": self.model.tokenizer,
        }

        return lens.process(artifact)

    # ============================================================================
    # Recording methods (from federico_location.py, federico_dataset_batching.py, federico_visualization.py)
    # ============================================================================

    def run_recording(
        self, 
        input: Union[str, torch.Tensor, dict], 
        location: Union[Location, List[Location]], 
        **kwargs
    ) -> dict:
        """
        Run the model with tracing enabled to record activations at specified locations.
        Computes a single forward pass for a batch of inputs.
        
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
            raise ValueError("Input must be a string, tensor, or dictionary with 'input_ids'.")

        with self.model.trace() as tracer:
            with tracer.invoke(input_ids, max_length=10, **kwargs):
                layers_list = list(self.model.transformer.h)

                # Handle single Location object
                if isinstance(location, Location) and not isinstance(location, list):
                    for layer in location.layers:
                        layer_activation = []
                        for module in location.modules:
                            # Special handling for "output" module (from federico_visualization.py line 226-228)
                            if module == "output":
                                layer_module = layers_list[layer]
                                # Original code exactly: layer_module.output[0][ :, location.token_pos, :] if location.token_pos is not None else layer_module.output[0]
                                hidden_states = layer_module.output[0][:, location.token_pos, :] if location.token_pos is not None else layer_module.output[0]
                            else:
                                # Original code exactly from federico_location.py line 66-71
                                layer_module = getattr(layers_list[layer], module)
                                output = layer_module.output
                                if isinstance(output, tuple):
                                    hidden_states = output[0][:, location.token_pos, :] if location.token_pos is not None else output[0]
                                else:
                                    hidden_states = output[:, location.token_pos, :] if location.token_pos is not None else output

                            module_activation = hidden_states.save()
                            layer_activation.append(module_activation)
                        activations.append(layer_activation)
                
                # Handle List[Location] format
                elif isinstance(location, list) and all(isinstance(loc, Location) for loc in location):
                    layer_indices = []
                    for loc in location:
                        if isinstance(loc.layers, slice):
                            selected_layers = list(range(len(layers_list)))
                        else:
                            selected_layers = (
                                loc.layers
                                if isinstance(loc.layers, list)
                                else [loc.layers]
                            )

                        for layer_idx in selected_layers:
                            layer = layers_list[layer_idx]
                            output = layer.mlp.output  # Default to mlp (from federico_dataset_batching.py)
                            if isinstance(output, tuple):
                                hidden_states = output[0]
                            else:
                                hidden_states = output

                            saved_output = hidden_states.save()
                            activations.append(saved_output)
                            layer_indices.append(layer_idx)
                    
                    # Handle tuple outputs (from federico_dataset_batching.py)
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

    def _stack_activations(self, obj, location: Optional[Union[Location, List[Location]]] = None) -> torch.Tensor:
        """
        Utility function that reshapes activations to appropriate format.
        Necessary because activations are returned in heterogeneous nested structures.
        
        If location is provided (single Location with modules), uses 5D format.
        Otherwise, uses 4D format.
        """
        obj = self._stack_activations_recursive(obj)
        
        # Check if location has modules attribute (single Location vs list)
        has_modules = location is not None and hasattr(location, 'modules') and not isinstance(location, list)
        if has_modules:
            # Version from federico_visualization.py
            obj = obj.permute(0, 3, 1, 2, 4, 5)
            obj = obj.reshape(obj.shape[0] * obj.shape[1], obj.shape[2], obj.shape[3], obj.shape[4], obj.shape[5])
            # (num_examples, num_layers, num_modules, seq_len, hidden_dim)
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
                assert obj.shape[4] == self.model.config.hidden_size, f"Expected {self.model.config.hidden_size} hidden size, got {obj.shape[4]}"
            assert obj.dim() == 5, f"Expected 5 dimensions, got {obj.dim()}"
        else:
            # Version from federico_dataset_batching.py
            obj = obj.permute(0, 2, 1, 3, 4)
            obj = obj.reshape(obj.shape[0] * obj.shape[1], obj.shape[2], obj.shape[3], obj.shape[4])
        
        return obj

    def _stack_activations_recursive(self, obj):
        """
        Recursively stack activations from a nested structure.
        """
        if isinstance(obj, dict):
            # Only recurse into the "activations" field
            if "activations" not in obj:
                raise KeyError(f"Expected 'activations' key in dict, got keys: {list(obj.keys())}")
            return self._stack_activations_recursive(obj["activations"])
        
        elif isinstance(obj, (list, tuple)):
            # Recurse and stack along new dimension
            return torch.stack([self._stack_activations_recursive(item) for item in obj], dim=0)
        
        elif isinstance(obj, torch.Tensor):
            return obj
        
        elif hasattr(obj, 'value') and isinstance(obj.value, torch.Tensor):
            return obj.value
        
        else:
            raise TypeError(f"Unsupported type in structure: {type(obj)}")

    def _get_dataloader(self, dataset: Dataset, batch_size: int = 4) -> DataLoader:
        """
        Create a DataLoader for the given dataset.
        """
        # Custom collate function stitches separate inputs coming from dataset
        def collate_fn(batch):
            collated = {}
            for key in batch[0]:
                values = [example[key] for example in batch]
                if key in ["input_ids", "attention_mask"]:  # tensorize only these
                    collated[key] = torch.stack([torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in values])
                else:
                    collated[key] = values  # keep as list
            return collated
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def run_task(
        self, 
        dataset: Dataset, 
        location: Union[Location, List[Location]], 
        **kwargs
    ) -> dict:
        """
        Run the model on a dataset with tracing enabled to record activations 
        at specified location.
        Processes the dataset in batches by calling run_recording for each batch.
        
        Args:
            dataset: HuggingFace Dataset with 'text' field
            location: Location or list of Location objects specifying where to record
            **kwargs: batch_size and other arguments passed to run_recording
        """
        # Utility function to tokenize used in map dataset
        def process_dataset(example, tokenizer, max_length=10):
            example["input_ids"] = tokenizer(example["text"], return_tensors="pt",
                                             max_length=max_length)["input_ids"][0]
            return example

        dataset = dataset.map(
            lambda x: process_dataset(x, self.model.tokenizer),
            batched=False,
        )
        batch_size = kwargs.pop("batch_size", 4)
        dataloader = self._get_dataloader(dataset, batch_size)
        activations = []
        global_metadata = {
            "model_name": self.model_name,
            "tokenizer": self.model.tokenizer.name_or_path,
            "batch_size": batch_size,
            "location": location,
        }
        for example in dataloader:
            input_ids = example["input_ids"]
            activation = self.run_recording(input_ids, location, **kwargs)
            activations.append(activation)
        
        # Stack activations
        has_modules = hasattr(location, 'modules') and not isinstance(location, list)
        activations = self._stack_activations(activations, location if has_modules else None)
        
        # Try to return ActivationDataset if available, otherwise return dict
        has_modules_single_location = isinstance(location, Location) and not isinstance(location, list)
        if has_modules_single_location:
            try:
                from federico_visualization import ActivationDataset
                artifact = ActivationDataset(
                    activations=activations,
                    location=location,
                    global_metadata=global_metadata,
                    dataset=dataset
                )
            except ImportError:
                # Fallback to dict if ActivationDataset not available
                artifact = {
                    "activations": activations,
                    "global_metadata": global_metadata,
                    "dataset": dataset,
                    "location": location,
                }
        else:
            # Return dict for list of locations
            artifact = {
                "activations": activations,
                "global_metadata": global_metadata,
                "dataset": dataset,
            }
        
        return artifact

    # ============================================================================
    # Intervention methods (from record_intervene.py)
    # ============================================================================

    def record_intervene(
        self,
        input: Union[str, torch.Tensor, dict],
        intervene_location: Location,
        record_location: Location,
        activation_dataset,  # ActivationDataset from federico_visualization
        mode: str = "replacement",
    ) -> dict:
        """
        Replace or add activations at intervene_location with activation_dataset, then record at record_location.
        
        Args:
            mode: "replacement" (default) to replace activations, "addition" to add activations.
        
        Returns: {"activations": [...], "input_ids": tensor}
        
        Requires: ActivationDataset from examples/federico_visualization.py to be available.
        """
        from .utils import prepare_input_ids, prepare_intervention_activation
        
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
                            # Apply intervention based on mode
                            if mode == "replacement":
                                target[:, pos, :] = intervention_activation
                            elif mode == "addition":
                                target[:, pos, :] = target[:, pos, :] + intervention_activation
                            else:
                                raise ValueError(f"Invalid mode: {mode}. Must be 'replacement' or 'addition'.")
                        else:
                            # Apply to all positions based on mode
                            if mode == "replacement":
                                target[:] = intervention_activation
                            elif mode == "addition":
                                target[:] = target[:] + intervention_activation
                            else:
                                raise ValueError(f"Invalid mode: {mode}. Must be 'replacement' or 'addition'.")

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
        activation_dataset,  # ActivationDataset from federico_visualization
        max_new_tokens: int = 20,
        **generation_kwargs,
    ) -> dict:
        """
        Generate text with intervention applied via hooks at each forward pass.
        
        Uses HF's generate() with hooks that add activation_dataset at intervene_location.
        Returns: {"output_ids": tensor, "input_ids": tensor}
        
        Requires: ActivationDataset from examples/federico_visualization.py to be available.
        """
        from .utils import (
            prepare_input_ids,
            prepare_intervention_activation,
            create_intervention_hook,
        )
        
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
                    "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    **generation_kwargs
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
