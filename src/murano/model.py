from typing import List, Union, Optional
import torch
from torch.utils.data import DataLoader
from datasets import Dataset

from nnsight import LanguageModel

from .lenses.base_lens import BaseLens
from .utils import LayerLocation

# Optional imports for extended functionality
# These imports work when running from examples directory or when examples is in path
_HAS_INTERVENTION_UTILS = False
try:
    # Try importing from examples directory (works when running from examples/)
    try:
        from federico_visualization import ActivationDataset, Location
        from utils import (
            prepare_input_ids,
            prepare_intervention_activation,
            steering_vector_to_activation_dataset,
            create_intervention_hook,
        )
        _HAS_INTERVENTION_UTILS = True
    except ImportError:
        # Try importing with examples prefix (if examples is a package)
        import sys
        import os
        examples_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'examples')
        if examples_path not in sys.path:
            sys.path.insert(0, examples_path)
        from federico_visualization import ActivationDataset, Location
        from utils import (
            prepare_input_ids,
            prepare_intervention_activation,
            steering_vector_to_activation_dataset,
            create_intervention_hook,
        )
        _HAS_INTERVENTION_UTILS = True
except ImportError:
    # Define dummy types for type hints when imports fail
    Location = type('Location', (), {})
    ActivationDataset = type('ActivationDataset', (), {})


class MuranoModel:
    def __init__(self, model_name: str):
        self.model = LanguageModel(model_name, device_map="auto", dispatch=True)
        self.model_name = model_name

    @classmethod
    def from_pretrained(cls, model_name: str):
        return cls(model_name)

    def run_with_lens(
        self, prompt: str, lens: BaseLens, locations: List[LayerLocation]
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
        location: Union["Location", List[LayerLocation]], 
        **kwargs
    ) -> dict:
        """
        Run the model with tracing enabled to record activations at specified locations.
        Computes a single forward pass for a batch of inputs.
        
        Supports both Location (from examples) and List[LayerLocation] (from src) formats.
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

                # Handle Location format (from examples)
                # Check by class name to handle import failures gracefully
                location_type_name = type(location).__name__
                if location_type_name == "Location" and _HAS_INTERVENTION_UTILS:
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
                
                # Handle List[LayerLocation] format (from src)
                elif isinstance(location, list) and all(isinstance(loc, LayerLocation) for loc in location):
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

    def _stack_activations(self, obj, location: Optional[Union["Location", List[LayerLocation]]] = None) -> torch.Tensor:
        """
        Utility function that reshapes activations to appropriate format.
        Necessary because activations are returned in heterogeneous nested structures.
        
        If location is provided (Location type), uses the federico_visualization.py version.
        Otherwise, uses the federico_dataset_batching.py version.
        """
        obj = self._stack_activations_recursive(obj)
        
        location_type_name = type(location).__name__ if location else None
        if location is not None and _HAS_INTERVENTION_UTILS and location_type_name == "Location":
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
        location: Union["Location", List[LayerLocation]], 
        **kwargs
    ) -> dict:
        """
        Run the model on a dataset with tracing enabled to record activations 
        at specified location.
        Processes the dataset in batches by calling run_recording for each batch.
        
        Supports both Location (from examples) and List[LayerLocation] (from src) formats.
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
        location_type_name = type(location).__name__ if location else None
        activations = self._stack_activations(activations, location if _HAS_INTERVENTION_UTILS and location_type_name == "Location" else None)
        
        # Return format depends on location type
        if _HAS_INTERVENTION_UTILS and location_type_name == "Location":
            # Return ActivationDataset (from federico_visualization.py)
            artifact = ActivationDataset(
                activations=activations,
                location=location,
                global_metadata=global_metadata,
                dataset=dataset
            )
        else:
            # Return dict (from federico_location.py and federico_dataset_batching.py)
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
        intervene_location: "Location",
        record_location: "Location",
        activation_dataset: "ActivationDataset",
        mode: str = "replacement",
    ) -> dict:
        """
        Replace or add activations at intervene_location with activation_dataset, then record at record_location.
        
        Args:
            mode: "replacement" (default) to replace activations, "addition" to add activations.
        
        Returns: {"activations": [...], "input_ids": tensor}
        
        Requires: examples/utils.py and examples/federico_visualization.py to be available.
        """
        # Import utilities when needed
        try:
            from utils import prepare_input_ids, prepare_intervention_activation
        except ImportError:
            import sys
            import os
            examples_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'examples')
            if examples_path not in sys.path:
                sys.path.insert(0, examples_path)
            try:
                from utils import prepare_input_ids, prepare_intervention_activation
            except ImportError:
                raise ImportError("record_intervene requires examples/utils.py to be available. Make sure you're running from the examples directory or examples is in your Python path.")
        
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
        intervene_location: "Location",
        activation_dataset: "ActivationDataset",
        max_new_tokens: int = 20,
        **generation_kwargs,
    ) -> dict:
        """
        Generate text with intervention applied via hooks at each forward pass.
        
        Uses HF's generate() with hooks that add activation_dataset at intervene_location.
        Returns: {"output_ids": tensor, "input_ids": tensor}
        
        Requires: examples/utils.py and examples/federico_visualization.py to be available.
        """
        # Import utilities when needed
        try:
            from utils import (
                prepare_input_ids,
                prepare_intervention_activation,
                create_intervention_hook,
            )
        except ImportError:
            import sys
            import os
            examples_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'examples')
            if examples_path not in sys.path:
                sys.path.insert(0, examples_path)
            try:
                from utils import (
                    prepare_input_ids,
                    prepare_intervention_activation,
                    create_intervention_hook,
                )
            except ImportError:
                raise ImportError("generate_intervene requires examples/utils.py to be available. Make sure you're running from the examples directory or examples is in your Python path.")
        
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
