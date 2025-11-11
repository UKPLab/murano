from typing import List, Union
from murano import LayerLocation, LogitLens, MuranoModel
from murano import Location
from torch.utils.data import DataLoader
from datasets import Dataset
import torch
import random

from murano.lenses.base_lens import BaseLens

# Custom collate function stitches separate inputs coming from dataset
def collate_fn(batch):
    collated = {}
    for key in batch[0]:
        values = [example[key] for example in batch]
        if key in ["input_ids", "attention_mask"]:  # tensorize only these
            collated[key] = torch.stack([torch.tensor(v) for v in values])
        else:
            collated[key] = values  # keep as list
    return collated

# Utility function to tokenize used in map dataset
def process_dataset(example, tokenizer, max_length=10):
    example["input_ids"] = tokenizer(example["text"], return_tensors="pt",
                                     max_length=max_length)["input_ids"][0]
    return example


class Location:
    def __init__(self, layers: Union[int, List[int]], modules: Union[str, List[str]] = "mlp",
                 token_pos: Union[int, List[int]] = None):
        self.layers = layers if isinstance(layers, list) else [layers]
        self.modules = modules if isinstance(modules, list) else [modules]
        # TODO: implement keyword based indexing for token_pos
        self.token_pos = token_pos

    def __repr__(self):
        return f"Location(layers={self.layers}, modules={self.modules}, token_pos={self.token_pos})"


class BatchedMuranoModel(MuranoModel):
    def run_recording(
        self, input: Union[str, torch.Tensor, dict], location: Location, **kwargs
    ) -> dict:
        """
        Run the model with tracing enabled to record activations at specified locations.
        Computes a single forward pass for a batch of inputs.
        """
        activations = []
        if isinstance(input, torch.Tensor):
            input_ids = input
        elif isinstance(input, dict):
            input_ids = input["input_ids"]
        else:
            raise ValueError("Input must be a string, tensor, or dictionary with 'input_ids'.")

        with self.model.trace() as tracer:
            with tracer.invoke(input_ids, max_length=10, **kwargs):
                
                # Perform nested indexing of modules and layers
                layers_list = list(self.model.transformer.h)

                for layer in location.layers:
                    layer_activation = []
                    for module in location.modules:
                        layer_module = getattr(layers_list[layer], module)
                        output = layer_module.output
                        if isinstance(output, tuple):
                            hidden_states = output[0][ :, location.token_pos, :] if location.token_pos is not None else output[0]
                        else:
                            hidden_states = output[ :, location.token_pos, :] if location.token_pos is not None else output

                        module_activation = hidden_states.save()
                        layer_activation.append(module_activation)
                    activations.append(layer_activation)

        artifact = {
            "activations": activations, # nested list of shape: (num_layers, num_modules, batch_size, seq_len, hidden_dim)
            "input_ids": input_ids,  # type: ignore
        }

        return artifact
    
    def _stack_activations(self, obj):
        """
        Utility function that reshapes activations to
        (num_examples, num_layers, seq_len, hidden_dim)
        Necessary because activations are returned in heterogeneous nested structures.
        """
        obj = self._stack_activations_recursive(obj)
        obj = obj.permute(0, 3, 1, 2, 4, 5)
        obj = obj.reshape(obj.shape[0] * obj.shape[1], obj.shape[2], obj.shape[3], obj.shape[4], obj.shape[5])
         # (num_examples, num_layers, num_modules, seq_len, hidden_dim)
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
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def run_task(self, dataset: Dataset, location: List[LayerLocation], **kwargs) -> dict:
        """
        Run the model on a dataset with tracing enabled to record activations 
        at specified location.
        Processes the dataset in batches by calling run_recording for each batch.
        """
        dataset = dataset.map(
            lambda x: process_dataset(x, self.model.tokenizer),
            batched=False,
        )
        batch_size = kwargs.get("batch_size", 4)
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
        activations = self._stack_activations(activations)
        artifact = {
            "activations": activations,
            "global_metadata": global_metadata,
            "dataset": dataset,
        }
        return artifact
        

# Helper function to convert integers to ordinal strings
def ordinal(n):
    return f"{n}{'th' if 11<=n<=13 else {1:'st', 2:'nd', 3:'rd'}.get(n%10, 'th')}"

data = [
    {
        "text": f"Mary is born on the {i+1} of August.",
        "label": i % 2,
        "metadata": {"id": i, "source": "synthetic"},
        "confidence": torch.tensor(i / 15.0),
    }
    for i in range(16)
]


model = BatchedMuranoModel.from_pretrained("gpt2")
tokenizer = model.model.tokenizer
toy_dataset = Dataset.from_list(data)
processed_dataset = toy_dataset.map(
    lambda x: process_dataset(x, tokenizer),
    batched=False,
)
processed_dataset.set_format(type="torch")
dataloader = DataLoader(processed_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

lens = LogitLens()
location = Location(layers=[2, 4, 6], modules=["mlp"], token_pos=None)
# 3 Layers, 1 module, all token positions
artifact = model.run_task(processed_dataset, location)
pass
