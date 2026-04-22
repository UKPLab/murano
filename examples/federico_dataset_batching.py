from typing import List, Union
from murano import LayerLocation, LogitLens, MuranoModel
from torch.utils.data import DataLoader
from datasets import Dataset
import torch


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
    example["input_ids"] = tokenizer(
        example["text"], return_tensors="pt", max_length=max_length
    )["input_ids"][0]
    return example


class BatchedMuranoModel(MuranoModel):
    def run_recording(
        self,
        input: Union[str, torch.Tensor, dict],
        locations: List[LayerLocation],
        **kwargs,
    ) -> dict:
        """
        Run the model with tracing enabled to record activations at specified locations.
        Computes a single forward pass for a batch of inputs.
        """
        activations = []
        layer_indices = []
        if isinstance(input, torch.Tensor):
            input_ids = input
        elif isinstance(input, dict):
            input_ids = input["input_ids"]
        else:
            raise ValueError(
                "Input must be a string, tensor, or dictionary with 'input_ids'."
            )

        with self.model.trace() as tracer:
            with tracer.invoke(input_ids, max_length=10, **kwargs):
                layers_list = list(self.model.transformer.h)

                # TODO: define more flexible locations
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

                        # Specify a different location here
                        # output = layer.output
                        # output = layer.attn.output
                        output = layer.mlp.output
                        if isinstance(output, tuple):
                            hidden_states = output[0]
                        else:
                            hidden_states = output

                        saved_output = hidden_states.save()
                        activations.append(saved_output)
                        layer_indices.append(layer_idx)
        # Could be a tuple (tensor, None) if output_attentions=False
        if isinstance(activations[0].value, tuple):
            activations = [act.value[0] for act in activations]
        else:
            activations = [act.value for act in activations]

        artifact = {
            "activations": activations,
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

    def run_task(
        self, dataset: Dataset, locations: List[LayerLocation], **kwargs
    ) -> dict:
        """
        Run the model on a dataset with tracing enabled to record activations
        at specified locations.
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
            "locations": locations,
        }
        for example in dataloader:
            input_ids = example["input_ids"]
            activation = self.run_recording(input_ids, locations, **kwargs)
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
    return (
        f"{n}{'th' if 11 <= n <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"
    )


data = [
    {
        "text": f"Mary is born on the {i + 1} of August.",
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
dataloader = DataLoader(
    processed_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
)

lens = LogitLens()
locations = [LayerLocation(layers=slice(None))]
# 12 Layers, 4 sentences per batch, 9 tokens per sentence, 768-dimensional embeddings, 4 batches
artifact = model.run_task(processed_dataset, locations)
pass
