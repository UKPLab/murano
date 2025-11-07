from typing import List, Union
from murano import LayerLocation, LogitLens, MuranoModel
from murano import Location
from torch.utils.data import DataLoader
from datasets import Dataset
import torch
import random
import os
import numpy as np
import torch
import pandas as pd
import plotly.express as px
from typing import Union, List
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

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
    """
    Location specifies a slice of model activations to extract or analyze.
    """
    def __init__(self, layers: Union[int, List[int]], modules: Union[str, List[str]] = "mlp",
                 token_pos: Union[int, List[int]] = None):
        self.layers = layers if isinstance(layers, list) else [layers]
        self.modules = modules if isinstance(modules, list) else [modules]
        self.token_pos = token_pos if isinstance(token_pos, list) else [token_pos]
        # TODO: implement keyword based indexing for token_pos

    def __repr__(self):
        return f"(layers={self.layers}, modules={self.modules}, token_pos={self.token_pos})"
    
    def get_slice_idx(self, _slice):
        """
        Convert a subset of a Location (slice) into indices for numpy array slicing.
        """
        if not self.is_valid_slice(_slice):
            raise ValueError(f"Slice {_slice} is not included within location {self}.")

        # Get indices for slicing activations based on slice Location
        layer_idx = [self.layers.index(l) for l in _slice.layers]
        module_idx = [self.modules.index(m) for m in _slice.modules]
        token_idx = [self.token_pos.index(p) for p in _slice.token_pos]
        return (slice(None), layer_idx, module_idx, token_idx, slice(None))

    def is_valid_slice(self, slice: Location) -> bool:
        """
        Check if the slice is contained within this location.
        """
        return (all(l in self.layers for l in slice.layers) and
                all(m in self.modules for m in slice.modules) and
                all(p in self.token_pos for p in slice.token_pos))


class ActivationDataset:
    """
    ActivationDataset stores model activations and associated metadata.
    Internally uses NumPy for storage.
    """

    def __init__(self, activations: Union[np.ndarray, torch.Tensor],
                 location: Location,
                 global_metadata: dict, 
                 dataset: Dataset):
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

    def __getitem__(self, slice: Location):
        """
        Expected indexing order: 
            [examples, layers, modules, tokens, hidden_dim]
        Layers, modules, tokens may be strings or lists of strings.
        """
        idx = self.location.get_slice_idx(slice)
        return self.activations[idx].squeeze()

    def _to_numpy(self, activations: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(activations, torch.Tensor):
            return activations.detach().cpu().numpy()
        elif isinstance(activations, np.ndarray):
            return activations
        else:
            raise TypeError(f"Activations must be a numpy array or torch tensor, got {type(activations)}.")

# TODO: create base class
class PlotterLens:
    """
    Class for plotting model activations using an sklearn dimensionality reduction method.

    Args:
        reducer: Any sklearn-style object supporting fit_transform(X[, y]).
                 Examples: PCA(n_components=2), TSNE(), LDA(), UMAP().
        save_path: Path to save the resulting Plotly figure (HTML or image file).
        label_columns: Optional column names from dataset to use as supervised labels.
    """

    def __init__(self, reducer, save_path: str, label_columns: Union[str, List[str]] = None):
        self.reducer = reducer
        self.save_path = save_path
        self.label_columns = [label_columns] if isinstance(label_columns, str) else label_columns

    def observe(self, artifact: ActivationDataset, location, title: str = "Activation Visualization"):
        """
        Observe activations by applying dimensionality reduction and plotting.

        Args:
            artifact: ActivationDataset object containing activations and metadata.
            location: Location object specifying which layers/modules/tokens to extract.
            title: Plot title.
        """
        dataset = artifact.dataset

        # Convert dataset to DataFrame if needed
        df = pd.DataFrame(dataset)

        # Extract feature matrix
        X = artifact[location]
        X = Normalizer().fit_transform(X)

        # If supervised and labels are provided
        if self.label_columns is not None:
            if len(self.label_columns) == 1:
                y = df[self.label_columns[0]]
            else:
                y = df[self.label_columns].apply(lambda row: tuple(row.values), axis=1)

            reduced = self.reducer.fit_transform(X, y)
        else:
            reduced = self.reducer.fit_transform(X)

        # Build result DataFrame
        reduced_df = pd.DataFrame(reduced, columns=[f"Component {i+1}" for i in range(reduced.shape[1])])
        reduced_df = pd.concat([reduced_df, df], axis=1)

        # Choose coloring column
        color_col = self.label_columns[0] if self.label_columns else None

        # Create Plotly scatter plot
        fig = px.scatter(
            reduced_df,
            x="Component 1",
            y="Component 2",
            color=color_col,
            hover_data=df.columns,
            title=title,
            labels={"Component 1": "Dim 1", "Component 2": "Dim 2"},
        )

        fig.update_traces(marker=dict(size=6, opacity=0.7))
        fig.update_layout(width=800, height=800)
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        # Save
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        if self.save_path.endswith(".html"):
            fig.write_html(self.save_path)
        else:
            fig.write_image(self.save_path)

        print(f"Plot saved to {self.save_path}")
        return fig



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
                        if module == "output":
                            layer_module = layers_list[layer]
                            hidden_states = layer_module.output[0][ :, location.token_pos, :] if location.token_pos is not None else layer_module.output[0]
                        else:
                            layer_module = getattr(layers_list[layer], module)
                            hidden_states = layer_module.output[ :, location.token_pos, :] if location.token_pos is not None else layer_module.output

                        module_activation = hidden_states.save()
                        layer_activation.append(module_activation)
                    activations.append(layer_activation)

        artifact = {
            "activations": activations, # nested list of shape: (num_layers, num_modules, batch_size, seq_len, hidden_dim)
            "input_ids": input_ids,  # type: ignore
        }

        return artifact
    
    def _stack_activations(self, obj, location: Location) -> torch.Tensor:
        """
        Utility function that reshapes activations to
        (num_examples, num_layers, seq_len, hidden_dim)
        Necessary because activations are returned in heterogeneous nested structures.
        """
        obj = self._stack_activations_recursive(obj)
        print(f"Stacked activations shape before reshape: {obj.shape}")
        # Put batch and number of batches as first two dimensions
        obj = obj.permute(0, 3, 1, 2, 4, 5) # (num_batches, num_layers, num_modules, batch_size, seq_len, hidden_dim)
        obj = obj.reshape(obj.shape[0] * obj.shape[1], obj.shape[2], obj.shape[3], obj.shape[4], obj.shape[5])
         # (num_examples, num_layers, num_modules, seq_len, hidden_dim)
        assert obj.shape[1] == len(location.layers), f"Expected {len(location.layers)} layers, got {obj.shape[1]}"
        assert obj.shape[2] == len(location.modules), f"Expected {len(location.modules)} modules, got {obj.shape[2]}"
        assert obj.shape[3] == len(location.token_pos), f"Expected {len(location.token_pos)} positions, got {obj.shape[3]}"
        assert obj.shape[4] == self.model.config.hidden_size, f"Expected {self.model.config.hidden_size} hidden size, got {obj.shape[4]}"
        assert obj.dim() == 5, f"Expected 5 dimensions, got {obj.dim()}"
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

    # TODO: move this in the pipeline logic. MuranoModel should not know about datasets.
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
        activations = self._stack_activations(activations, location)
        artifact = ActivationDataset(
            activations=activations,
            location=location,
            global_metadata=global_metadata,
            dataset=dataset
        )
        return artifact
        

# Helper function to convert integers to ordinal strings
def ordinal(n):
    return f"{n}{'th' if 11<=n<=13 else {1:'st', 2:'nd', 3:'rd'}.get(n%10, 'th')}"


model = BatchedMuranoModel.from_pretrained("gpt2")
tokenizer = model.model.tokenizer

# Load as pandas dataframe
df = pd.read_csv('examples/dates.csv')
# Add day of year int column
df['month'] = pd.to_datetime(df['date']).dt.month_name()
toy_dataset = Dataset.from_pandas(df)
processed_dataset = toy_dataset.map(
    lambda x: process_dataset(x, tokenizer),
    batched=False,
)
processed_dataset.set_format(type="torch")
dataloader = DataLoader(processed_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

lens = LogitLens()
location = Location(layers=[2, 6, 11], modules=["output", "mlp"], token_pos=-1)
# 3 Layers, 2 modules, last token
artifact = model.run_task(processed_dataset, location)

# Initialize plotter
plotter = PlotterLens(
    reducer=LDA(n_components=2),
    save_path="plots/activation_pca.html",
    label_columns="month",
)

plot_slice = Location(layers=[6], modules=["output"], token_pos=-1)

# Observe activations
fig = plotter.observe(artifact, plot_slice, title="PCA of Layer 6 Activations")
fig.show()