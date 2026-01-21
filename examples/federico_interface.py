from typing import List, Union
from murano import MuranoModel
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, StandardScaler

from murano.utils import ActivationDataset, Location

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
def process_dataset(example, tokenizer, max_length=None):
    example["input_ids"] = tokenizer(example["text"], return_tensors="pt",
                                     max_length=max_length)["input_ids"][0]
    return example

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
        # TODO: implement function for extracting labels from artifact metadata
        dataset = artifact.dataset

        # Convert dataset to DataFrame if needed
        df = pd.DataFrame(dataset)

        # Extract feature matrix
        X = artifact[location]
        X = StandardScaler().fit_transform(X)

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


def build_dataloader(dataset: Dataset, batch_size: int = 4) -> DataLoader:
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
        shuffle=False,
        collate_fn=collate_fn,
    )


# TODO: move this in the pipeline logic. MuranoModel should not know about datasets.
def run_recording(model: MuranoModel, dataset: DataLoader, location: List[Location], **kwargs) -> dict:
    """
    Run the model on a dataset with tracing enabled to record activations 
    at specified location.
    Processes the dataset in batches by calling run_recording for each batch.
    """
    batch_size = kwargs.pop("batch_size", 4)
    dataloader = build_dataloader(dataset, batch_size)
    activations = []
    global_metadata = {
        "model_name": model.model_name,
        "tokenizer": model.model.tokenizer.name_or_path,
        "batch_size": batch_size,
        "location": location,
    }
    for example in dataloader:
        input_ids = example["input_ids"]
        activation = model.record(input_ids, location, **kwargs)
        activations.append(activation)
    activations = model._stack_activations(activations, location)
    artifact = ActivationDataset(
        activations=activations,
        location=location,
        global_metadata=global_metadata,
        dataset=dataset
    )
    return artifact

def run_recording_intervention(model: MuranoModel, dataset: Dataset, location_rec: List[Location], location_int: List[Location], **kwargs) -> dict:
    """
    Run the model on a dataset with tracing enabled to record activations 
    at specified location.
    Processes the dataset in batches by calling run_recording for each batch.
    """
    batch_size = kwargs.pop("batch_size", 4)
    dataloader = build_dataloader(dataset, batch_size)
    activations = []
    global_metadata = {
        "model_name": model.model_name,
        "tokenizer": model.model.tokenizer.name_or_path,
        "batch_size": batch_size,
        "location": location_rec,
    }
    intervention_placeholder = torch.zeros(size=(1, len(location_int.layers), len(location_int.modules), len(location_int.token_pos)))
    for example in dataloader:
        input_ids = example["input_ids"]
        activation = model.record_intervene(input_ids, location_intervention=location_int, 
                                            location_recording=location_rec, intervention_activation=intervention_placeholder, **kwargs)
        activations.append(activation)
    activations = model._stack_activations(activations, location_rec)
    artifact = ActivationDataset(
        activations=activations,
        location=location_rec,
        global_metadata=global_metadata,
        dataset=dataset
    )
    return artifact

if __name__ == "__main__":
    model = MuranoModel.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer = model.model.tokenizer

    # Load as pandas dataframe
    df = pd.read_csv('examples/dates.csv')
    # df = pd.read_csv('examples/numbered_data.csv')

    # Add day of year int column
    df['month'] = pd.to_datetime(df['date']).dt.month_name()
    toy_dataset = Dataset.from_pandas(df)
    processed_dataset = toy_dataset.map(
        lambda x: process_dataset(x, tokenizer),
        batched=False,
    )
    processed_dataset.set_format(type="torch")

    # Recording example
    # location = Location(layers=[1, 6, 11], modules=["mlp", "output"], token_pos=[-2])
    # # 3 Layers, 2 modules, last token
    # artifact_recording = run_recording(model, processed_dataset, location)
    # # Initialize plotter
    # plotter = PlotterLens(
    #     reducer=PCA(n_components=2),
    #     save_path="plots/activation_pca.html",
    #     label_columns="month",
    # )

    # plot_slice = Location(layers=[11], modules=["output"], token_pos=[-2])

    # # Observe activations
    # fig = plotter.observe(artifact_recording, plot_slice, title="PCA of Layer 6 Activations")
    # fig.show()

    # Intervention recording example
    location_rec = Location(layers=[7], modules=["output"], token_pos=[-2])
    location_int = Location(layers=[6], modules=["output"], token_pos=[-2])
    artifact_intervention = run_recording_intervention(model, processed_dataset, location_rec, location_int)
    # Initialize plotter
    plotter_int = PlotterLens(
        reducer=PCA(n_components=2),
        save_path="plots/activation_intervention_pca.html",
        label_columns="month",
    )
    plot_slice_int = Location(layers=[7], modules=["output"], token_pos=[-2])
    # Observe activations
    fig_int = plotter_int.observe(artifact_intervention, plot_slice_int, title="PCA of Layer 7 Activations after Intervention")
    fig_int.show()