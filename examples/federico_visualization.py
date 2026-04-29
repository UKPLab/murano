"""
Activation visualization using the Pipeline API.

Demonstrates:
  1. Loading a contrastive dataset via MuranoDataset.
  2. Recording activations with the Record step.
  3. Visualizing the resulting ActivationStore with PlotterLens
     (PCA / LDA dimensionality reduction + Plotly scatter plot).
"""

from __future__ import annotations

import os
from typing import List, Union

import pandas as pd
import plotly.express as px
import torch
from sklearn.preprocessing import Normalizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from murano import MuranoModel, Pipeline
from murano.dataset import MuranoDataset
from murano.steps.load import Load
from murano.steps.record import ActivationStore, Record
from murano.steps.train import SteeringVector


class PlotterLens:
    """Dimensionality reduction + Plotly scatter plot for ActivationStore.

    Args:
        reducer: Any sklearn-style object supporting ``fit_transform(X[, y])``.
                 Examples: PCA(n_components=2), LDA(), TSNE(), UMAP().
        save_path: Path to save the resulting Plotly figure (HTML or image).
    """

    def __init__(self, reducer, save_path: str):
        self.reducer = reducer
        self.save_path = save_path

    def observe(
        self,
        store: ActivationStore,
        layer: int,
        title: str = "Activation Visualization",
    ) -> None:
        """Reduce activations for a single layer and plot.

        Args:
            store: ActivationStore produced by the Record step.
            layer: Which layer index to visualise.
            title: Plot title.
        """
        pos = store.positive[layer]  # [N_pos, d_model]
        neg = store.negative[layer]  # [N_neg, d_model]

        X = torch.cat([pos, neg], dim=0).numpy()
        y = ["positive"] * len(pos) + ["negative"] * len(neg)

        X = Normalizer().fit_transform(X)

        # Fit reducer — pass labels for supervised methods (LDA)
        reduced = self.reducer.fit_transform(X, y)

        n_dims = reduced.shape[1]
        columns = [f"Component {i + 1}" for i in range(n_dims)]
        df = pd.DataFrame(reduced, columns=columns)
        df["label"] = y

        if n_dims == 1:
            # Single component — scatter along x with a constant y offset per label
            df["y_offset"] = df["label"].map({"positive": 0, "negative": 1})
            fig = px.scatter(
                df,
                x="Component 1",
                y="y_offset",
                color="label",
                title=title,
                labels={"Component 1": "Dim 1", "y_offset": ""},
            )
            fig.update_yaxes(showticklabels=False)
        else:
            fig = px.scatter(
                df,
                x="Component 1",
                y="Component 2",
                color="label",
                title=title,
                labels={"Component 1": "Dim 1", "Component 2": "Dim 2"},
            )
            fig.update_yaxes(scaleanchor="x", scaleratio=1)

        fig.update_traces(marker=dict(size=6, opacity=0.7))
        fig.update_layout(width=800, height=800)

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        if self.save_path.endswith(".html"):
            fig.write_html(self.save_path)
        else:
            fig.write_image(self.save_path)

        print(f"Plot saved to {self.save_path}")


def main():
    model = MuranoModel("gpt2")

    # Small synthetic contrastive dataset
    positive_texts = [
        "I love this product",
        "This is amazing",
        "What a wonderful day",
        "I am so happy",
        "This is fantastic",
    ]
    negative_texts = [
        "I hate this product",
        "This is terrible",
        "What a horrible day",
        "I am so sad",
        "This is awful",
    ]

    dataset = MuranoDataset(
        positive_texts=positive_texts,
        negative_texts=negative_texts,
    )

    # Pipeline: Load → Record (layer 6, last token)
    pipe = Pipeline(
        [
            Load(dataset),
            Record(model, layers=[6], position="last", batch_size=4),
        ]
    )
    results = pipe.run()

    store: ActivationStore = results["record"]

    # Visualise layer 6 with LDA
    plotter = PlotterLens(
        reducer=LDA(n_components=2),
        save_path="plots/activation_lda.html",
    )
    plotter.observe(store, layer=6, title="LDA of Layer 6 Activations")


if __name__ == "__main__":
    main()
