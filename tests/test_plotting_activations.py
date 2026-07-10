"""Tests for the activation-projection plot.

Covers both activation stores it accepts, the supervised and unsupervised
reducer paths, and the guard that rejects stores it cannot interpret. The
pipeline-fed cases use the shared tiny local model from conftest, so nothing is
downloaded.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("plotly")
pytest.importorskip("sklearn")

from murano import Pipeline
from murano.dataset import LabeledDataset, MuranoDataset
from murano.plotting import plot_activation_projection
from murano.steps.load import Load
from murano.steps.record import (
    ActivationStore,
    LabeledActivationStore,
    Record,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def contrastive_dataset():
    return MuranoDataset(
        positive_texts=["hello world", "good world"],
        negative_texts=["bad world", "bad prompt"],
    )


@pytest.fixture
def labeled_dataset():
    return LabeledDataset(
        texts=["hello world", "good world", "bad world", "bad prompt"],
        labels=[0, 0, 1, 1],
        label_names=["good", "bad"],
    )


@pytest.fixture
def separable_store():
    """Two well-separated blobs, so a projection has real structure to find."""
    torch.manual_seed(0)
    return ActivationStore(
        positive={(0, "residual"): torch.randn(8, 16) + 3.0},
        negative={(0, "residual"): torch.randn(8, 16) - 3.0},
        position="last",
    )


def _pca(n_components: int = 2):
    from sklearn.decomposition import PCA

    return PCA(n_components=n_components)


def _lda(n_components: int = 1):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    return LinearDiscriminantAnalysis(n_components=n_components)


# ── Store types ───────────────────────────────────────────────────────


def test_contrastive_store_yields_one_trace_per_class(separable_store):
    fig = plot_activation_projection(separable_store, 0, _pca())

    assert [trace.type for trace in fig.data] == ["scatter", "scatter"]
    assert [trace.name for trace in fig.data] == ["positive", "negative"]
    assert len(fig.data[0].x) == 8


def test_labeled_store_uses_label_names():
    store = LabeledActivationStore(
        activations={(0, "residual"): torch.randn(6, 8)},
        labels=torch.tensor([0, 0, 0, 1, 1, 1]),
        position="last",
    )
    fig = plot_activation_projection(store, 0, _pca(), label_names=["good", "bad"])

    assert [trace.name for trace in fig.data] == ["good", "bad"]


def test_labeled_store_without_names_falls_back_to_label_ints():
    store = LabeledActivationStore(
        activations={(0, "residual"): torch.randn(4, 8)},
        labels=torch.tensor([0, 0, 1, 1]),
        position="last",
    )
    fig = plot_activation_projection(store, 0, _pca())

    assert [trace.name for trace in fig.data] == ["0", "1"]


def test_label_names_that_miss_a_label_raise_rather_than_index_error():
    """LabeledDataset does not force labels to be 0..n-1, so guard the lookup."""
    store = LabeledActivationStore(
        activations={(0, "residual"): torch.randn(4, 8)},
        labels=torch.tensor([0, 0, 5, 5]),
        position="last",
    )
    with pytest.raises(ValueError, match="label_names has 2 entries"):
        plot_activation_projection(store, 0, _pca(), label_names=["a", "b"])


def test_class_colors_do_not_depend_on_row_order():
    """Shuffling the dataset must not silently swap which class is which color."""
    torch.manual_seed(0)
    activations = torch.randn(4, 8)
    labels = torch.tensor([0, 0, 1, 1])
    order = torch.tensor([2, 3, 0, 1])

    def color_of_class(acts, labs):
        store = LabeledActivationStore(
            activations={(0, "residual"): acts}, labels=labs, position="last"
        )
        fig = plot_activation_projection(store, 0, _pca(), label_names=["pos", "neg"])
        return {trace.name: trace.marker.color for trace in fig.data}

    assert color_of_class(activations, labels) == color_of_class(
        activations[order], labels[order]
    )


def test_reducer_returning_one_dimensional_array_is_accepted():
    """Not every reducer returns a column vector for a single component."""

    class Flat:
        def fit_transform(self, X, y=None):
            return np.asarray(X)[:, 0]

    store = ActivationStore(
        positive={(0, "residual"): torch.randn(3, 4)},
        negative={(0, "residual"): torch.randn(3, 4)},
        position="last",
    )
    fig = plot_activation_projection(store, 0, Flat())

    assert len(fig.data) == 2
    assert fig.layout.yaxis.showticklabels is False


# ── Reducers ──────────────────────────────────────────────────────────


def test_supervised_reducer_receives_the_class_labels(separable_store):
    """LDA needs y; a one-component fit means it must have been supplied."""
    fig = plot_activation_projection(separable_store, 0, _lda(n_components=1))

    # One component: classes are fanned out on y purely to avoid overlap.
    assert set(np.asarray(fig.data[0].y)) == {0.0}
    assert set(np.asarray(fig.data[1].y)) == {1.0}
    assert fig.layout.yaxis.showticklabels is False


def test_two_component_reducer_plots_both_axes(separable_store):
    fig = plot_activation_projection(separable_store, 0, _pca(n_components=2))

    assert fig.layout.yaxis.title.text == "Component 2"
    assert fig.layout.yaxis.showticklabels is None  # not suppressed


# ── Numerics ──────────────────────────────────────────────────────────


def test_zero_rows_do_not_become_nan_under_normalization():
    """A zero activation has no direction; normalizing must not divide by zero."""
    store = ActivationStore(
        positive={(0, "residual"): torch.zeros(4, 8)},
        negative={(0, "residual"): torch.randn(4, 8)},
        position="last",
    )
    fig = plot_activation_projection(store, 0, _pca())

    assert not np.isnan(np.asarray(fig.data[0].x)).any()


def test_accepts_bf16_activations(separable_store):
    """bf16 has no numpy dtype, so the conversion must go through float32."""
    store = ActivationStore(
        positive={(0, "residual"): separable_store.positive[0].bfloat16()},
        negative={(0, "residual"): separable_store.negative[0].bfloat16()},
        position="last",
    )
    fig = plot_activation_projection(store, 0, _pca())

    assert len(fig.data) == 2


@pytest.mark.parametrize("address", [0, (0, "residual"), "L0.resid_post"])
def test_accepts_address_shorthand(separable_store, address):
    fig = plot_activation_projection(separable_store, address, _pca())

    assert fig.layout.title.text.endswith("(L0.resid_post)")


# ── Guards ────────────────────────────────────────────────────────────


def test_rejects_full_position_store():
    store = ActivationStore(
        positive={(0, "residual"): torch.randn(4, 3, 8)},
        negative={(0, "residual"): torch.randn(4, 3, 8)},
        position="none",
    )
    with pytest.raises(ValueError, match="reduced activation store"):
        plot_activation_projection(store, 0, _pca())


def test_rejects_per_head_store():
    store = ActivationStore(
        positive={(0, "self_attn"): torch.randn(4, 2, 4)},
        negative={(0, "self_attn"): torch.randn(4, 2, 4)},
        position="last",
        per_head=True,
    )
    with pytest.raises(ValueError, match="reduced activation store"):
        plot_activation_projection(store, (0, "self_attn"), _pca())


# ── Fed by a real Record pipeline ─────────────────────────────────────


def test_projects_a_recorded_contrastive_store(murano_model, contrastive_dataset):
    results = Pipeline(
        [
            Load(contrastive_dataset),
            Record(murano_model, layers=[0], position="last", batch_size=2),
        ]
    ).run()

    fig = plot_activation_projection(results["record"], 0, _pca())

    assert [trace.name for trace in fig.data] == ["positive", "negative"]


def test_projects_a_recorded_labeled_store(murano_model, labeled_dataset):
    """The same Record that feeds Probe also feeds this plot."""
    results = Pipeline(
        [
            Load(labeled_dataset),
            Record(murano_model, layers=[0], position="last", batch_size=2),
        ]
    ).run()
    store = results["record"]

    assert isinstance(store, LabeledActivationStore)
    fig = plot_activation_projection(
        store, 0, _lda(n_components=1), label_names=labeled_dataset.label_names
    )

    assert [trace.name for trace in fig.data] == ["good", "bad"]
