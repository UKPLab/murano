"""Smoke tests for the probing plots (previously untested public API)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("plotly")
pytest.importorskip("sklearn")

from sklearn.linear_model import LogisticRegression

from murano.nodes import Node, RESID_POST
from murano.plotting.probing import plot_confusion_matrix
from murano.steps.probe import ProbeResult
from murano.steps.record import LabeledActivationStore


def _labeled_store(n_per_class: int = 8, d: int = 6):
    # Two linearly separable clusters so the refit classifier is meaningful.
    g = torch.Generator().manual_seed(0)
    pos = torch.randn(n_per_class, d, generator=g) + 3.0
    neg = torch.randn(n_per_class, d, generator=g) - 3.0
    acts = torch.cat([pos, neg], dim=0)
    labels = torch.tensor([1] * n_per_class + [0] * n_per_class)
    return LabeledActivationStore(
        activations={Node(0, RESID_POST): acts},
        labels=labels,
        position="last",
        per_head=False,
    )


def _probe_result(store, refit: bool):
    node = Node(0, RESID_POST)
    classifiers = {}
    if refit:
        clf = LogisticRegression(max_iter=200)
        clf.fit(store.activations[node].numpy(), store.labels.numpy())
        classifiers[node] = clf
    return ProbeResult(
        accuracy_per_layer={node: 1.0},
        cv_scores={node: np.array([1.0, 1.0])},
        best_layer=node,
        classifiers=classifiers,
        label_names=["neg", "pos"],
    )


def test_plot_confusion_matrix_returns_figure_when_refit():
    store = _labeled_store()
    probe = _probe_result(store, refit=True)
    fig = plot_confusion_matrix(probe, store)
    assert fig is not None
    data = fig.to_dict()
    assert data["data"][0]["type"] == "heatmap"


def test_plot_confusion_matrix_returns_none_without_refit():
    store = _labeled_store()
    probe = _probe_result(store, refit=False)
    # No refitted classifier at the best layer -> None, not a crash.
    assert plot_confusion_matrix(probe, store) is None
