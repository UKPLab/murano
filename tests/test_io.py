"""Save/load round-trips for activation stores, and the warn-on-skip behavior.

Runs on CPU with synthetic data, no model loading.
"""

import json
import logging

import torch

from murano.dataset import MuranoDataset
from murano.io import (
    load_activation_store,
    load_labeled_activation_store,
    save_results,
)
from murano.results import Results
from murano.steps.record import ActivationStore, LabeledActivationStore


def _activation_store() -> ActivationStore:
    return ActivationStore(
        positive={0: torch.randn(3, 8), 1: torch.randn(3, 8)},
        negative={0: torch.randn(3, 8), 1: torch.randn(3, 8)},
    )


def test_save_writes_and_reloads_activation_store(tmp_path):
    store = _activation_store()
    results = Results()
    results["record"] = store
    save_results(results, output_dir=str(tmp_path))

    path = tmp_path / "activations" / "record.pt"
    assert path.exists(), (
        f"activation store not saved; found "
        f"{sorted(p.name for p in tmp_path.iterdir())}"
    )

    loaded = load_activation_store(path)
    assert loaded.positive.keys() == store.positive.keys()
    for k in store.positive:
        assert torch.equal(loaded.positive[k], store.positive[k])
    for k in store.negative:
        assert torch.equal(loaded.negative[k], store.negative[k])

    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["record"]["kind"] == "contrastive"
    assert meta["record"]["n_positive"] == 3
    assert meta["record"]["n_negative"] == 3


def test_save_writes_and_reloads_labeled_activation_store(tmp_path):
    store = LabeledActivationStore(
        activations={0: torch.randn(4, 8)},
        labels=torch.tensor([0, 1, 0, 1]),
    )
    results = Results()
    results["record"] = store
    save_results(results, output_dir=str(tmp_path))

    path = tmp_path / "activations" / "record.pt"
    assert path.exists()

    loaded = load_labeled_activation_store(path)
    assert torch.equal(loaded.labels, store.labels)
    assert torch.equal(loaded.activations[0], store.activations[0])

    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["record"]["kind"] == "labeled"
    assert meta["record"]["n_examples"] == 4


def test_save_warns_on_unregistered_artifact(tmp_path, caplog):
    class Mystery:
        pass

    results = Results()
    results["mystery"] = Mystery()
    with caplog.at_level(logging.WARNING, logger="murano"):
        save_results(results, output_dir=str(tmp_path))

    assert any("mystery" in r.message for r in caplog.records)
    assert not (tmp_path / "mystery.pt").exists()


def test_save_does_not_warn_on_dataset_or_transient(tmp_path, caplog):
    from pathlib import Path

    results = Results()
    results["dataset"] = MuranoDataset(positive_texts=["a"], negative_texts=["b"])
    results["output_dir"] = Path("/tmp/somewhere")
    with caplog.at_level(logging.WARNING, logger="murano"):
        save_results(results, output_dir=str(tmp_path))

    warned = " ".join(r.message for r in caplog.records)
    assert "dataset" not in warned
    assert "output_dir" not in warned
