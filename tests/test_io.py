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
    save_activation_store,
    save_results,
)
from murano.results import Results
from murano.steps.record import ActivationStore, LabeledActivationStore


def _activation_store() -> ActivationStore:
    return ActivationStore(
        positive={
            (0, "residual"): torch.randn(3, 8),
            (1, "residual"): torch.randn(3, 8),
        },
        negative={
            (0, "residual"): torch.randn(3, 8),
            (1, "residual"): torch.randn(3, 8),
        },
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
        activations={(0, "residual"): torch.randn(4, 8)},
        labels=torch.tensor([0, 1, 0, 1]),
    )
    results = Results()
    results["record"] = store
    save_results(results, output_dir=str(tmp_path))

    path = tmp_path / "activations" / "record.pt"
    assert path.exists()

    loaded = load_labeled_activation_store(path)
    assert torch.equal(loaded.labels, store.labels)
    assert torch.equal(
        loaded.activations[(0, "residual")], store.activations[(0, "residual")]
    )

    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["record"]["kind"] == "labeled"
    assert meta["record"]["n_examples"] == 4


def test_activation_store_roundtrip_preserves_new_fields(tmp_path):
    """position / per_head / token masks survive a save+load round-trip."""
    store = ActivationStore(
        positive={(0, "self_attn"): torch.randn(2, 3, 4, 8)},
        negative={(0, "self_attn"): torch.randn(2, 3, 4, 8)},
        position="none",
        per_head=True,
        positive_token_mask=torch.ones(2, 3),
        negative_token_mask=torch.ones(2, 3),
    )
    path = tmp_path / "store.pt"
    save_activation_store(store, path)
    loaded = load_activation_store(path)

    assert loaded.position == "none"
    assert loaded.per_head is True
    assert torch.equal(loaded.positive_token_mask, store.positive_token_mask)
    assert torch.equal(loaded.negative_token_mask, store.negative_token_mask)
    assert torch.equal(
        loaded.positive[(0, "self_attn")], store.positive[(0, "self_attn")]
    )


def test_load_legacy_activation_store_payload_defaults(tmp_path):
    """A pre-redesign payload (no new keys) loads with backward-compat defaults."""
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "positive": {(0, "residual"): torch.randn(3, 8)},
            "negative": {(0, "residual"): torch.randn(3, 8)},
        },
        path,
    )
    loaded = load_activation_store(path)

    assert loaded.position == "last"
    assert loaded.per_head is False
    assert loaded.positive_token_mask is None
    assert loaded.negative_token_mask is None


def test_metadata_records_position_and_per_head(tmp_path):
    """The serializer summary should be self-describing about position/per_head."""
    store = _activation_store()
    results = Results()
    results["record"] = store
    save_results(results, output_dir=str(tmp_path))

    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["record"]["position"] == "last"
    assert meta["record"]["per_head"] is False


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
