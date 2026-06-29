"""Save/load round-trips for activation stores, and the warn-on-skip behavior.

Runs on CPU with synthetic data, no model loading.
"""

import json
import logging
import math

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


def test_metadata_records_paired_dataset(tmp_path):
    """A clean/corrupt paired dataset is summarized as provenance, not serialized."""
    from murano.dataset import CleanCorruptDataset

    results = Results()
    results["dataset"] = CleanCorruptDataset(
        clean=["a", "b"],
        corrupt=["c", "d"],
        correct=[1, 2],
        raw_clean=["a", "b"],
    )
    save_results(results, output_dir=str(tmp_path))

    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["dataset"]["type"] == "paired"
    assert meta["dataset"]["n_pairs"] == 2
    assert meta["dataset"]["has_answers"] is True
    assert meta["dataset"]["chat_templated"] is True


def test_two_prompt_batches_record_separate_provenance(tmp_path):
    """Clean and corrupt prompt batches each get their own metadata entry."""
    from murano.artifacts import PromptBatch

    results = Results()
    results["prompts"] = PromptBatch(prompts=["a", "b"], source="dataset.clean")
    results["corrupt_prompts"] = PromptBatch(
        prompts=["c", "d"], source="dataset.corrupt"
    )
    save_results(results, output_dir=str(tmp_path))

    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["prompts"]["source"] == "dataset.clean"
    assert meta["corrupt_prompts"]["source"] == "dataset.corrupt"
    assert (tmp_path / "prompts" / "prompts.json").exists()
    assert (tmp_path / "prompts" / "corrupt_prompts.json").exists()


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


def test_save_and_reload_evaluation_result(tmp_path):
    from murano.artifacts import EvaluationResult
    from murano.io import load_evaluation

    results = Results()
    results["logit_diff"] = EvaluationResult(
        metric_name="logit_diff",
        value=2.5,
        per_example=[2.0, 3.0],
        metadata={"logits_key": "final_logits", "positions": [3, 3]},
    )
    save_results(results, output_dir=str(tmp_path))

    path = tmp_path / "metrics" / "logit_diff.json"
    assert path.exists()
    loaded = load_evaluation(path)
    assert loaded.metric_name == "logit_diff"
    assert loaded.value == 2.5
    assert loaded.per_example == [2.0, 3.0]
    assert loaded.metadata["positions"] == [3, 3]


def test_evaluation_nan_is_valid_json_and_round_trips(tmp_path):
    from murano.artifacts import EvaluationResult
    from murano.io import load_evaluation, save_evaluation

    path = tmp_path / "metrics" / "recovered.json"
    save_evaluation(EvaluationResult(metric_name="recovered", value=float("nan")), path)

    # Must be strict (RFC-8259) JSON: parse_constant fires on bare NaN/Infinity.
    def _reject(token):
        raise ValueError(f"non-strict JSON token: {token}")

    json.loads(path.read_text(), parse_constant=_reject)

    # nan is stored as null and restored as nan.
    assert math.isnan(load_evaluation(path).value)
