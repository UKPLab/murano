"""Public API smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import murano
from murano.model import MuranoModel
from murano.steps.train import SteeringResult


def test_compliance_rate_matches_keyword_detector():
    rates = murano.compliance_rate(
        clean=["Sure, here's how."],
        ablated=["I'm sorry, I can't help with that."],
    )
    assert rates == {"clean": 1.0, "ablated": 0.0}


def test_model_generate_supports_single_and_batch(monkeypatch):
    model = object.__new__(MuranoModel)

    def fake_generate_single(
        text, fn=None, layers="all", modules="residual", gen_kwargs=None
    ):
        prefix = "modified" if fn is not None else "clean"
        return f"{prefix}:{text}"

    monkeypatch.setattr(model, "_generate_single", fake_generate_single)

    assert model.generate("hello") == "clean:hello"
    assert model.generate(["a", "b"]) == ["clean:a", "clean:b"]


def test_model_generate_supports_ablate(monkeypatch):
    model = object.__new__(MuranoModel)

    def fake_generate_single(
        text, fn=None, layers="all", modules="residual", gen_kwargs=None
    ):
        assert fn is not None
        return f"done:{text}"

    monkeypatch.setattr(model, "_generate_single", fake_generate_single)
    result = model.generate(
        "prompt",
        ablate=SteeringResult(
            direction_per_layer={0: torch.ones(4)},
            separation_scores={0: 1.0},
            best_layer=0,
        ),
    )
    assert result == "done:prompt"


def test_model_generate_rejects_conflicting_interventions():
    model = object.__new__(MuranoModel)
    with pytest.raises(ValueError, match="either 'ablate' or 'steer'"):
        model.generate("prompt", ablate={}, steer=({}, 1.0))


def test_readme_examples_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "examples" / "quick_prototype.py").exists()
    assert (root / "examples" / "refusal_direction.py").exists()
