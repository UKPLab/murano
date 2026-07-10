"""Tests for the Logits step, its next-token targets, and the model quick API."""

from __future__ import annotations

import pytest
import torch

from murano import Pipeline
from murano.results import Results
from murano.steps.logits import Logits
from murano.steps.metrics import AccuracyStep, CrossEntropyLossStep
from murano.steps.prompts import LoadPrompts


def _tokenize(model, prompts):
    """Tokenize exactly as the Logits step does, for direct comparison."""
    return model.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_token_type_ids=False,
    )


# ── Logits step: shapes, dtype, raw-logit contract ────────────────────


class TestLogitsStep:
    def test_writes_logits_and_targets(self, murano_model):
        prompts = ["hello world", "good world"]
        results = Pipeline([LoadPrompts(prompts), Logits(murano_model)]).run()

        logits = results["final_logits"]
        targets = results["target_ids"]
        mask = results["attention_mask"]
        vocab = murano_model.tokenizer.vocab_size

        assert logits.ndim == 3
        assert logits.shape[0] == len(prompts)
        assert logits.shape[2] == vocab
        assert logits.dtype == torch.float32
        assert logits.device.type == "cpu"

        assert targets.shape == logits.shape[:2]
        assert targets.dtype == torch.long

        assert mask.shape == logits.shape[:2]
        assert mask.device.type == "cpu"

    def test_logits_are_raw_not_probabilities(self, murano_model):
        """final_logits must be the model's raw output, never softmaxed."""
        prompts = ["hello world"]
        results = Pipeline([LoadPrompts(prompts), Logits(murano_model)]).run()
        logits = results["final_logits"]

        # A probability distribution sums to 1 over the vocab axis; raw logits
        # do not. This is the guard against an accidental softmax in the path.
        sums = logits.sum(dim=-1)
        assert not torch.allclose(sums, torch.ones_like(sums), atol=1e-3)

        # And it is exactly the model's forward_logits output (same path).
        direct = murano_model.forward_logits(_tokenize(murano_model, prompts))
        assert torch.allclose(logits, direct)

    def test_custom_keys(self, murano_model):
        step = Logits(murano_model, logits_key="my_logits", targets_key="my_targets")
        results = step(_prompts_results(["hello"]))
        assert "my_logits" in results
        assert "my_targets" in results
        assert step.writes == ["my_logits", "attention_mask", "my_targets"]

    def test_targets_none_writes_logits_and_mask(self, murano_model):
        step = Logits(murano_model, targets=None)
        assert step.writes == ["final_logits", "attention_mask"]

        results = step(_prompts_results(["hello world"]))
        assert "final_logits" in results
        assert "attention_mask" in results
        assert "target_ids" not in results

    def test_invalid_targets_raises(self, murano_model):
        with pytest.raises(ValueError, match="targets must be 'next_token' or None"):
            Logits(murano_model, targets="bogus")


# ── Next-token target construction ────────────────────────────────────


class TestNextTokenTargets:
    def test_uneven_batch_right_padded(self):
        """Right-padded batch: shift by one, -100 at last real + padded slots."""
        # row 0: three real tokens; row 1: one real token then padding.
        input_ids = torch.tensor([[4, 5, 6], [4, 0, 0]])
        attention_mask = torch.tensor([[1, 1, 1], [1, 0, 0]])

        targets = Logits._next_token_targets(input_ids, attention_mask)

        expected = torch.tensor([[5, 6, -100], [-100, -100, -100]])
        assert torch.equal(targets, expected)
        assert targets.dtype == torch.long

    def test_length_one_batch_all_ignored(self):
        """A length-1 sequence has no next token anywhere."""
        targets = Logits._next_token_targets(torch.tensor([[4]]), torch.tensor([[1]]))
        assert torch.equal(targets, torch.tensor([[-100]]))

    def test_left_padding_handled(self):
        """Validity comes from the mask, so left padding works too."""
        input_ids = torch.tensor([[0, 0, 4, 5]])
        attention_mask = torch.tensor([[0, 0, 1, 1]])

        targets = Logits._next_token_targets(input_ids, attention_mask)

        # Only position 2 has a valid next token (id 5); the rest are ignored.
        assert torch.equal(targets, torch.tensor([[-100, -100, 5, -100]]))


# ── End to end: the previously dead metric keys now run ────────────────


class TestLogitsFeedsMetrics:
    def test_logits_then_cross_entropy(self, murano_model):
        results = Pipeline(
            [
                LoadPrompts(["hello world", "good world"]),
                Logits(murano_model),
                CrossEntropyLossStep(),
            ]
        ).run()

        loss = results["loss"]
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_logits_then_accuracy(self, murano_model):
        results = Pipeline(
            [
                LoadPrompts(["hello world", "good world"]),
                Logits(murano_model),
                AccuracyStep(),
            ]
        ).run()

        acc = results["accuracy"]
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_length_one_prompt_gives_zero_loss(self, murano_model):
        """All targets are -100 here; the loss guard must return 0.0, not NaN."""
        results = Pipeline(
            [LoadPrompts(["hello"]), Logits(murano_model), CrossEntropyLossStep()]
        ).run()
        assert results["loss"].item() == 0.0

    def test_full_metric_chain_validates(self, murano_model):
        pipe = Pipeline(
            [
                LoadPrompts(["hello world"]),
                Logits(murano_model),
                CrossEntropyLossStep(),
                AccuracyStep(),
            ]
        )
        produced = pipe.validate()
        assert "loss" in produced
        assert "accuracy" in produced

    def test_metric_chain_without_logits_fails(self, murano_model):
        """Dropping Logits leaves final_logits unwritten — validation must fail."""
        pipe = Pipeline([LoadPrompts(["hello world"]), CrossEntropyLossStep()])
        with pytest.raises(KeyError, match="final_logits"):
            pipe.validate()


# ── model.logits quick API ────────────────────────────────────────────


class TestModelLogitsQuickApi:
    def test_single_text(self, murano_model):
        logits = murano_model.logits("hello world")
        assert logits.shape[0] == 1
        assert logits.shape[2] == murano_model.tokenizer.vocab_size
        assert logits.dtype == torch.float32

    def test_batch_text(self, murano_model):
        logits = murano_model.logits(["hello world", "good"])
        assert logits.shape[0] == 2


def _prompts_results(prompts) -> Results:
    """Run LoadPrompts so a step can be exercised outside a full Pipeline."""
    return LoadPrompts(prompts)(Results())


# ── Intervened forward pass ───────────────────────────────────────────


class TestInterventionPassthrough:
    """``Logits(fn=...)`` is the forward-pass analogue of ``Intervene``."""

    def test_no_fn_leaves_the_logits_unmodified(self, murano_model):
        plain = Logits(murano_model)(_prompts_results(["hello world"]))["final_logits"]
        explicit = Logits(murano_model, fn=None, layers=[0])(
            _prompts_results(["hello world"])
        )["final_logits"]
        assert torch.equal(plain, explicit)

    def test_fn_changes_the_logits(self, murano_model):
        plain = Logits(murano_model)(_prompts_results(["hello world"]))["final_logits"]
        zeroed = Logits(
            murano_model, fn=lambda activation, _node: activation * 0.0, layers=[0]
        )(_prompts_results(["hello world"]))["final_logits"]
        assert not torch.equal(plain, zeroed)

    def test_matches_the_backend_primitive(self, murano_model):
        prompts = ["hello world", "good"]

        def halve(activation, _node):
            return activation * 0.5

        via_step = Logits(murano_model, fn=halve, layers=[0], targets=None)(
            _prompts_results(prompts)
        )["final_logits"]
        via_backend = murano_model.forward_logits(
            _tokenize(murano_model, prompts), fn=halve, layers=[0], modules="residual"
        )
        assert torch.equal(via_step, via_backend)

    def test_layers_selects_where_the_edit_lands(self, murano_model):
        # An additive edit, not a zeroing one: the fixture is a bias-free Llama,
        # so zeroing any layer's residual annihilates every layer after it and
        # zeroing layer 0 or layer 1 both yield all-zero logits.
        def shift(activation, _node):
            return activation + 1.0

        first = Logits(murano_model, fn=shift, layers=[0])(
            _prompts_results(["hello world"])
        )["final_logits"]
        second = Logits(murano_model, fn=shift, layers=[1])(
            _prompts_results(["hello world"])
        )["final_logits"]
        assert not torch.equal(first, second)

    def test_still_writes_the_mask_and_targets(self, murano_model):
        out = Logits(murano_model, fn=lambda activation, _node: activation)(
            _prompts_results(["hello world"])
        )
        assert "attention_mask" in out and "target_ids" in out
