"""Tests for the forward-pass evaluation metrics.

Mostly synthetic logits on CPU so the metric math is checked directly; a few
integration tests run the tiny ``murano_model`` fixture through the pipeline.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from murano import Pipeline
from murano.artifacts import MetricScore
from murano.results import Results
from murano.steps.logits import Logits
from murano.steps.metrics import (
    AnswerLogProbStep,
    KLDivergenceStep,
    LogitDiffStep,
    RecoveredMetricStep,
)
from murano.steps.prompts import LoadPrompts


def _results(logits: torch.Tensor, mask: torch.Tensor | None = None) -> Results:
    r = Results()
    r["final_logits"] = logits
    if mask is not None:
        r["attention_mask"] = mask
    return r


# ── LogitDiffStep ─────────────────────────────────────────────────────


class TestLogitDiffStep:
    def test_positive_when_correct_token_is_higher(self):
        logits = torch.zeros(2, 3, 10)
        logits[0, -1, 1] = 5.0
        logits[0, -1, 3] = 1.0
        logits[1, -1, 2] = 3.0
        logits[1, -1, 4] = 0.0

        out = LogitDiffStep(correct=[1, 2], incorrect=[3, 4])(_results(logits))
        result = out["logit_diff"]

        assert isinstance(result, MetricScore)
        assert result.value == pytest.approx(3.5)  # mean of 4.0 and 3.0
        assert result.per_example == pytest.approx([4.0, 3.0])

    def test_matches_manual_gather(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 5, 10)
        correct = [1, 2, 3, 4]
        incorrect = [5, 6, 7, 8]

        out = LogitDiffStep(correct=correct, incorrect=incorrect)(_results(logits))

        last = logits[:, -1, :]
        rows = torch.arange(4)
        manual = last[rows, torch.tensor(correct)] - last[rows, torch.tensor(incorrect)]
        assert out["logit_diff"].value == pytest.approx(float(manual.mean()))

    def test_answer_position_from_mask(self):
        # Right-padded row: the real answer sits at index 1, not the last column.
        logits = torch.zeros(1, 4, 10)
        logits[0, 1, 1] = 5.0  # correct high at the real answer position
        logits[0, 3, 2] = 5.0  # incorrect high in the padded tail
        mask = torch.tensor([[1, 1, 0, 0]])

        with_mask = LogitDiffStep(correct=1, incorrect=2)(_results(logits, mask))
        assert with_mask["logit_diff"].value == pytest.approx(5.0)

        # Without a mask the step falls back to the last column (the pad tail).
        without_mask = LogitDiffStep(correct=1, incorrect=2)(_results(logits))
        assert without_mask["logit_diff"].value == pytest.approx(-5.0)

    def test_explicit_positions_override(self):
        logits = torch.zeros(1, 4, 10)
        logits[0, 0, 1] = 2.0
        out = LogitDiffStep(correct=1, incorrect=2, positions=[0])(_results(logits))
        assert out["logit_diff"].value == pytest.approx(2.0)

    def test_token_set_is_averaged(self):
        logits = torch.zeros(1, 1, 10)
        logits[0, 0, 1] = 2.0
        logits[0, 0, 2] = 4.0
        logits[0, 0, 3] = 1.0
        out = LogitDiffStep(
            correct=torch.tensor([[1, 2]]), incorrect=torch.tensor([[3]])
        )(_results(logits))
        # mean(2, 4) - 1 = 2.0
        assert out["logit_diff"].value == pytest.approx(2.0)

    def test_string_answers_are_tokenized(self, murano_model):
        vocab = murano_model.tokenizer.vocab_size
        logits = torch.zeros(1, 1, vocab)
        logits[0, 0, 5] = 3.0  # "world"
        logits[0, 0, 7] = 1.0  # "bad"
        out = LogitDiffStep(correct="world", incorrect="bad", model=murano_model)(
            _results(logits)
        )
        assert out["logit_diff"].value == pytest.approx(2.0)

    def test_out_of_range_token_raises(self):
        logits = torch.zeros(1, 1, 10)
        with pytest.raises(ValueError, match="must be in"):
            LogitDiffStep(correct=100, incorrect=1)(_results(logits))

    def test_shared_token_set_broadcasts_to_batch(self):
        # One shared [1, k] set must score every example, not just row 0.
        logits = torch.zeros(3, 1, 10)
        logits[:, 0, 1] = torch.tensor([2.0, 4.0, 6.0])
        logits[:, 0, 2] = 1.0
        out = LogitDiffStep(
            correct=torch.tensor([[1, 2]]), incorrect=torch.tensor([[3]])
        )(_results(logits))
        # per row: mean(l1, l2) - l3 = mean(x, 1) - 0
        assert out["logit_diff"].per_example == pytest.approx([1.5, 2.5, 3.5])

    def test_batch_length_mismatch_raises(self):
        logits = torch.zeros(3, 1, 10)
        with pytest.raises(ValueError, match="entries but the batch"):
            LogitDiffStep(correct=[5, 0], incorrect=[3, 7])(_results(logits))

    def test_out_of_range_position_raises(self):
        logits = torch.zeros(1, 4, 10)
        with pytest.raises(ValueError, match="index"):
            LogitDiffStep(correct=1, incorrect=2, positions=[7])(_results(logits))

    def test_2d_positions_raises(self):
        logits = torch.zeros(1, 4, 10)
        with pytest.raises(ValueError, match="0-D or 1-D"):
            LogitDiffStep(correct=1, incorrect=2, positions=[[1, 2]])(_results(logits))


# ── KLDivergenceStep ──────────────────────────────────────────────────


class TestKLDivergenceStep:
    def test_identical_distributions_zero(self):
        torch.manual_seed(1)
        logits = torch.randn(3, 4, 10)
        r = Results()
        r["p"] = logits
        r["q"] = logits.clone()
        out = KLDivergenceStep(p_key="p", q_key="q")(r)
        assert out["kl_div"].value == pytest.approx(0.0, abs=1e-6)

    def test_matches_manual(self):
        torch.manual_seed(2)
        p = torch.randn(2, 3, 10)
        q = torch.randn(2, 3, 10)
        r = Results()
        r["p"] = p
        r["q"] = q
        out = KLDivergenceStep(p_key="p", q_key="q")(r)

        log_p = F.log_softmax(p[:, -1, :], dim=-1)
        log_q = F.log_softmax(q[:, -1, :], dim=-1)
        manual = (log_p.exp() * (log_p - log_q)).sum(-1).mean()
        assert out["kl_div"].value == pytest.approx(float(manual))

    def test_direction_matters(self):
        torch.manual_seed(3)
        p = torch.randn(2, 1, 10)
        q = torch.randn(2, 1, 10)
        r = Results()
        r["p"] = p
        r["q"] = q
        pq = KLDivergenceStep(p_key="p", q_key="q", output_key="pq")(r)["pq"].value
        qp = KLDivergenceStep(p_key="q", q_key="p", output_key="qp")(r)["qp"].value
        assert pq != pytest.approx(qp)


# ── AnswerLogProbStep ─────────────────────────────────────────────────


class TestAnswerLogProbStep:
    def test_matches_log_softmax(self):
        torch.manual_seed(4)
        logits = torch.randn(3, 2, 10)
        correct = [1, 2, 3]
        out = AnswerLogProbStep(correct=correct)(_results(logits))

        last = F.log_softmax(logits[:, -1, :], dim=-1)
        rows = torch.arange(3)
        manual = last[rows, torch.tensor(correct)].mean()
        assert out["answer_logprob"].value == pytest.approx(float(manual))

    def test_as_loss_is_negative_logprob(self):
        torch.manual_seed(5)
        logits = torch.randn(2, 2, 10)
        lp = AnswerLogProbStep(correct=[1, 2], output_key="lp")(_results(logits))
        nll = AnswerLogProbStep(correct=[1, 2], as_loss=True, output_key="nll")(
            _results(logits)
        )
        assert nll["nll"].value == pytest.approx(-lp["lp"].value)
        assert nll["nll"].metric_name == "answer_nll"

    def test_batch_length_mismatch_raises(self):
        logits = torch.zeros(3, 2, 10)
        with pytest.raises(ValueError, match="entries but the batch"):
            AnswerLogProbStep(correct=[5, 3])(_results(logits))


# ── RecoveredMetricStep ───────────────────────────────────────────────


class TestRecoveredMetricStep:
    def test_formula(self):
        r = Results()
        r["clean"] = MetricScore(metric_name="logit_diff", value=4.0)
        r["corrupted"] = MetricScore(metric_name="logit_diff", value=0.0)
        r["patched"] = MetricScore(metric_name="logit_diff", value=3.0)
        out = RecoveredMetricStep("clean", "corrupted", "patched")(r)
        assert out["recovered"].value == pytest.approx(0.75)

    def test_accepts_floats_and_tensors(self):
        r = Results()
        r["clean"] = 4.0
        r["corrupted"] = torch.tensor(0.0)
        r["patched"] = MetricScore(metric_name="x", value=3.0)
        out = RecoveredMetricStep("clean", "corrupted", "patched")(r)
        assert out["recovered"].value == pytest.approx(0.75)

    def test_zero_span_returns_nan(self):
        r = Results()
        r["clean"] = 2.0
        r["corrupted"] = 2.0
        r["patched"] = 2.0
        out = RecoveredMetricStep("clean", "corrupted", "patched")(r)
        assert math.isnan(out["recovered"].value)


# ── End to end: Logits feeds the metrics ──────────────────────────────


class TestEndToEnd:
    def test_logits_then_logit_diff(self, murano_model):
        results = Pipeline(
            [
                LoadPrompts(["hello world", "good world"]),
                Logits(murano_model),
                LogitDiffStep(correct=5, incorrect=7),
            ]
        ).run()
        result = results["logit_diff"]
        assert isinstance(result, MetricScore)
        assert math.isfinite(result.value)
        assert result.per_example is not None and len(result.per_example) == 2

    def test_full_chain_validates(self, murano_model):
        pipe = Pipeline(
            [
                LoadPrompts(["hello world"]),
                Logits(murano_model),
                LogitDiffStep(correct=5, incorrect=7),
            ]
        )
        produced = pipe.validate()
        assert "logit_diff" in produced

    def test_chain_without_logits_fails(self):
        pipe = Pipeline([LoadPrompts(["hello"]), LogitDiffStep(correct=5, incorrect=7)])
        with pytest.raises(KeyError, match="final_logits"):
            pipe.validate()
