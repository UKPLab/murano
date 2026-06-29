"""Tests for the clean/corrupt paired dataset and the LoadPaired step.

The type and loader are checked on synthetic data; the end-to-end flow runs the
tiny ``murano_model`` fixture so the clean and corrupt sides actually pass
through the model and feed the metric steps.
"""

from __future__ import annotations

import logging
import math

import pytest
import torch

from murano import Pipeline, keys
from murano.artifacts import PromptBatch
from murano.dataset import CleanCorruptDataset
from murano.results import Results
from murano.steps.logits import Logits
from murano.steps.metrics import (
    KLDivergenceStep,
    LogitDiffStep,
    RecoveredMetricStep,
)
from murano.steps.paired import LoadPaired
from murano.steps.prompts import LoadPrompts

CLEAN = ["hello world", "good world"]
CORRUPT = ["good world", "hello world"]


# ── CleanCorruptDataset ───────────────────────────────────────────────


class TestCleanCorruptDataset:
    def test_basic_fields_and_len(self):
        ds = CleanCorruptDataset(clean=CLEAN, corrupt=CORRUPT)
        assert ds.clean == CLEAN
        assert ds.corrupt == CORRUPT
        assert len(ds) == 2
        assert ds.correct is None and ds.incorrect is None
        assert ds.metadata == {}

    def test_repr_reports_pairs_and_answers(self):
        ds = CleanCorruptDataset(clean=CLEAN, corrupt=CORRUPT, correct=[5, 6])
        text = repr(ds)
        assert "pairs=2" in text and "answers=True" in text

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same number of prompts"):
            CleanCorruptDataset(clean=["a", "b"], corrupt=["a"])

    def test_correct_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="entries but there are 2 pairs"):
            CleanCorruptDataset(clean=CLEAN, corrupt=CORRUPT, correct=[5, 6, 7])

    def test_incorrect_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="entries but there are 2 pairs"):
            CleanCorruptDataset(clean=CLEAN, corrupt=CORRUPT, incorrect=[5, 6, 7])

    def test_tensor_answer_length_mismatch_raises(self):
        # A per-example tensor must match the pair count too, not just lists.
        with pytest.raises(ValueError, match="entries but there are 2 pairs"):
            CleanCorruptDataset(
                clean=CLEAN, corrupt=CORRUPT, correct=torch.tensor([1, 2, 3])
            )

    def test_shared_scalar_answer_allowed(self):
        # A single shared id applies to every pair; no length check.
        ds = CleanCorruptDataset(clean=CLEAN, corrupt=CORRUPT, correct=5, incorrect=7)
        assert ds.correct == 5 and ds.incorrect == 7

    def test_length_one_answer_broadcasts(self):
        # LogitDiffStep broadcasts a length-1 answer, so the type must allow it.
        ds = CleanCorruptDataset(clean=CLEAN, corrupt=CORRUPT, correct=[5])
        assert ds.correct == [5]

    def test_from_pairs_classmethod(self):
        ds = CleanCorruptDataset.from_pairs(CLEAN, CORRUPT, correct=[5, 6])
        assert isinstance(ds, CleanCorruptDataset)
        assert len(ds) == 2 and ds.correct == [5, 6]

    def test_from_pairs_without_answers(self):
        ds = CleanCorruptDataset.from_pairs(CLEAN, CORRUPT)
        assert ds.correct is None and ds.incorrect is None


# ── LoadPaired ────────────────────────────────────────────────────────


class TestLoadPaired:
    def test_writes_both_sides_and_dataset(self):
        ds = CleanCorruptDataset(clean=CLEAN, corrupt=CORRUPT)
        out = LoadPaired(ds)(Results())

        assert out["dataset"] is ds
        clean_batch = out["prompts"]
        corrupt_batch = out["corrupt_prompts"]
        assert isinstance(clean_batch, PromptBatch)
        assert isinstance(corrupt_batch, PromptBatch)
        assert clean_batch.prompts == CLEAN
        assert corrupt_batch.prompts == CORRUPT
        assert clean_batch.source == "dataset.clean"
        assert corrupt_batch.source == "dataset.corrupt"

    def test_carries_raw_prompts(self):
        ds = CleanCorruptDataset(
            clean=CLEAN,
            corrupt=CORRUPT,
            raw_clean=["raw a", "raw b"],
            raw_corrupt=["raw c", "raw d"],
        )
        out = LoadPaired(ds)(Results())
        assert out["prompts"].raw_prompts == ["raw a", "raw b"]
        assert out["corrupt_prompts"].raw_prompts == ["raw c", "raw d"]

    def test_expected_write_types(self):
        ds = CleanCorruptDataset(clean=CLEAN, corrupt=CORRUPT)
        types = LoadPaired(ds).expected_write_types()
        assert types["dataset"] is CleanCorruptDataset
        assert types["prompts"] is PromptBatch
        assert types["corrupt_prompts"] is PromptBatch


# ── Logits prompts_key ────────────────────────────────────────────────


class TestLogitsPromptsKey:
    def test_reads_alternate_prompt_key(self, murano_model):
        results = Results()
        results["corrupt_prompts"] = PromptBatch(prompts=["hello"])
        out = Logits(murano_model, prompts_key="corrupt_prompts")(results)
        assert out["final_logits"].shape[0] == 1
        assert out.get("final_logits") is not None

    def test_reads_declaration_follows_prompts_key(self, murano_model):
        step = Logits(murano_model, prompts_key="corrupt_prompts")
        assert step.reads == ["corrupt_prompts"]

    def test_default_prompt_key_unchanged(self, murano_model):
        step = Logits(murano_model)
        assert step.reads == ["prompts"]


# ── End to end (paired pipeline) ──────────────────────────────────────


class TestEndToEnd:
    def test_clean_vs_corrupt_metrics(self, murano_model):
        ds = CleanCorruptDataset(
            clean=CLEAN, corrupt=CORRUPT, correct=[5, 6], incorrect=[6, 5]
        )
        results = Pipeline(
            [
                LoadPaired(ds),
                Logits(murano_model),  # clean -> final_logits + attention_mask
                Logits(
                    murano_model,
                    prompts_key=keys.CORRUPT_PROMPTS,
                    logits_key=keys.CORRUPT_LOGITS,
                    mask_key=keys.CORRUPT_MASK,
                    targets=None,
                ),
                LogitDiffStep(
                    correct=ds.correct, incorrect=ds.incorrect, output_key="clean_ld"
                ),
                LogitDiffStep(
                    correct=ds.correct,
                    incorrect=ds.incorrect,
                    logits_key=keys.CORRUPT_LOGITS,
                    mask_key=keys.CORRUPT_MASK,
                    output_key="corrupt_ld",
                ),
                KLDivergenceStep(p_key=keys.FINAL_LOGITS, q_key=keys.CORRUPT_LOGITS),
            ]
        ).run()

        assert math.isfinite(results["clean_ld"].value)
        assert math.isfinite(results["corrupt_ld"].value)
        # The corrupt side must actually move the distribution: clean and corrupt
        # prompts differ here, so KL is positive and the two logit-diffs differ.
        # A no-op or mis-wired corrupt run would give KL == 0 and equal diffs.
        assert results["kl_div"].value > 0
        assert results["clean_ld"].value != pytest.approx(results["corrupt_ld"].value)

    def test_recovered_composes_over_three_scalars(self, murano_model):
        ds = CleanCorruptDataset(
            clean=CLEAN, corrupt=CORRUPT, correct=[5, 6], incorrect=[6, 5]
        )
        r = Pipeline(
            [
                LoadPaired(ds),
                Logits(murano_model),
                Logits(
                    murano_model,
                    prompts_key="corrupt_prompts",
                    logits_key="corrupt_logits",
                    mask_key="corrupt_mask",
                    targets=None,
                ),
                LogitDiffStep(
                    correct=ds.correct, incorrect=ds.incorrect, output_key="clean_ld"
                ),
                LogitDiffStep(
                    correct=ds.correct,
                    incorrect=ds.incorrect,
                    logits_key="corrupt_logits",
                    mask_key="corrupt_mask",
                    output_key="corrupt_ld",
                ),
                RecoveredMetricStep(
                    "clean_ld", "corrupt_ld", "clean_ld", output_key="recovered"
                ),
            ]
        ).run()
        # patched == clean -> recovered is 1.0 (or nan if clean == corrupt).
        value = r["recovered"].value
        assert value == pytest.approx(1.0) or math.isnan(value)

    def test_validate_passes(self, murano_model):
        ds = CleanCorruptDataset(clean=CLEAN, corrupt=CORRUPT)
        pipe = Pipeline(
            [
                LoadPaired(ds),
                Logits(
                    murano_model,
                    prompts_key="corrupt_prompts",
                    logits_key="corrupt_logits",
                ),
            ]
        )
        assert "corrupt_logits" in pipe.validate()

    def test_chain_without_corrupt_prompts_fails(self, murano_model):
        pipe = Pipeline(
            [
                Logits(murano_model, prompts_key="corrupt_prompts"),
            ]
        )
        with pytest.raises(KeyError, match="corrupt_prompts"):
            pipe.validate()

    def test_duplicate_write_key_warns(self, murano_model, caplog):
        # Two Logits with default output keys both write final_logits: the second
        # silently overwrites the first, so the pipeline must warn rather than
        # leave a confident-but-wrong clean-vs-corrupt comparison.
        pipe = Pipeline(
            [LoadPrompts(["hi"]), Logits(murano_model), Logits(murano_model)]
        )
        with caplog.at_level(logging.WARNING, logger="murano"):
            pipe.validate()
        assert any(
            "final_logits" in r.message and "overwritten" in r.message
            for r in caplog.records
        )
