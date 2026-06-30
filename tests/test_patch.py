"""Tests for the cross-run activation-patching step (Patch).

Patch is a thin preset over Ablate's resample engine, so these tests focus on its
contract (read/write keys and defaults), the cross-run direction, and the
denoising recovery wiring; the shared capture/blend/per-head/positions machinery
is covered in test_ablate.py. End-to-end behavior runs the tiny ``murano_model``
fixture on CPU.
"""

from __future__ import annotations

import math

import pytest
import torch

from murano import Pipeline, keys
from murano.dataset import CleanCorruptDataset
from murano.nodes import SELF_ATTN, Node, NodeSet
from murano.results import Results
from murano.steps.logits import Logits
from murano.steps.metrics import LogitDiffStep, RecoveredMetricStep
from murano.steps.paired import LoadPaired
from murano.steps.patch import Patch

# Equal token length per pair (two known vocab words, no BOS), so positions align.
CLEAN = ["hello world", "good world"]
CORRUPT = ["good world", "bad world"]


def _loaded(clean=CLEAN, corrupt=CORRUPT, correct=5, incorrect=6):
    """Build a CleanCorruptDataset and the Results that LoadPaired produces."""
    ds = CleanCorruptDataset(
        clean=clean, corrupt=corrupt, correct=correct, incorrect=incorrect
    )
    return ds, LoadPaired(ds)(Results())


def _clean_logits(model, results) -> torch.Tensor:
    return Logits(model)(results)["final_logits"]


def _corrupt_logits(model, results) -> torch.Tensor:
    return Logits(
        model,
        prompts_key=keys.CORRUPT_PROMPTS,
        logits_key=keys.CORRUPT_LOGITS,
        mask_key=keys.CORRUPT_MASK,
        targets=None,
    )(results)[keys.CORRUPT_LOGITS]


# ── Contract ──────────────────────────────────────────────────────────


class TestPatchContract:
    def test_default_read_write_keys(self, murano_model):
        step = Patch(murano_model, 0)
        assert step.reads == [keys.CORRUPT_PROMPTS, keys.PROMPTS]
        assert step.writes == [keys.PATCHED_LOGITS, keys.PATCHED_MASK]
        assert step.write_types == {
            keys.PATCHED_LOGITS: torch.Tensor,
            keys.PATCHED_MASK: torch.Tensor,
        }

    def test_writes_logits_and_base_mask(self, murano_model):
        _, results = _loaded()
        out = Patch(murano_model, 0)(results)
        logits = out[keys.PATCHED_LOGITS]
        assert logits.dtype == torch.float32
        assert logits.shape[0] == len(CORRUPT)
        # The mask describes the base (corrupt) batch the metric will score.
        expected = murano_model.tokenizer(
            CORRUPT,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_token_type_ids=False,
        )["attention_mask"]
        assert torch.equal(out[keys.PATCHED_MASK], expected)

    def test_per_head_autodetect(self, murano_model):
        assert Patch(murano_model, Node(0, SELF_ATTN, head=0)).per_head is True
        assert Patch(murano_model, 0).per_head is False


# ── Mechanism / direction ─────────────────────────────────────────────


class TestPatchMechanism:
    def test_patch_last_layer_recovers_clean(self, murano_model):
        # Patching the final residual fully transplants the clean run's last
        # hidden state, so the corrupt run produces exactly the clean logits.
        _, results = _loaded()
        clean = _clean_logits(murano_model, results)
        corrupt = _corrupt_logits(murano_model, results)
        # The recovery check below only proves something if the runs differ.
        assert not torch.allclose(clean, corrupt, atol=1e-4)
        last = murano_model.n_layers - 1
        patched = Patch(murano_model, last)(results)[keys.PATCHED_LOGITS]
        assert torch.allclose(patched, clean, atol=1e-4)

    def test_patch_differs_from_corrupt(self, murano_model):
        _, results = _loaded()
        corrupt = _corrupt_logits(murano_model, results)
        last = murano_model.n_layers - 1
        patched = Patch(murano_model, last)(results)[keys.PATCHED_LOGITS]
        assert not torch.allclose(patched, corrupt)

    def test_direction_swap_is_noising(self, murano_model):
        # Swapping base/source runs the clean prompts and patches in corrupt:
        # the final-layer patch then yields exactly the corrupt logits.
        _, results = _loaded()
        clean = _clean_logits(murano_model, results)
        corrupt = _corrupt_logits(murano_model, results)
        # The recovery check below only proves something if the runs differ.
        assert not torch.allclose(clean, corrupt, atol=1e-4)
        last = murano_model.n_layers - 1
        patched = Patch(
            murano_model,
            last,
            base_key=keys.PROMPTS,
            source_key=keys.CORRUPT_PROMPTS,
        )(results)[keys.PATCHED_LOGITS]
        assert torch.allclose(patched, corrupt, atol=1e-4)

    def test_positions_restrict_the_patch(self, murano_model):
        # Patching the final residual only at the last position can change only
        # the last position's logits relative to the unpatched corrupt run.
        _, results = _loaded()
        corrupt = _corrupt_logits(murano_model, results)
        last = murano_model.n_layers - 1
        patched = Patch(murano_model, last, positions=[-1])(results)[
            keys.PATCHED_LOGITS
        ]
        assert torch.allclose(patched[:, :-1], corrupt[:, :-1], atol=1e-5)
        assert not torch.allclose(patched[:, -1], corrupt[:, -1])

    def test_per_head_patch_runs(self, murano_model):
        _, results = _loaded()
        clean = _clean_logits(murano_model, results)
        out = Patch(murano_model, Node(0, SELF_ATTN, head=0))(results)
        patched = out[keys.PATCHED_LOGITS]
        assert patched.shape == clean.shape
        assert torch.isfinite(patched).all()

    def test_per_head_with_positions_runs(self, murano_model):
        # Per-head patching restricted to one position exercises the combined
        # per-head + position write mask through a real forward pass.
        _, results = _loaded()
        clean = _clean_logits(murano_model, results)
        out = Patch(murano_model, Node(0, SELF_ATTN, head=0), positions=[-1])(results)
        patched = out[keys.PATCHED_LOGITS]
        assert patched.shape == clean.shape
        assert torch.isfinite(patched).all()

    def test_all_heads_equals_whole_attention(self, murano_model):
        # Patching every head of an attention site with the source must equal
        # patching the whole attention component there (the output projection is
        # linear with no bias). This pins per-head source placement, not just the
        # output shape.
        _, whole_results = _loaded()
        whole = Patch(murano_model, Node(0, SELF_ATTN))(whole_results)[
            keys.PATCHED_LOGITS
        ]
        _, head_results = _loaded()
        all_heads = Patch(
            murano_model, NodeSet.expand_heads(0, range(murano_model.n_heads))
        )(head_results)[keys.PATCHED_LOGITS]
        assert torch.allclose(whole, all_heads, atol=1e-5)


# ── Alignment guard ───────────────────────────────────────────────────


class TestPatchAlignment:
    def test_unequal_pair_length_raises(self, murano_model):
        # "good" is one token, "hello world" is two: the per-pair positions
        # cannot align, so the patch refuses rather than mixing in padding.
        _, results = _loaded(clean=["hello world"], corrupt=["good"])
        with pytest.raises(ValueError, match="token-length-matched"):
            Patch(murano_model, 0)(results)

    def test_long_context_pair_does_not_false_raise(self, murano_model):
        # A pair longer than the model's max length truncates identically on
        # both sides, so equal natural lengths must still patch. Regression for
        # comparing an untruncated source length against a truncated base mask.
        long_prompt = " ".join(
            ["hello"] * (murano_model.tokenizer.model_max_length + 8)
        )
        _, results = _loaded(
            clean=[long_prompt], corrupt=[long_prompt], correct=4, incorrect=5
        )
        out = Patch(murano_model, 0)(results)
        assert torch.isfinite(out[keys.PATCHED_LOGITS]).all()


# ── Pipeline / recovery ───────────────────────────────────────────────


class TestPatchRecovery:
    def test_recovered_full_patch_is_one(self, murano_model):
        ds = CleanCorruptDataset(
            clean=CLEAN, corrupt=CORRUPT, correct=[5, 6], incorrect=[6, 5]
        )
        all_layers = NodeSet.expand_layers(range(murano_model.n_layers))
        results = Pipeline(
            [
                LoadPaired(ds),
                Logits(murano_model),
                Logits(
                    murano_model,
                    prompts_key=keys.CORRUPT_PROMPTS,
                    logits_key=keys.CORRUPT_LOGITS,
                    mask_key=keys.CORRUPT_MASK,
                    targets=None,
                ),
                Patch(murano_model, all_layers),
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
                LogitDiffStep(
                    correct=ds.correct,
                    incorrect=ds.incorrect,
                    logits_key=keys.PATCHED_LOGITS,
                    mask_key=keys.PATCHED_MASK,
                    output_key="patched_ld",
                ),
                RecoveredMetricStep("clean_ld", "corrupt_ld", "patched_ld"),
            ]
        ).run()
        # Patching every layer reproduces the clean run, so patched == clean.
        clean_v = results["clean_ld"].value
        corrupt_v = results["corrupt_ld"].value
        assert results["patched_ld"].value == pytest.approx(clean_v, abs=1e-3)
        # The recovered fraction is then exactly 1.0 whenever the clean/corrupt
        # span is non-degenerate; nan is acceptable only if the two happen to tie.
        recovered = results["recovered"].value
        if abs(clean_v - corrupt_v) > 1e-3:
            assert recovered == pytest.approx(1.0, abs=1e-2)
        else:
            assert math.isnan(recovered)

    def test_validate_passes(self, murano_model):
        ds = CleanCorruptDataset(clean=CLEAN, corrupt=CORRUPT)
        pipe = Pipeline(
            [
                LoadPaired(ds),
                Patch(murano_model, 0),
            ]
        )
        assert keys.PATCHED_LOGITS in pipe.validate()

    def test_chain_without_source_fails(self, murano_model):
        pipe = Pipeline([Patch(murano_model, 0)])
        with pytest.raises(KeyError, match=keys.CORRUPT_PROMPTS):
            pipe.validate()
