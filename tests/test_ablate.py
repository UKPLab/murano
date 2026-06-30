"""Tests for the component-ablation step (zero / mean / resample).

The pure helpers (target grouping, mean, write-mask, blend) are checked on
synthetic tensors; the end-to-end behavior runs the tiny ``murano_model``
fixture through real intervened forward passes on CPU.
"""

from __future__ import annotations

import math

import pytest
import torch

from murano import Pipeline
from murano.artifacts import PromptBatch
from murano.nodes import SELF_ATTN, Node, NodeSet
from murano.results import Results
from murano.steps.ablate import (
    Ablate,
    _ablation_fn,
    _component_mean,
    _coerce_targets,
    _group_targets,
    _site_mask,
)
from murano.steps.logits import Logits
from murano.steps.metrics import LogitDiffStep
from murano.steps.prompts import LoadPrompts

ABLATED = "ablated_logits"


def _prompts(texts: list[str]) -> Results:
    r = Results()
    r["prompts"] = PromptBatch(prompts=texts)
    return r


def _clean_logits(model, texts: list[str]) -> torch.Tensor:
    return Logits(model)(_prompts(texts))["final_logits"]


# ── _group_targets ────────────────────────────────────────────────────


class TestGroupTargets:
    def test_whole_component_mode(self):
        sites, per_head = _group_targets(_coerce_targets([5, (6, "mlp")]))
        assert per_head is False
        assert sites == {(5, "resid_post"): None, (6, "mlp"): None}

    def test_per_head_mode_groups_heads(self):
        targets = NodeSet.expand_heads(3, [2, 0, 1])
        sites, per_head = _group_targets(list(targets))
        assert per_head is True
        assert sites == {(3, SELF_ATTN): [0, 1, 2]}  # sorted

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one target"):
            _group_targets([])

    def test_mixing_modes_raises(self):
        mixed = [Node(0, "mlp"), Node(0, SELF_ATTN, head=1)]
        with pytest.raises(ValueError, match="Mixing whole-component and per-head"):
            _group_targets(mixed)

    def test_resid_pre_mid_raise(self):
        with pytest.raises(ValueError, match="no single output hook"):
            _group_targets([Node(0, "resid_pre")])
        with pytest.raises(ValueError, match="no single output hook"):
            _group_targets([Node(0, "resid_mid")])

    def test_side_addressing_raises(self):
        from murano.nodes import Side

        with pytest.raises(ValueError, match="head-side"):
            _group_targets([Node(0, SELF_ATTN, side=Side.O)])


# ── _component_mean ───────────────────────────────────────────────────


class TestComponentMean:
    def test_pooled_mean_ignores_padding(self):
        captured = torch.tensor(
            [[[1.0, 1.0], [3.0, 3.0]], [[5.0, 5.0], [9.0, 9.0]]]
        )  # [B=2, S=2, d=2]
        mask = torch.tensor([[1, 1], [1, 0]])  # last row's 2nd token is padding
        mean = _component_mean(captured, mask, "all")
        # valid values: 1, 3, 5 -> mean 3.0 per feature
        assert mean.shape == (2,)
        assert torch.allclose(mean, torch.tensor([3.0, 3.0]))

    def test_position_mean_is_per_token(self):
        captured = torch.tensor([[[2.0], [4.0]], [[6.0], [10.0]]])  # [B=2, S=2, d=1]
        mask = torch.tensor([[1, 1], [1, 0]])
        mean = _component_mean(captured, mask, "position")
        # pos0 over both rows: (2+6)/2=4; pos1 only first row valid: 4
        assert mean.shape == (2, 1)
        assert torch.allclose(mean, torch.tensor([[4.0], [4.0]]))


# ── _site_mask ────────────────────────────────────────────────────────


class TestSiteMask:
    def test_whole_all_positions_is_scalar(self):
        mask = _site_mask(2, 3, None, None, 4)
        assert mask.dim() == 0 and float(mask) == 1.0

    def test_whole_specific_position(self):
        pos = torch.tensor([2, 1])
        mask = _site_mask(2, 3, pos, None, 4)
        assert mask.shape == (2, 3, 1)
        assert mask[0, 2, 0] == 1.0 and mask[1, 1, 0] == 1.0
        assert mask.sum() == 2.0

    def test_per_head_all_positions(self):
        mask = _site_mask(2, 3, None, [1, 3], 4)
        assert mask.shape == (1, 1, 4, 1)
        assert mask[0, 0, 1, 0] == 1.0 and mask[0, 0, 3, 0] == 1.0
        assert mask[0, 0, 0, 0] == 0.0 and mask[0, 0, 2, 0] == 0.0

    def test_per_head_specific_position(self):
        pos = torch.tensor([2, 0])
        mask = _site_mask(2, 3, pos, [0], 4)
        assert mask.shape == (2, 3, 4, 1)
        assert mask[0, 2, 0, 0] == 1.0 and mask[1, 0, 0, 0] == 1.0
        assert mask[0, 2, 1, 0] == 0.0  # head 1 untouched


# ── _ablation_fn ──────────────────────────────────────────────────────


class TestAblationFn:
    def test_non_target_site_passes_through(self):
        fn = _ablation_fn({(0, "resid_post"): (torch.tensor(5.0), torch.tensor(1.0))})
        act = torch.ones(2, 3, 4)
        assert fn(act, Node(1, "resid_post")) is act

    def test_full_mask_replaces_all(self):
        fn = _ablation_fn({(0, "resid_post"): (torch.tensor(5.0), torch.tensor(1.0))})
        out = fn(torch.ones(2, 3, 4), Node(0, "resid_post"))
        assert torch.allclose(out, torch.full((2, 3, 4), 5.0))

    def test_partial_mask_blends(self):
        mask = torch.zeros(1, 2, 1)
        mask[0, 1, 0] = 1.0
        fn = _ablation_fn({(0, "mlp"): (torch.tensor(0.0), mask)})
        act = torch.ones(1, 2, 3)
        out = fn(act, Node(0, "mlp"))
        assert torch.allclose(out[0, 0], torch.ones(3))  # kept
        assert torch.allclose(out[0, 1], torch.zeros(3))  # zeroed


# ── End to end: zero ablation ─────────────────────────────────────────


class TestZeroAblation:
    def test_zero_changes_logits_and_writes_mask(self, murano_model):
        texts = ["hello world", "good world"]
        clean = _clean_logits(murano_model, texts)
        out = Ablate(murano_model, 0, method="zero")(_prompts(texts))
        ablated = out[ABLATED]
        assert ablated.shape == clean.shape
        assert ablated.dtype == torch.float32
        assert not torch.allclose(ablated, clean)
        expected_mask = murano_model.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_token_type_ids=False,
        )["attention_mask"]
        assert torch.equal(out["attention_mask"], expected_mask)

    def test_mixed_module_multi_layer_runs(self, murano_model):
        # Targets spanning different modules at different layers: the forward
        # must hook exactly these sites, not the layer-by-module product, and
        # the non-target product cells must be left untouched.
        texts = ["hello world", "good world"]
        clean = _clean_logits(murano_model, texts)
        out = Ablate(
            murano_model, [Node(0, "mlp"), Node(1, "resid_post")], method="zero"
        )(_prompts(texts))[ABLATED]
        assert out.shape == clean.shape
        assert torch.isfinite(out).all()
        assert not torch.allclose(out, clean)

    def test_zero_only_last_position_is_causal(self, murano_model):
        # Ablating the residual at the last token can only change the last
        # position's logits; earlier positions never attend to it.
        texts = ["hello world"]
        clean = _clean_logits(murano_model, texts)
        out = Ablate(murano_model, 0, method="zero", positions=[-1])(_prompts(texts))
        ablated = out[ABLATED]
        assert torch.allclose(ablated[:, :-1], clean[:, :-1], atol=1e-5)
        assert not torch.allclose(ablated[:, -1], clean[:, -1])


# ── Mean ablation ─────────────────────────────────────────────────────


class TestMeanAblation:
    @pytest.mark.parametrize("mean_over", ["all", "position"])
    def test_mean_changes_logits(self, murano_model, mean_over):
        texts = ["hello world", "good world"]
        clean = _clean_logits(murano_model, texts)
        out = Ablate(murano_model, 0, method="mean", mean_over=mean_over)(
            _prompts(texts)
        )
        ablated = out[ABLATED]
        assert ablated.shape == clean.shape
        assert torch.isfinite(ablated).all()
        assert not torch.allclose(ablated, clean)

    def test_precomputed_means_for_per_head_raises(self, murano_model):
        with pytest.raises(NotImplementedError, match="per-head"):
            Ablate(
                murano_model,
                NodeSet.expand_heads(0, [0]),
                method="mean",
                means={Node(0, SELF_ATTN): torch.zeros(8)},
            )

    def test_external_means_of_zero_equals_zero_method(self, murano_model):
        # A precomputed all-zero mean must reproduce the zero ablation exactly.
        texts = ["hello world", "good world"]
        d = murano_model.d_model
        zeroed = Ablate(murano_model, 0, method="zero")(_prompts(texts))[ABLATED]
        meaned = Ablate(murano_model, 0, method="mean", means={0: torch.zeros(d)})(
            _prompts(texts)
        )[ABLATED]
        assert torch.allclose(zeroed, meaned, atol=1e-5)

    def test_external_means_missing_site_raises(self, murano_model):
        d = murano_model.d_model
        with pytest.raises(ValueError, match="no entry for target"):
            Ablate(murano_model, [0, 1], method="mean", means={0: torch.zeros(d)})

    def test_external_means_wrong_shape_raises(self, murano_model):
        with pytest.raises(ValueError, match="expected d_model"):
            Ablate(murano_model, 0, method="mean", means={0: torch.zeros(3)})

    def test_external_means_with_mean_over_raises(self, murano_model):
        d = murano_model.d_model
        with pytest.raises(ValueError, match="mean_over does not apply"):
            Ablate(
                murano_model,
                0,
                method="mean",
                mean_over="position",
                means={0: torch.zeros(d)},
            )


# ── Resample ablation ─────────────────────────────────────────────────


class TestResampleAblation:
    def test_identity_permutation_is_noop(self, murano_model):
        # Replacing each example's activation with its own is a no-op, so the
        # ablated logits must match the clean run.
        texts = ["hello world", "good world"]
        clean = _clean_logits(murano_model, texts)
        out = Ablate(murano_model, 0, method="resample", permutation=[0, 1])(
            _prompts(texts)
        )
        assert torch.allclose(out[ABLATED], clean, atol=1e-5)

    def test_swap_permutation_changes_logits(self, murano_model):
        texts = ["hello world", "good world"]
        clean = _clean_logits(murano_model, texts)
        out = Ablate(murano_model, 0, method="resample", permutation=[1, 0])(
            _prompts(texts)
        )
        assert not torch.allclose(out[ABLATED], clean)

    def test_swap_last_layer_plants_partner_values(self, murano_model):
        # Resampling the final layer's residual swaps what reaches the
        # unembedding, so each example ends up with its partner's clean logits.
        # This checks resample writes the *correct* source values, not just that
        # something changed.
        texts = ["hello world", "good world"]
        clean = _clean_logits(murano_model, texts)
        last = murano_model.n_layers - 1
        out = Ablate(murano_model, last, method="resample", permutation=[1, 0])(
            _prompts(texts)
        )[ABLATED]
        assert torch.allclose(out[0], clean[1], atol=1e-4)
        assert torch.allclose(out[1], clean[0], atol=1e-4)

    def test_unequal_length_within_batch_raises(self, murano_model):
        # "good" is one token, "hello world" is two: within-batch positions
        # cannot align, so the step refuses rather than mixing in padding.
        step = Ablate(murano_model, 0, method="resample", permutation=[1, 0])
        with pytest.raises(ValueError, match="equal-length prompts"):
            step(_prompts(["hello world", "good"]))

    def test_source_resample_runs(self, murano_model):
        texts = ["hello world", "good world"]
        clean = _clean_logits(murano_model, texts)
        out = Ablate(
            murano_model,
            0,
            method="resample",
            source=["good world", "hello world"],
        )(_prompts(texts))[ABLATED]
        assert out.shape == clean.shape
        assert torch.isfinite(out).all()
        assert not torch.allclose(out, clean)

    def test_bad_permutation_raises(self, murano_model):
        with pytest.raises(ValueError, match="rearrangement of range"):
            Ablate(murano_model, 0, method="resample", permutation=[0, 0])(
                _prompts(["hello world", "good world"])
            )

    def test_source_length_mismatch_raises(self, murano_model):
        step = Ablate(murano_model, 0, method="resample", source=["good world"])
        with pytest.raises(ValueError, match="prompts but the batch"):
            step(_prompts(["hello world", "good world"]))

    def test_source_token_length_mismatch_raises(self, murano_model):
        # "good" is one token, "hello world" is two: positions cannot align.
        step = Ablate(murano_model, 0, method="resample", source=["good", "good world"])
        with pytest.raises(ValueError, match="token-length-matched"):
            step(_prompts(["hello world", "good world"]))

    def test_source_longer_than_base_raises(self, murano_model):
        # An over-long source must be rejected on its untruncated length, not
        # silently truncated to the base length (which would mis-align positions).
        step = Ablate(
            murano_model,
            0,
            method="resample",
            source=["hello good world", "good world"],
        )
        with pytest.raises(ValueError, match="token-length-matched"):
            step(_prompts(["good world", "good world"]))


# ── Per-head ablation ─────────────────────────────────────────────────


class TestPerHeadAblation:
    def test_single_head_changes_logits(self, murano_model):
        texts = ["hello world"]
        clean = _clean_logits(murano_model, texts)
        out = Ablate(murano_model, Node(0, SELF_ATTN, head=0), method="zero")(
            _prompts(texts)
        )
        assert out[ABLATED].shape == clean.shape
        assert not torch.allclose(out[ABLATED], clean)

    def test_all_heads_equals_whole_attention(self, murano_model):
        # Zeroing every head's projection input drives the o_proj output to
        # zero (no bias), which equals zeroing the whole attention output.
        texts = ["hello world", "good world"]
        whole = Ablate(murano_model, Node(0, SELF_ATTN), method="zero")(
            _prompts(texts)
        )[ABLATED]
        all_heads = Ablate(
            murano_model,
            NodeSet.expand_heads(0, range(murano_model.n_heads)),
            method="zero",
        )(_prompts(texts))[ABLATED]
        assert torch.allclose(whole, all_heads, atol=1e-5)


# ── Construction guards ───────────────────────────────────────────────


class TestConstruction:
    def test_unknown_method_raises(self, murano_model):
        with pytest.raises(ValueError, match="method must be"):
            Ablate(murano_model, 0, method="nope")  # type: ignore[arg-type]

    def test_means_with_wrong_method_raises(self, murano_model):
        with pytest.raises(ValueError, match="means= is only valid"):
            Ablate(murano_model, 0, method="zero", means={0: torch.zeros(8)})

    def test_source_with_wrong_method_raises(self, murano_model):
        with pytest.raises(ValueError, match="source= is only valid"):
            Ablate(murano_model, 0, method="zero", source=["x"])

    def test_source_key_with_wrong_method_raises(self, murano_model):
        with pytest.raises(ValueError, match="source_key= is only valid"):
            Ablate(murano_model, 0, method="zero", source_key="prompts")

    def test_source_and_source_key_together_raises(self, murano_model):
        with pytest.raises(ValueError, match="either source="):
            Ablate(
                murano_model,
                0,
                method="resample",
                source=["x"],
                source_key="corrupt_prompts",
            )

    def test_permutation_with_source_raises(self, murano_model):
        with pytest.raises(ValueError, match="do not apply when"):
            Ablate(murano_model, 0, method="resample", source=["x"], permutation=[0])

    def test_seed_with_source_key_raises(self, murano_model):
        with pytest.raises(ValueError, match="do not apply when"):
            Ablate(murano_model, 0, method="resample", source_key="prompts", seed=0)

    def test_bad_mean_over_raises(self, murano_model):
        with pytest.raises(ValueError, match="mean_over must be"):
            Ablate(murano_model, 0, method="mean", mean_over="bogus")  # type: ignore[arg-type]

    def test_permutation_with_wrong_method_raises(self, murano_model):
        with pytest.raises(ValueError, match="only valid for method='resample'"):
            Ablate(murano_model, 0, method="zero", permutation=[0, 1])

    def test_permutation_and_seed_together_raises(self, murano_model):
        with pytest.raises(ValueError, match="either permutation= or seed="):
            Ablate(murano_model, 0, method="resample", permutation=[0, 1], seed=0)


# ── Pipeline integration ──────────────────────────────────────────────


class TestPipeline:
    def test_ablated_logits_feed_logit_diff(self, murano_model):
        results = Pipeline(
            [
                LoadPrompts(["hello world", "good world"]),
                Ablate(murano_model, 0, method="zero"),
                LogitDiffStep(correct=5, incorrect=7, logits_key=ABLATED),
            ]
        ).run()
        result = results["logit_diff"]
        assert result.per_example is not None and len(result.per_example) == 2

    def test_ablated_logits_feed_kl_against_clean(self, murano_model):
        from murano.steps.metrics import KLDivergenceStep

        results = Pipeline(
            [
                LoadPrompts(["hello world", "good world"]),
                Logits(murano_model),  # clean final_logits
                Ablate(murano_model, 0, method="zero"),  # ablated_logits
                KLDivergenceStep(p_key="final_logits", q_key=ABLATED),
            ]
        ).run()
        assert math.isfinite(results["kl_div"].value)

    def test_validate_passes(self, murano_model):
        pipe = Pipeline(
            [
                LoadPrompts(["hello world"]),
                Ablate(murano_model, 0, method="zero"),
                LogitDiffStep(correct=5, incorrect=7, logits_key=ABLATED),
            ]
        )
        assert ABLATED in pipe.validate()

    def test_chain_without_prompts_fails(self, murano_model):
        pipe = Pipeline([Ablate(murano_model, 0, method="zero")])
        with pytest.raises(KeyError, match="prompts"):
            pipe.validate()
