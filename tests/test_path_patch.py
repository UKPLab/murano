"""Tests for the path-patching step (PathPatch).

PathPatch isolates the direct sender-to-receiver effect: it injects a sender's
source activation while freezing every other component at its base value. These
tests pin the contract, the freeze mechanism (a base==source run is an exact
no-op), and an exact cross-check against the already-tested resample engine (a
last-MLP sender equals an Ablate resample of that MLP, since nothing runs after
it). The freeze/reshape plumbing is exercised on both norm families
(``murano_model`` RMSNorm, ``gpt2_model`` LayerNorm + Conv1D). End-to-end runs on
CPU; the name-mover reproduction on a real model is a separate GPU check.
"""

from __future__ import annotations

import pytest
import torch

from murano import Pipeline, keys
from murano.dataset import CleanCorruptDataset
from murano.nodes import MLP, RESID_POST, SELF_ATTN, Edge, Node, NodeSet
from murano.results import Results
from murano.steps.ablate import Ablate
from murano.steps.logits import Logits
from murano.steps.metrics import LogitDiffStep, RecoveredMetricStep
from murano.steps.paired import LoadPaired
from murano.steps.path_patch import PathPatch

FIXTURES = ["murano_model", "gpt2_model"]

# Equal token length per pair (known vocab words, no BOS) so positions align.
CLEAN = ["hello world", "good world"]
CORRUPT = ["good world", "bad world"]


def _loaded(clean=CLEAN, corrupt=CORRUPT, correct=5, incorrect=6):
    ds = CleanCorruptDataset(
        clean=clean, corrupt=corrupt, correct=correct, incorrect=incorrect
    )
    return ds, LoadPaired(ds)(Results())


def _clean_logits(model, results):
    return Logits(model)(results)[keys.FINAL_LOGITS]


# ── Contract ──────────────────────────────────────────────────────────


class TestContract:
    def test_default_read_write_keys(self, murano_model):
        step = PathPatch(murano_model, Node(0, SELF_ATTN, head=0))
        assert step.reads == [keys.PROMPTS, keys.CORRUPT_PROMPTS]
        assert step.writes == [keys.PATH_PATCHED_LOGITS, keys.PATH_PATCHED_MASK]
        assert step.write_types == {
            keys.PATH_PATCHED_LOGITS: torch.Tensor,
            keys.PATH_PATCHED_MASK: torch.Tensor,
        }

    def test_default_receiver_is_final_residual(self, murano_model):
        step = PathPatch(murano_model, Node(0, SELF_ATTN, head=0))
        assert step.receiver == Node(murano_model.n_layers - 1, RESID_POST)
        assert step._receiver_is_final is True

    def test_edge_sets_sender_and_receiver(self, murano_model):
        last = murano_model.n_layers - 1
        edge = Edge(Node(0, SELF_ATTN, head=1), Node(last, RESID_POST))
        step = PathPatch(murano_model, edge)
        assert step.attn_senders == {0: [1]}
        assert step.receiver == Node(last, RESID_POST)

    def test_edge_and_receiver_together_raises(self, murano_model):
        edge = Edge(Node(0, SELF_ATTN, head=0), Node(1, RESID_POST))
        with pytest.raises(ValueError, match="Edge OR"):
            PathPatch(murano_model, edge, receiver=Node(1, RESID_POST))

    def test_component_receiver_raises(self, murano_model):
        with pytest.raises(NotImplementedError, match="residual-stream"):
            PathPatch(murano_model, Node(0, SELF_ATTN, head=0), receiver=Node(1, MLP))

    def test_head_side_sender_out_of_scope(self, murano_model):
        with pytest.raises(ValueError, match="head-side"):
            PathPatch(murano_model, Node(0, SELF_ATTN, head=0, side="V"))

    def test_attention_sender_without_head_raises(self, murano_model):
        with pytest.raises(ValueError, match="must name a head"):
            PathPatch(murano_model, Node(0, SELF_ATTN))

    def test_sender_position_rejected(self, murano_model):
        with pytest.raises(ValueError, match="position"):
            PathPatch(murano_model, Node(0, SELF_ATTN, head=0, position=1))

    def test_duplicate_sender_heads_deduped(self, murano_model):
        step = PathPatch(
            murano_model, [Node(0, SELF_ATTN, head=1), Node(0, SELF_ATTN, head=1)]
        )
        assert step.attn_senders == {0: [1]}

    def test_out_of_range_receiver_raises(self, murano_model):
        with pytest.raises(ValueError, match="out of range"):
            PathPatch(
                murano_model, Node(0, SELF_ATTN, head=0), receiver=Node(99, RESID_POST)
            )

    def test_out_of_range_sender_head_raises(self, murano_model):
        with pytest.raises(ValueError, match="out of range"):
            PathPatch(murano_model, Node(0, SELF_ATTN, head=99))

    def test_validate_passes(self, murano_model):
        ds = CleanCorruptDataset(clean=CLEAN, corrupt=CORRUPT)
        pipe = Pipeline(
            [LoadPaired(ds), PathPatch(murano_model, Node(0, SELF_ATTN, head=0))]
        )
        assert keys.PATH_PATCHED_LOGITS in pipe.validate()


# ── Freeze mechanism ──────────────────────────────────────────────────


class TestFreezeMechanism:
    @pytest.mark.parametrize("fixture", FIXTURES)
    def test_same_base_source_is_noop_final_receiver(self, fixture, request):
        # source == base means every injected activation equals the base value, so
        # freezing + injecting is the identity: the logits must match the clean run.
        model = request.getfixturevalue(fixture)
        _, results = _loaded()
        clean = _clean_logits(model, results)
        heads = NodeSet.expand_heads(0, range(model.n_heads))
        out = PathPatch(model, heads, source_key=keys.PROMPTS)(results)
        assert torch.allclose(out[keys.PATH_PATCHED_LOGITS], clean, atol=1e-4)

    def test_same_base_source_is_noop_intermediate_receiver(self, murano_model):
        # The extra capture+patch pass for a non-final receiver must also be an
        # identity when source == base.
        _, results = _loaded()
        clean = _clean_logits(murano_model, results)
        out = PathPatch(
            murano_model,
            Node(0, SELF_ATTN, head=0),
            receiver=Node(0, RESID_POST),
            source_key=keys.PROMPTS,
        )(results)
        assert torch.allclose(out[keys.PATH_PATCHED_LOGITS], clean, atol=1e-4)

    @pytest.mark.parametrize("fixture", FIXTURES)
    def test_last_mlp_sender_matches_resample(self, fixture, request):
        # Nothing runs after the last MLP, so path-patching it (freeze everything
        # else, a no-op on the base run, and swap the last MLP to source) is exactly
        # an Ablate resample of that MLP. Pins the sender injection + freeze against
        # the already-tested engine, on both norm families.
        model = request.getfixturevalue(fixture)
        last = model.n_layers - 1
        _, pp_results = _loaded()
        pp = PathPatch(model, Node(last, MLP))(pp_results)[keys.PATH_PATCHED_LOGITS]
        _, ab_results = _loaded()
        ab = Ablate(
            model, Node(last, MLP), method="resample", source_key=keys.CORRUPT_PROMPTS
        )(ab_results)[keys.ABLATED_LOGITS]
        assert torch.allclose(pp, ab, atol=1e-4)

    def test_per_head_sender_changes_and_differs(self, murano_model):
        # A single-head sender must move the logits, and a different head must give
        # a different result: the injection targets the addressed head, not a fixed
        # slice.
        _, results = _loaded()
        clean = _clean_logits(murano_model, results)
        head0 = PathPatch(murano_model, Node(0, SELF_ATTN, head=0))(results)[
            keys.PATH_PATCHED_LOGITS
        ]
        head1 = PathPatch(murano_model, Node(0, SELF_ATTN, head=1))(results)[
            keys.PATH_PATCHED_LOGITS
        ]
        assert not torch.allclose(head0, clean, atol=1e-4)
        assert not torch.allclose(head0, head1, atol=1e-4)

    def test_freeze_mlps_changes_the_result(self, murano_model):
        # With an early sender, freezing the MLPs blocks their mediation, so the
        # result must differ from letting them recompute; it is not just a flag.
        _, results = _loaded()
        sender = Node(0, SELF_ATTN, head=0)
        free = PathPatch(murano_model, sender, freeze_mlps=False)(results)[
            keys.PATH_PATCHED_LOGITS
        ]
        frozen = PathPatch(murano_model, sender, freeze_mlps=True)(results)[
            keys.PATH_PATCHED_LOGITS
        ]
        assert not torch.allclose(free, frozen, atol=1e-4)

    def test_intermediate_receiver_has_real_effect(self, murano_model):
        # A non-final receiver with source != base drives the capture+patch (Pass D)
        # path and must move the logits off the clean run.
        _, results = _loaded()
        clean = _clean_logits(murano_model, results)
        out = PathPatch(
            murano_model, Node(0, SELF_ATTN, head=0), receiver=Node(0, RESID_POST)
        )(results)
        assert not torch.allclose(out[keys.PATH_PATCHED_LOGITS], clean, atol=1e-4)

    def test_positions_restrict_injection(self, murano_model):
        # Injecting only at the last position must differ from injecting at every
        # position (earlier positions keep the base value).
        _, results = _loaded()
        last = murano_model.n_layers - 1
        all_pos = PathPatch(murano_model, Node(last, MLP))(results)[
            keys.PATH_PATCHED_LOGITS
        ]
        last_pos = PathPatch(murano_model, Node(last, MLP), positions=[-1])(results)[
            keys.PATH_PATCHED_LOGITS
        ]
        assert not torch.allclose(all_pos, last_pos, atol=1e-4)

    def test_writes_base_mask(self, murano_model):
        _, results = _loaded()
        out = PathPatch(murano_model, Node(0, SELF_ATTN, head=0))(results)
        expected = murano_model.tokenizer(
            CLEAN,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_token_type_ids=False,
        )["attention_mask"]
        assert torch.equal(out[keys.PATH_PATCHED_MASK], expected)


# ── Alignment guard ───────────────────────────────────────────────────


class TestAlignment:
    def test_unequal_pair_length_raises(self, murano_model):
        _, results = _loaded(clean=["hello world"], corrupt=["good"])
        with pytest.raises(ValueError, match="token-length-matched"):
            PathPatch(murano_model, Node(0, SELF_ATTN, head=0))(results)


# ── Pipeline / recovery ───────────────────────────────────────────────


class TestPipeline:
    def test_recovered_metric_wires_through(self, murano_model):
        ds = CleanCorruptDataset(
            clean=CLEAN, corrupt=CORRUPT, correct=[5, 6], incorrect=[6, 5]
        )
        last = murano_model.n_layers - 1
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
                PathPatch(murano_model, Node(last, MLP)),
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
                    logits_key=keys.PATH_PATCHED_LOGITS,
                    mask_key=keys.PATH_PATCHED_MASK,
                    output_key="patched_ld",
                ),
                RecoveredMetricStep("clean_ld", "corrupt_ld", "patched_ld"),
            ]
        ).run()
        # The last-MLP path patch moves the metric off the clean run.
        assert results["patched_ld"].value != pytest.approx(results["clean_ld"].value)
        assert torch.isfinite(torch.tensor(results["patched_ld"].value))
