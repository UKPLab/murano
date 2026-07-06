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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from murano import Pipeline, keys
from murano.dataset import CleanCorruptDataset
from murano.nodes import MLP, RESID_POST, SELF_ATTN, Edge, Node, NodeSet
from murano.results import Results
from murano.steps.ablate import Ablate
from murano.steps.attention import _projection_weight
from murano.steps.logits import Logits
from murano.steps.metrics import LogitDiffStep, RecoveredMetricStep
from murano.steps.paired import LoadPaired
from murano.steps.path_patch import PathPatch, _receiver_slot

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


def _tokenize(model, prompts):
    return model.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_token_type_ids=False,
    )


def _side_slot(model, receiver):
    """Independently locate the receiver head's Q/K/V module and columns (MHA only).

    Deliberately does not call ``_receiver_slot``: this hard-codes the block layout
    so the gold-standard test cross-checks that arithmetic too. Assumes the fixture
    is multi-head (kv heads == query heads), so a head's key/value columns start at
    ``head * head_dim`` within their block.
    """
    hd, head, side = model.head_dim, receiver.head, receiver.side
    attn = model.resolve_module(receiver.layer, SELF_ATTN)._module
    if hasattr(attn, "q_proj"):
        module = {"Q": attn.q_proj, "K": attn.k_proj, "V": attn.v_proj}[side]
        return module, slice(head * hd, (head + 1) * hd)
    block = {"Q": 0, "K": 1, "V": 2}[side] * model.n_heads * hd
    return attn.c_attn, slice(block + head * hd, block + (head + 1) * hd)


def _direct_qkv_reference(model, sender, receiver, clean=CLEAN, corrupt=CORRUPT):
    """Reconstruct a Q/K/V receiver's direct-path logits from first principles.

    Independent of PathPatch's machinery: with a single upstream head sender and
    every other component frozen to base, the residual at the receiver layer is the
    base residual plus that sender head's ``source - base`` output contribution. This
    reads the receiver head's Q/K/V from the model's own norm and projection on that
    residual, then patches exactly that value into a normal base forward. Used to
    cross-check the step's capture and patch against a hand computation.
    """
    hd, n_heads = model.head_dim, model.n_heads
    layer = receiver.layer
    base = _tokenize(model, clean)
    source = _tokenize(model, corrupt)

    o_proj = model.attn_out_proj(sender.layer, SELF_ATTN)._module
    w_o_head = _projection_weight(o_proj)[:, sender.head * hd : (sender.head + 1) * hd]

    def head_contribution(tokens):
        cap = {}

        def pre(module, args):
            z = args[0].detach().float()
            b, s, _ = z.shape
            z_head = z.reshape(b, s, n_heads, hd)[:, :, sender.head, :]
            cap["o"] = z_head @ w_o_head.t()

        handle = o_proj.register_forward_pre_hook(pre)
        try:
            model.forward_logits(tokens, fn=None)
        finally:
            handle.remove()
        return cap["o"]

    prev = model.layer(layer - 1)._module
    cap = {}

    def cap_resid(module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        cap["r"] = out.detach().float()

    handle = prev.register_forward_hook(cap_resid)
    try:
        model.forward_logits(base, fn=None)
    finally:
        handle.remove()
    resid_frozen = cap["r"] + (head_contribution(source) - head_contribution(base))

    module, cols = _side_slot(model, receiver)

    def replace_resid(module, args, output):
        value = resid_frozen.to(
            (output[0] if isinstance(output, tuple) else output).dtype
        )
        return (value, *output[1:]) if isinstance(output, tuple) else value

    val_cap = {}

    def cap_value(module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        val_cap["v"] = out[:, :, cols].detach().clone()

    handles = [
        prev.register_forward_hook(replace_resid),
        module.register_forward_hook(cap_value),
    ]
    try:
        model.forward_logits(base, fn=None)
    finally:
        for handle in handles:
            handle.remove()
    value_ref = val_cap["v"]

    def patch_value(module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        edited = out.clone()
        edited[:, :, cols] = value_ref.to(edited.dtype)
        return (edited, *output[1:]) if isinstance(output, tuple) else edited

    handle = module.register_forward_hook(patch_value)
    try:
        logits = model.forward_logits(base, fn=None)
    finally:
        handle.remove()
    return logits


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


# ── Q/K/V receiver: slot resolution ───────────────────────────────────


def _stub_model(attn, n_heads, head_dim):
    """Wrap a bare attention module so _receiver_slot can resolve it."""
    return SimpleNamespace(
        n_heads=n_heads,
        head_dim=head_dim,
        resolve_module=lambda layer, module: SimpleNamespace(_module=attn),
    )


class TestReceiverSlot:
    def test_separate_q_projection(self, murano_model):
        # head_dim = 32 / 4 = 8; head 2's query is columns [16, 24) of q_proj.
        attn = murano_model.resolve_module(1, SELF_ATTN)._module
        module, start, end = _receiver_slot(
            murano_model, Node(1, SELF_ATTN, head=2, side="Q")
        )
        assert module is attn.q_proj
        assert (start, end) == (16, 24)

    def test_fused_c_attn_blocks(self, gpt2_model):
        # Fused q|k|v: block width d = 4*8 = 32. K is block 1, V is block 2.
        attn = gpt2_model.resolve_module(1, SELF_ATTN)._module
        k_module, k_start, k_end = _receiver_slot(
            gpt2_model, Node(1, SELF_ATTN, head=1, side="K")
        )
        _, v_start, v_end = _receiver_slot(
            gpt2_model, Node(1, SELF_ATTN, head=1, side="V")
        )
        assert k_module is attn.c_attn
        assert (k_start, k_end) == (32 + 8, 32 + 16)
        assert (v_start, v_end) == (64 + 8, 64 + 16)

    def test_mqa_shares_one_kv_head(self, murano_model_attn_mqa):
        # num_key_value_heads=1, so every query head's value maps to kv head 0.
        attn = murano_model_attn_mqa.resolve_module(1, SELF_ATTN)._module
        for query_head in (0, 3):
            module, start, end = _receiver_slot(
                murano_model_attn_mqa, Node(1, SELF_ATTN, head=query_head, side="V")
            )
            assert module is attn.v_proj
            assert (start, end) == (0, 8)

    def test_grouped_query_maps_to_shared_kv_head(self):
        # 8 query heads, 2 kv heads, group size 4: heads 0-3 -> kv 0, 4-7 -> kv 1.
        attn = nn.Module()
        attn.q_proj = nn.Linear(32, 8 * 4, bias=False)
        attn.k_proj = nn.Linear(32, 2 * 4, bias=False)
        attn.v_proj = nn.Linear(32, 2 * 4, bias=False)
        model = _stub_model(attn, n_heads=8, head_dim=4)
        _, s2, e2 = _receiver_slot(model, Node(0, SELF_ATTN, head=2, side="K"))
        _, s5, e5 = _receiver_slot(model, Node(0, SELF_ATTN, head=5, side="K"))
        assert (s2, e2) == (0, 4)
        assert (s5, e5) == (4, 8)

    def test_interleaved_qkv_raises(self):
        attn = nn.Module()
        attn.query_key_value = nn.Linear(32, 3 * 32, bias=False)
        model = _stub_model(attn, n_heads=8, head_dim=4)
        with pytest.raises(NotImplementedError, match="query_key_value"):
            _receiver_slot(model, Node(0, SELF_ATTN, head=0, side="Q"))


# ── Q/K/V receiver: contract ──────────────────────────────────────────


class TestQKVReceiverContract:
    def test_qkv_receiver_kind_and_not_final(self, murano_model):
        step = PathPatch(
            murano_model,
            Node(0, SELF_ATTN, head=0),
            receiver=Node(1, SELF_ATTN, head=1, side="Q"),
        )
        assert step._receiver_kind == "qkv"
        assert step._receiver_is_final is False

    def test_output_side_receiver_raises(self, murano_model):
        with pytest.raises(NotImplementedError, match="output side"):
            PathPatch(
                murano_model,
                Node(0, SELF_ATTN, head=0),
                receiver=Node(1, SELF_ATTN, head=0, side="O"),
            )

    def test_side_less_head_receiver_raises(self, murano_model):
        with pytest.raises(NotImplementedError, match="side=Q"):
            PathPatch(
                murano_model,
                Node(0, SELF_ATTN, head=0),
                receiver=Node(1, SELF_ATTN, head=0),
            )

    def test_out_of_range_receiver_head_raises(self, murano_model):
        with pytest.raises(ValueError, match="receiver head"):
            PathPatch(
                murano_model,
                Node(0, SELF_ATTN, head=0),
                receiver=Node(1, SELF_ATTN, head=99, side="Q"),
            )

    def test_receiver_position_rejected(self, murano_model):
        with pytest.raises(ValueError, match="receiver position"):
            PathPatch(
                murano_model,
                Node(0, SELF_ATTN, head=0),
                receiver=Node(1, SELF_ATTN, head=1, side="Q", position=1),
            )

    def test_validate_passes(self, murano_model):
        ds = CleanCorruptDataset(clean=CLEAN, corrupt=CORRUPT)
        pipe = Pipeline(
            [
                LoadPaired(ds),
                PathPatch(
                    murano_model,
                    Node(0, SELF_ATTN, head=0),
                    receiver=Node(1, SELF_ATTN, head=1, side="Q"),
                ),
            ]
        )
        assert keys.PATH_PATCHED_LOGITS in pipe.validate()


# ── Q/K/V receiver: behaviour ─────────────────────────────────────────


class TestQKVReceiverBehavior:
    @pytest.mark.parametrize("fixture", FIXTURES)
    @pytest.mark.parametrize("side", ["Q", "K", "V"])
    def test_same_base_source_is_noop(self, fixture, side, request):
        # source == base makes the captured q/k/v equal the base value, so the
        # capture+patch pass is the identity: the logits must match the clean run.
        model = request.getfixturevalue(fixture)
        _, results = _loaded()
        clean = _clean_logits(model, results)
        out = PathPatch(
            model,
            Node(0, SELF_ATTN, head=0),
            receiver=Node(1, SELF_ATTN, head=1, side=side),
            source_key=keys.PROMPTS,
        )(results)
        assert torch.allclose(out[keys.PATH_PATCHED_LOGITS], clean, atol=1e-4)

    @pytest.mark.parametrize("fixture", FIXTURES)
    def test_same_layer_sender_cannot_reach_receiver_query(self, fixture, request):
        # A sender at the receiver's own layer writes into the residual only after
        # that layer's query is computed, so it has no direct path to the query: the
        # captured value equals base and the logits equal the clean run exactly.
        model = request.getfixturevalue(fixture)
        _, results = _loaded()
        clean = _clean_logits(model, results)
        out = PathPatch(
            model,
            Node(1, SELF_ATTN, head=0),
            receiver=Node(1, SELF_ATTN, head=1, side="Q"),
        )(results)
        assert torch.allclose(out[keys.PATH_PATCHED_LOGITS], clean, atol=1e-4)

    def test_upstream_sender_moves_logits(self, murano_model):
        # A value receiver flows linearly through the head's OV map and the
        # unembedding, so an upstream sender produces a clearly measurable change
        # (a query/key receiver only reshuffles the softmax and, on a two-token toy
        # prompt, barely moves the logits).
        _, results = _loaded()
        clean = _clean_logits(murano_model, results)
        senders = NodeSet.expand_heads(0, range(murano_model.n_heads))
        out = PathPatch(
            murano_model, senders, receiver=Node(1, SELF_ATTN, head=1, side="V")
        )(results)
        assert not torch.allclose(out[keys.PATH_PATCHED_LOGITS], clean, atol=1e-4)

    def test_query_and_value_receivers_differ(self, murano_model):
        _, results = _loaded()
        senders = NodeSet.expand_heads(0, range(murano_model.n_heads))
        q = PathPatch(
            murano_model, senders, receiver=Node(1, SELF_ATTN, head=1, side="Q")
        )(results)[keys.PATH_PATCHED_LOGITS]
        v = PathPatch(
            murano_model, senders, receiver=Node(1, SELF_ATTN, head=1, side="V")
        )(results)[keys.PATH_PATCHED_LOGITS]
        assert not torch.allclose(q, v, atol=1e-4)

    def test_different_receiver_heads_differ(self, murano_model):
        _, results = _loaded()
        senders = NodeSet.expand_heads(0, range(murano_model.n_heads))
        head0 = PathPatch(
            murano_model, senders, receiver=Node(1, SELF_ATTN, head=0, side="V")
        )(results)[keys.PATH_PATCHED_LOGITS]
        head1 = PathPatch(
            murano_model, senders, receiver=Node(1, SELF_ATTN, head=1, side="V")
        )(results)[keys.PATH_PATCHED_LOGITS]
        assert not torch.allclose(head0, head1, atol=1e-4)

    def test_gqa_shared_kv_receivers_alias(self, murano_model_attn_mqa):
        # num_key_value_heads=1: every query head's value resolves to the single
        # shared kv head, so a value receiver on head 0 and on head 3 patch the same
        # columns and must give identical logits (and move off the clean run). Guards
        # the query-head -> kv-head mapping end-to-end, not just the slot resolution.
        model = murano_model_attn_mqa
        _, results = _loaded()
        clean = _clean_logits(model, results)
        senders = NodeSet.expand_heads(0, range(model.n_heads))
        head0 = PathPatch(
            model, senders, receiver=Node(1, SELF_ATTN, head=0, side="V")
        )(results)[keys.PATH_PATCHED_LOGITS]
        head3 = PathPatch(
            model, senders, receiver=Node(1, SELF_ATTN, head=3, side="V")
        )(results)[keys.PATH_PATCHED_LOGITS]
        assert torch.allclose(head0, head3, atol=1e-5)
        assert not torch.allclose(head0, clean, atol=1e-4)


# ── Q/K/V receiver: direct-path recompute (gold standard) ─────────────


class TestQKVReceiverGoldStandard:
    @pytest.mark.parametrize("fixture", FIXTURES)
    def test_value_receiver_matches_direct_reconstruction(self, fixture, request):
        # With one upstream sender and everything else frozen, the direct-path
        # value is the base residual plus that head's (source - base) contribution,
        # which _direct_qkv_reference rebuilds independently. The value side moves
        # the logits enough that the match is a genuine check, not clean == clean.
        model = request.getfixturevalue(fixture)
        sender = Node(0, SELF_ATTN, head=0)
        receiver = Node(1, SELF_ATTN, head=1, side="V")
        _, results = _loaded()
        clean = _clean_logits(model, results)
        out = PathPatch(model, sender, receiver=receiver, freeze_mlps=True)(results)
        reference = _direct_qkv_reference(model, sender, receiver)
        assert torch.allclose(out[keys.PATH_PATCHED_LOGITS], reference, atol=1e-4)
        assert not torch.allclose(reference, clean, atol=1e-3)
