"""Tests for the attention-pattern analysis and intervention step.

RecordAttention captures per-head softmax weights [B, H, Q, K]; the
AttentionResult reductions (entropy, sink, distance, at_offset) are pinned
against hand-crafted patterns with known values, then sanity-checked on real
tiny models (both norm families, since eager attention is exercised on each).
AblateAttention writes the settable accessor: a base==source resample is an
exact no-op, zeroing a head moves the logits, and the alignment guard rejects
mismatched pairs. ov_circuit is checked on the separate-projection Llama and
must reject the fused-QKV GPT-2. All end-to-end runs are on CPU.
"""

from __future__ import annotations

import math

import pytest
import torch

from murano import Pipeline, keys
from murano.dataset import CleanCorruptDataset
from murano.io import load_attention, save_attention, save_results
from murano.nodes import SELF_ATTN, Node, NodeSet
from murano.results import Results
from murano.steps.attention import (
    AblateAttention,
    AttentionResult,
    RecordAttention,
    ov_circuit,
)
from murano.steps.logits import Logits
from murano.steps.paired import LoadPaired

# Equal token length per pair (known vocab words, no BOS) so positions align.
CLEAN = ["hello world", "good world"]
CORRUPT = ["good world", "bad world"]

# A more varied batch so a batch-mean pattern differs clearly from each example.
DIVERSE = ["hello world", "good world", "bad world", "world hello"]

FIXTURES = ["murano_model_attn", "gpt2_model_attn"]


def _loaded(clean=CLEAN, corrupt=CORRUPT):
    ds = CleanCorruptDataset(clean=clean, corrupt=corrupt)
    return LoadPaired(ds)(Results())


def _clean_logits(model, results):
    return Logits(model)(results)[keys.FINAL_LOGITS]


def _hand_result(pattern: torch.Tensor, mask: torch.Tensor) -> AttentionResult:
    """Wrap one layer's [B, H, S, S] pattern in an AttentionResult for reductions."""
    seq = pattern.shape[-1]
    return AttentionResult(
        patterns={0: pattern},
        attention_mask=mask,
        str_tokens=[[str(i) for i in range(seq)] for _ in range(pattern.shape[0])],
        layers=[0],
        addresses=[Node(0, SELF_ATTN)],
    )


# ── Contract ──────────────────────────────────────────────────────────


class TestContract:
    def test_default_read_write_keys(self, murano_model_attn):
        step = RecordAttention(murano_model_attn)
        assert step.reads == [keys.PROMPTS]
        assert step.writes == [keys.ATTENTION_PATTERN]
        assert step.write_types == {keys.ATTENTION_PATTERN: AttentionResult}

    def test_records_all_layers_with_head_axis(self, murano_model_attn):
        out = RecordAttention(murano_model_attn)(_loaded())
        result = out[keys.ATTENTION_PATTERN]
        assert result.layers == list(range(murano_model_attn.n_layers))
        for layer in result.layers:
            pattern = result.patterns[layer]
            assert pattern.shape[0] == len(CLEAN)
            assert pattern.shape[1] == murano_model_attn.n_heads
            assert pattern.shape[2] == pattern.shape[3]

    def test_layer_subset(self, murano_model_attn):
        result = RecordAttention(murano_model_attn, layers=[0])(_loaded())[
            keys.ATTENTION_PATTERN
        ]
        assert result.layers == [0]
        assert set(result.patterns) == {0}

    def test_validate_passes(self, murano_model_attn):
        ds = CleanCorruptDataset(clean=CLEAN, corrupt=CORRUPT)
        pipe = Pipeline([LoadPaired(ds), RecordAttention(murano_model_attn)])
        assert keys.ATTENTION_PATTERN in pipe.validate()

    def test_requires_attention_probs(self, murano_model):
        # The plain fixture is loaded without enable_attention_probs, so the step
        # must refuse rather than silently capturing nothing.
        with pytest.raises(RuntimeError, match="enable_attention_probs"):
            RecordAttention(murano_model, layers=[0])(_loaded())

    def test_bad_layers_string_raises(self, murano_model_attn):
        with pytest.raises(ValueError, match="must be 'all'"):
            RecordAttention(murano_model_attn, layers="first")

    def test_non_ascending_layers(self, murano_model_attn):
        # nnterp requires ascending read order inside a trace; the step must
        # capture correctly regardless of the requested layer order.
        result = RecordAttention(murano_model_attn, layers=[1, 0])(_loaded())[
            keys.ATTENTION_PATTERN
        ]
        assert result.layers == [1, 0]
        assert set(result.patterns) == {0, 1}


# ── Generic reductions (hand-crafted patterns) ────────────────────────


class TestReductions:
    def test_identity_pattern(self):
        # Each query attends only to itself: one-hot rows on the main diagonal.
        pattern = torch.eye(3).view(1, 1, 3, 3)
        result = _hand_result(pattern, torch.ones(1, 3))
        assert torch.allclose(result.entropy(), torch.zeros(1, 1), atol=1e-5)
        assert torch.allclose(result.at_offset(0), torch.ones(1, 1), atol=1e-5)
        assert torch.allclose(result.at_offset(-1), torch.zeros(1, 1), atol=1e-5)
        assert torch.allclose(result.distance(), torch.zeros(1, 1), atol=1e-5)
        # sink(0) is the mass on key 0: 1 for query 0, 0 for the other two rows.
        assert torch.allclose(result.sink(0), torch.full((1, 1), 1 / 3), atol=1e-5)

    def test_uniform_pattern_entropy(self):
        pattern = torch.full((1, 1, 3, 3), 1 / 3)
        result = _hand_result(pattern, torch.ones(1, 3))
        assert torch.allclose(
            result.entropy(), torch.full((1, 1), math.log(3)), atol=1e-5
        )

    def test_all_mass_on_key_zero(self):
        pattern = torch.zeros(1, 1, 3, 3)
        pattern[..., 0] = 1.0
        result = _hand_result(pattern, torch.ones(1, 3))
        assert torch.allclose(result.sink(0), torch.ones(1, 1), atol=1e-5)
        # Distance query - key with key fixed at 0 averages (0 + 1 + 2) / 3.
        assert torch.allclose(result.distance(), torch.full((1, 1), 1.0), atol=1e-5)

    def test_masking_excludes_padding(self):
        # Query position 2 is padding; the reductions must ignore it.
        pattern = torch.eye(3).view(1, 1, 3, 3)
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        result = _hand_result(pattern, mask)
        # sink(0) over valid queries {0, 1}: (1 + 0) / 2 = 0.5.
        assert torch.allclose(result.sink(0), torch.full((1, 1), 0.5), atol=1e-5)

    def test_sink_is_first_real_key_under_left_padding(self):
        # Left padding: columns 0-1 are <pad>, the first real key is column 2.
        # Every valid query attends fully to that first real key, so sink(0) = 1,
        # not 0 (which reading absolute column 0 would give).
        pattern = torch.zeros(1, 1, 3, 3)
        pattern[..., 2] = 1.0
        mask = torch.tensor([[0.0, 0.0, 1.0]])
        result = _hand_result(pattern, mask)
        assert torch.allclose(result.sink(0), torch.ones(1, 1), atol=1e-5)

    def test_attention_to_reads_the_named_cell(self):
        # Identity pattern: query q attends only to key q.
        result = _hand_result(torch.eye(3).view(1, 1, 3, 3), torch.ones(1, 3))
        assert torch.allclose(result.attention_to(1, 1), torch.ones(1, 1), atol=1e-5)
        assert torch.allclose(result.attention_to(2, 0), torch.zeros(1, 1), atol=1e-5)
        # Default query is the last real token (row 2), attending to itself.
        assert torch.allclose(result.attention_to(key=2), torch.ones(1, 1), atol=1e-5)

    def test_attention_to_per_example_positions(self):
        # Two examples with different query/key targets, resolved per example.
        pattern = torch.zeros(2, 1, 3, 3)
        pattern[0, 0, 1, 0] = 1.0  # example 0: query 1 -> key 0
        pattern[1, 0, 2, 1] = 1.0  # example 1: query 2 -> key 1
        result = _hand_result(pattern, torch.ones(2, 3))
        got = result.attention_to(query=[1, 2], key=[0, 1])
        assert torch.allclose(got, torch.ones(1, 1), atol=1e-5)


# ── Reductions on real models ─────────────────────────────────────────


class TestReductionsFromModel:
    @pytest.mark.parametrize("fixture", FIXTURES)
    def test_patterns_sum_to_one(self, fixture, request):
        model = request.getfixturevalue(fixture)
        result = RecordAttention(model)(_loaded())[keys.ATTENTION_PATTERN]
        for pattern in result.patterns.values():
            sums = pattern.sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    @pytest.mark.parametrize("fixture", FIXTURES)
    def test_reduction_shapes_and_ranges(self, fixture, request):
        model = request.getfixturevalue(fixture)
        result = RecordAttention(model)(_loaded())[keys.ATTENTION_PATTERN]
        expected = (model.n_layers, model.n_heads)
        assert result.entropy().shape == expected
        assert result.sink().shape == expected
        assert result.distance().shape == expected
        assert result.at_offset(-1).shape == expected
        assert result.attention_to().shape == expected
        # Sink mass is a probability; entropy is non-negative.
        assert (result.sink() >= -1e-5).all() and (result.sink() <= 1 + 1e-5).all()
        assert (result.entropy() >= -1e-5).all()


# ── OV circuit ────────────────────────────────────────────────────────


class TestOVCircuit:
    def test_shape_and_nonzero(self, murano_model_attn):
        ov = ov_circuit(murano_model_attn, 0, 1)
        d = murano_model_attn.d_model
        assert ov.shape == (d, d)
        assert ov.abs().sum() > 0

    def test_head_out_of_range(self, murano_model_attn):
        with pytest.raises(ValueError, match="out of range"):
            ov_circuit(murano_model_attn, 0, 99)

    def test_gpt2_fused_ov_matches_forward(self, gpt2_model_attn):
        # GPT-2 fuses q|k|v in a single Conv1D; the OV must match routing a
        # residual through the value block and the output projection by hand.
        model = gpt2_model_attn
        hd, d = model.head_dim, model.d_model
        attn = model.resolve_module(0, SELF_ATTN)._module
        c_attn = attn.c_attn.weight.detach().float()  # [d, 3d] Conv1D
        c_proj = attn.c_proj.weight.detach().float()  # [d, d] Conv1D
        for head in range(model.n_heads):
            x = torch.randn(d)
            v_head = (x @ c_attn[:, 2 * d : 3 * d])[head * hd : (head + 1) * hd]
            z = torch.zeros(d)
            z[head * hd : (head + 1) * hd] = v_head
            expected = z @ c_proj
            assert torch.allclose(ov_circuit(model, 0, head) @ x, expected, atol=1e-4)

    def test_grouped_query_kv_mapping(self, murano_model_attn_mqa):
        # Multi-query attention (n_kv_heads=1): every query head shares the single
        # kv head, so its OV must use the only value slice v_proj[0:head_dim].
        model = murano_model_attn_mqa
        head_dim = model.head_dim
        w_v = (
            model.resolve_module(0, "self_attn.v_proj")._module.weight.detach().float()
        )
        w_o = model.attn_out_proj(0, SELF_ATTN)._module.weight.detach().float()
        for head in range(model.n_heads):
            expected = w_o[:, head * head_dim : (head + 1) * head_dim] @ w_v[:head_dim]
            assert torch.allclose(ov_circuit(model, 0, head), expected, atol=1e-5)


# ── Intervention ──────────────────────────────────────────────────────


class TestAblateAttention:
    def test_default_read_write_keys(self, murano_model_attn):
        step = AblateAttention(murano_model_attn, Node(0, SELF_ATTN, head=0))
        assert step.reads == [keys.PROMPTS]
        assert step.writes == [keys.ATTN_ABLATED_LOGITS, keys.ATTN_ABLATED_MASK]

    @pytest.mark.parametrize("fixture", FIXTURES)
    def test_zero_changes_logits(self, fixture, request):
        model = request.getfixturevalue(fixture)
        results = _loaded()
        clean = _clean_logits(model, results)
        out = AblateAttention(model, Node(0, SELF_ATTN, head=0))(results)
        assert not torch.allclose(out[keys.ATTN_ABLATED_LOGITS], clean, atol=1e-5)

    def test_headless_target_zeros_all_heads(self, murano_model_attn):
        results = _loaded()
        clean = _clean_logits(murano_model_attn, results)
        out = AblateAttention(murano_model_attn, Node(0, SELF_ATTN))(results)
        assert not torch.allclose(out[keys.ATTN_ABLATED_LOGITS], clean, atol=1e-5)

    def test_resample_base_equals_source_is_noop(self, murano_model_attn):
        # Injecting the base run's own pattern is the identity: logits unchanged.
        results = _loaded()
        clean = _clean_logits(murano_model_attn, results)
        out = AblateAttention(
            murano_model_attn,
            Node(0, SELF_ATTN, head=0),
            method="resample",
            source_key=keys.PROMPTS,
        )(results)
        assert torch.allclose(out[keys.ATTN_ABLATED_LOGITS], clean, atol=1e-4)

    def test_resample_from_corrupt_changes_logits(self, murano_model_attn):
        results = _loaded()
        clean = _clean_logits(murano_model_attn, results)
        out = AblateAttention(
            murano_model_attn,
            Node(0, SELF_ATTN, head=0),
            method="resample",
            source_key=keys.CORRUPT_PROMPTS,
        )(results)
        assert not torch.allclose(out[keys.ATTN_ABLATED_LOGITS], clean, atol=1e-5)

    def test_mean_changes_logits(self, murano_model_attn):
        # Mean over a varied batch differs clearly from each example's pattern.
        results = LoadPaired(CleanCorruptDataset(clean=DIVERSE, corrupt=DIVERSE))(
            Results()
        )
        clean = _clean_logits(murano_model_attn, results)
        out = AblateAttention(murano_model_attn, Node(0, SELF_ATTN), method="mean")(
            results
        )
        assert not torch.allclose(out[keys.ATTN_ABLATED_LOGITS], clean, atol=1e-5)

    def test_multilayer_ablation_differs_from_single(self, murano_model_attn):
        # Editing both layers must differ from editing either layer alone, so the
        # per-layer edits dict is genuinely applied layer by layer.
        both = AblateAttention(
            murano_model_attn,
            [Node(0, SELF_ATTN, head=0), Node(1, SELF_ATTN, head=0)],
        )(_loaded())[keys.ATTN_ABLATED_LOGITS]
        layer0 = AblateAttention(murano_model_attn, Node(0, SELF_ATTN, head=0))(
            _loaded()
        )[keys.ATTN_ABLATED_LOGITS]
        layer1 = AblateAttention(murano_model_attn, Node(1, SELF_ATTN, head=0))(
            _loaded()
        )[keys.ATTN_ABLATED_LOGITS]
        assert not torch.allclose(both, layer0, atol=1e-5)
        assert not torch.allclose(both, layer1, atol=1e-5)

    def test_headless_equals_expand_heads(self, murano_model_attn):
        # A head-less target must resolve to exactly every head at the layer.
        headless = AblateAttention(murano_model_attn, Node(0, SELF_ATTN))(_loaded())[
            keys.ATTN_ABLATED_LOGITS
        ]
        all_heads = AblateAttention(
            murano_model_attn, NodeSet.expand_heads(0, range(murano_model_attn.n_heads))
        )(_loaded())[keys.ATTN_ABLATED_LOGITS]
        assert torch.allclose(headless, all_heads, atol=1e-6)

    def test_bf16_intervention_runs(self, murano_model_attn_bf16):
        # The attention weights are bf16, so a float32 replacement must be coerced
        # before the blend; the run must complete and return finite logits.
        out = AblateAttention(murano_model_attn_bf16, Node(0, SELF_ATTN, head=0))(
            _loaded()
        )
        logits = out[keys.ATTN_ABLATED_LOGITS]
        assert torch.isfinite(logits).all()

    def test_positions_restrict_intervention(self, murano_model_attn):
        results = _loaded()
        all_pos = AblateAttention(murano_model_attn, Node(0, SELF_ATTN, head=0))(
            results
        )[keys.ATTN_ABLATED_LOGITS]
        last_pos = AblateAttention(
            murano_model_attn, Node(0, SELF_ATTN, head=0), positions=[-1]
        )(_loaded())[keys.ATTN_ABLATED_LOGITS]
        assert not torch.allclose(all_pos, last_pos, atol=1e-5)

    def test_writes_mask(self, murano_model_attn):
        out = AblateAttention(murano_model_attn, Node(0, SELF_ATTN, head=0))(_loaded())
        expected = murano_model_attn.tokenizer(
            CLEAN,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_token_type_ids=False,
        )["attention_mask"]
        assert torch.equal(out[keys.ATTN_ABLATED_MASK], expected)

    def test_bad_method_raises(self, murano_model_attn):
        with pytest.raises(ValueError, match="method"):
            AblateAttention(murano_model_attn, Node(0, SELF_ATTN, head=0), method="x")

    def test_resample_without_source_raises(self, murano_model_attn):
        with pytest.raises(ValueError, match="resample"):
            AblateAttention(
                murano_model_attn, Node(0, SELF_ATTN, head=0), method="resample"
            )

    def test_source_and_source_key_conflict(self, murano_model_attn):
        with pytest.raises(ValueError, match="either"):
            AblateAttention(
                murano_model_attn,
                Node(0, SELF_ATTN, head=0),
                method="resample",
                source=CORRUPT,
                source_key=keys.CORRUPT_PROMPTS,
            )

    def test_empty_targets_raises(self, murano_model_attn):
        with pytest.raises(ValueError, match="at least one"):
            AblateAttention(murano_model_attn, [])

    def test_non_attention_target_raises(self, murano_model_attn):
        with pytest.raises(ValueError, match="attention head"):
            AblateAttention(murano_model_attn, Node(0, "mlp"))

    def test_head_out_of_range_raises(self, murano_model_attn):
        with pytest.raises(ValueError, match="out of range"):
            AblateAttention(murano_model_attn, Node(0, SELF_ATTN, head=99))

    def test_head_side_rejected(self, murano_model_attn):
        with pytest.raises(ValueError, match="head-side"):
            AblateAttention(murano_model_attn, Node(0, SELF_ATTN, head=0, side="V"))

    def test_sender_position_rejected(self, murano_model_attn):
        with pytest.raises(ValueError, match="position"):
            AblateAttention(murano_model_attn, Node(0, SELF_ATTN, head=0, position=1))


# ── Alignment guard ───────────────────────────────────────────────────


class TestAlignment:
    def test_unequal_source_length_raises(self, murano_model_attn):
        # Equal prompt count but mismatched per-example token length (1 vs 2), so
        # the length guard fires rather than the count guard.
        results = _loaded()
        with pytest.raises(ValueError, match="token-length-matched"):
            AblateAttention(
                murano_model_attn,
                Node(0, SELF_ATTN, head=0),
                method="resample",
                source=["hello", "good"],
            )(results)

    def test_unequal_source_count_raises(self, murano_model_attn):
        results = _loaded()
        with pytest.raises(ValueError, match="has 1 prompts"):
            AblateAttention(
                murano_model_attn,
                Node(0, SELF_ATTN, head=0),
                method="resample",
                source=["hello"],
            )(results)


# ── Persistence + plotting ────────────────────────────────────────────


class TestIO:
    def test_roundtrip(self, tmp_path, murano_model_attn):
        result = RecordAttention(murano_model_attn, layers=[0])(_loaded())[
            keys.ATTENTION_PATTERN
        ]
        path = tmp_path / "attention.pt"
        save_attention(result, path)
        loaded = load_attention(path)
        assert torch.allclose(loaded.patterns[0], result.patterns[0])
        assert torch.equal(loaded.attention_mask, result.attention_mask)
        assert loaded.str_tokens == result.str_tokens
        assert loaded.layers == result.layers
        assert loaded.addresses == result.addresses
        assert loaded.metadata == result.metadata

    def test_save_results_registers_artifact(self, tmp_path, murano_model_attn):
        # Exercise the serializer registry: save_results must persist the
        # AttentionResult rather than silently dropping an unregistered type.
        results = RecordAttention(murano_model_attn, layers=[0, 1])(_loaded())
        save_results(results, output_dir=str(tmp_path))
        assert (tmp_path / "attention" / "attention.pt").exists()
        reloaded = load_attention(tmp_path / "attention" / "attention.pt")
        assert reloaded.layers == [0, 1]


class TestPlotting:
    def test_plot_smoke(self, murano_model_attn):
        pytest.importorskip("plotly")
        from murano.plotting.attention import (
            plot_attention_pattern,
            plot_head_matrix,
        )

        result = RecordAttention(murano_model_attn, layers=[0])(_loaded())[
            keys.ATTENTION_PATTERN
        ]
        assert plot_attention_pattern(result, 0, 0) is not None
        assert plot_head_matrix(result.entropy(), layers=result.layers) is not None

    def test_plot_head_matrix_forwards_zmid(self, murano_model_attn):
        pytest.importorskip("plotly")
        from murano.plotting.attention import plot_head_matrix

        result = RecordAttention(murano_model_attn, layers=[0])(_loaded())[
            keys.ATTENTION_PATTERN
        ]
        fig = plot_head_matrix(result.entropy(), color_scale="RdBu", zmid=0)
        assert fig.data[0].zmid == 0


# ── Pipeline ──────────────────────────────────────────────────────────


class TestPipeline:
    def test_record_in_pipeline(self, murano_model_attn):
        ds = CleanCorruptDataset(clean=CLEAN, corrupt=CORRUPT)
        results = Pipeline(
            [LoadPaired(ds), RecordAttention(murano_model_attn, layers=[0])]
        ).run()
        result = results[keys.ATTENTION_PATTERN]
        assert result.entropy().shape == (1, murano_model_attn.n_heads)


def test_nodeset_targets_group_by_layer(murano_model_attn):
    # A NodeSet of heads is a valid target set and runs end to end.
    heads = NodeSet.expand_heads(0, range(murano_model_attn.n_heads))
    out = AblateAttention(murano_model_attn, heads)(_loaded())
    assert keys.ATTN_ABLATED_LOGITS in out
