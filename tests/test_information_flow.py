"""Tests for the InformationFlow step.

The importance math (Eq. 1, the renormalizing threshold) is pinned on hand-made
vectors against the dossier's oracles, which come from the reference repo's unit
tests. On the tiny Llama and GPT-2 fixtures the per-head parts are checked to
sum to the recorded attention output, and the route walk is checked for
completeness at tau = 0 and monotonicity in tau. Pattern capture itself is
covered by tests/test_attention.py.
"""

from __future__ import annotations

import pytest
import torch

from murano import Pipeline, keys
from murano.io import load_information_flow, save_results
from murano.nodes import MLP, RESID_MID, RESID_POST, SELF_ATTN, Node
from murano.results import Results
from murano.steps.information_flow import (
    InformationFlow,
    InformationFlowResult,
    _head_outputs,
    _importance,
    _threshold,
)
from murano.steps.prompts import LoadPrompts
from murano.steps.select import SelectComponents

# Fixture tokenizer vocab: hello world good bad prompt response (+ specials).
PROMPTS = ["hello world good bad", "good world hello prompt response"]
FIXTURES = ["murano_model_attn", "gpt2_model_attn"]


def _run(model, prompts=PROMPTS, **kwargs) -> InformationFlowResult:
    out = InformationFlow(model, **kwargs)(LoadPrompts(list(prompts))(Results()))
    return out[keys.INFORMATION_FLOW]


def _edge_set(result: InformationFlowResult, index: int = 0) -> set:
    return {(e.source, e.source_pos, e.dest, e.dest_pos) for e in result.edges[index]}


class TestInformationFlowContract:
    def test_default_keys(self, murano_model_attn):
        step = InformationFlow(murano_model_attn)
        assert step.reads == [keys.PROMPTS]
        assert step.writes == [keys.INFORMATION_FLOW]
        assert step.write_types[keys.INFORMATION_FLOW] is InformationFlowResult

    def test_custom_keys(self, murano_model_attn):
        step = InformationFlow(murano_model_attn, prompts_key="p", output_key="routes")
        assert step.reads == ["p"]
        assert step.writes == ["routes"]

    def test_pipeline_result_covers_every_component(self, gpt2_model_attn):
        model = gpt2_model_attn
        out = Pipeline([LoadPrompts(PROMPTS), InformationFlow(model)]).run()
        result = out[keys.INFORMATION_FLOW]
        heads = {
            Node(layer, SELF_ATTN, head=h)
            for layer in range(model.n_layers)
            for h in range(model.n_heads)
        }
        mlps = {Node(layer, MLP) for layer in range(model.n_layers)}
        assert set(result.contributions) == heads | mlps
        assert set(result.head_importance) == heads
        assert all(0.0 <= v <= 1.0 for v in result.contributions.values())
        assert len(result.edges) == len(result.tokens) == len(PROMPTS)
        # Default start is each prompt's last real token, padding excluded.
        assert result.positions == [len(t) - 1 for t in result.tokens]

    def test_select_components_ranks_contributions(self, gpt2_model_attn):
        out = Pipeline(
            [
                LoadPrompts(PROMPTS),
                InformationFlow(gpt2_model_attn, threshold=0.0),
                SelectComponents(keys.INFORMATION_FLOW, top_k=3, modules=SELF_ATTN),
            ]
        ).run()
        selection = out[keys.SELECTION]
        flow = out[keys.INFORMATION_FLOW]
        top = sorted(
            (v for n, v in flow.contributions.items() if n.module == SELF_ATTN),
            reverse=True,
        )[:3]
        assert sorted(selection.scores.values(), reverse=True) == top


class TestInformationFlowMechanism:
    def test_importance_sums_to_one(self):
        torch.manual_seed(0)
        whole = torch.randn(5, 16)
        parts = torch.randn(5, 7, 16) * 0.3 + whole.unsqueeze(1) / 7
        proximity = whole.norm(p=1, dim=-1, keepdim=True) - (
            whole.unsqueeze(1) - parts
        ).norm(p=1, dim=-1)
        # Precondition: some parts are close enough to score.
        assert (proximity > 0).any(dim=-1).all()
        importance = _importance(parts, whole)
        # Oracle 1: per-node importances sum to 1 over the parts.
        assert torch.allclose(importance.sum(-1), torch.ones(5), atol=1e-6)
        assert (importance >= 0).all()

    @pytest.mark.parametrize(
        ("order", "expected"),
        # Oracle 2: llm-transparency-tool@f1340f0 routes/test_contributions.py
        # test_decomposed_attn_contributions (L2); L1 hand-derived (proximities 2, 0, 3).
        [(2, [0.43613, 0.0, 0.56387]), (1, [0.4, 0.0, 0.6])],
    )
    def test_attention_importance_reference_values(self, order, expected):
        whole = torch.tensor([2.0, 2.0])  # resid_mid
        parts = torch.tensor([[1.0, 1.0], [-1.0, 0.0], [2.0, 1.0]])  # heads, resid_pre
        importance = _importance(parts, whole, order)
        assert torch.allclose(importance, torch.tensor(expected), atol=1e-4)

    def test_ffn_importance_reference_values(self):
        # Oracle 3: llm-transparency-tool@f1340f0 routes/test_contributions.py.
        ffn = _importance(
            torch.tensor([[1.0, 1.0], [0.0, 0.0]]), torch.tensor([1.0, 1.0])
        )
        assert torch.allclose(ffn, torch.tensor([1.0, 0.0]), atol=1e-4)
        parts = torch.tensor([[1.0, 1.0], [2.0, 2.0], [1.0, 1.0]])
        split = _importance(parts, torch.tensor([4.0, 4.0]), order=2)
        assert torch.allclose(split, torch.tensor([0.25, 0.5, 0.25]), atol=1e-4)

    def test_renormalizing_threshold(self):
        # Oracle 4: llm-transparency-tool@f1340f0 test_renormalizing_threshold.
        importance = torch.tensor([[0.05, 0.15, 0.8], [0.05, 0.05, 0.9]])
        out = _threshold(importance, 0.1)
        expected = torch.tensor([[0.0, 0.157894, 0.842105], [0.0, 0.0, 1.0]])
        assert torch.allclose(out, expected, atol=1e-4)

    @pytest.mark.parametrize("fixture", FIXTURES)
    def test_head_parts_sum_to_attention_output(self, fixture, request):
        model = request.getfixturevalue(fixture)
        step = InformationFlow(model)
        tokens = model.tokenizer(["hello world good bad"], return_tensors="pt")
        out_proj = [
            model.raw_attn_out_proj(layer, SELF_ATTN) for layer in range(model.n_layers)
        ]
        biases = [
            p.bias
            for layer in range(model.n_layers)
            for p in (
                model.raw_attn_out_proj(layer, SELF_ATTN),
                getattr(model.raw_module(layer, SELF_ATTN), "c_attn", None),
            )
            if p is not None and p.bias is not None
        ]
        saved = [b.detach().clone() for b in biases]
        torch.manual_seed(0)
        try:
            # GPT-2 initializes biases to zero; make b_V and b_O matter.
            with torch.no_grad():
                for b in biases:
                    b.normal_(0, 0.5)
            resid, patterns = step._capture(tokens)
            for layer in range(model.n_layers):
                pre, attn_in, mid, _ = (r[0] for r in resid[layer])
                w_v, b_v, w_o = step._value_output_weights(layer)
                head_out = _head_outputs(attn_in, w_v, b_v, w_o)  # [S, H, d]
                pattern = patterns[layer][0]  # [H, Q, K]
                summed = torch.einsum("hqk,khd->qd", pattern, head_out)
                b_o = out_proj[layer].bias
                target = mid - pre - (b_o if b_o is not None else 0)
                # Oracle 6: sum_h sum_j alpha f^h(x_j) == attn_out - b_O.
                assert torch.allclose(summed, target, atol=1e-4)
                # Precondition: a wrong value weight for one head breaks it.
                w_v_bad = w_v.clone()
                w_v_bad[0] += torch.randn_like(w_v_bad[0])
                broken = torch.einsum(
                    "hqk,khd->qd", pattern, _head_outputs(attn_in, w_v_bad, b_v, w_o)
                )
                assert not torch.allclose(broken, target, atol=1e-4)
        finally:
            with torch.no_grad():
                for b, s in zip(biases, saved):
                    b.copy_(s)

    @pytest.mark.parametrize("fixture", FIXTURES)
    def test_zero_threshold_reaches_every_causal_node(self, fixture, request):
        model = request.getfixturevalue(fixture)
        result = _run(model, prompts=["hello world good bad"], threshold=0.0)
        start = result.positions[0]
        reached = {(e.dest, e.dest_pos) for e in result.edges[0]}
        # Oracle 5: softmax weights are positive, so at tau = 0 the routes reach
        # every resid_mid node at or before the start position. The top layer
        # only has the start node: nothing reads the other last-layer residuals.
        last = model.n_layers - 1
        expected = {(Node(last, RESID_MID), start)} | {
            (Node(layer, RESID_MID), pos)
            for layer in range(last)
            for pos in range(start + 1)
        }
        assert expected <= reached

    @pytest.mark.parametrize("fixture", FIXTURES)
    def test_incoming_edges_sum_to_one(self, fixture, request):
        model = request.getfixturevalue(fixture)
        result = _run(model, threshold=0.0, renormalize=False)
        incoming: dict = {}
        for edges in result.edges[:1]:
            for e in edges:
                if e.dest.module != MLP:
                    key = (e.dest, e.dest_pos)
                    incoming[key] = incoming.get(key, 0.0) + e.importance
        # Precondition: the routes reach resid_mid and resid_post nodes.
        assert {node.module for node, _ in incoming} == {RESID_MID, RESID_POST}
        # Oracle 1 on the model (Eq. 4): every node's parts, heads summed and
        # the residual merged into the self edge, account for all of it.
        for total in incoming.values():
            assert total == pytest.approx(1.0, abs=1e-4)

    @pytest.mark.parametrize("renormalize", [True, False])
    def test_routes_shrink_as_threshold_grows(self, gpt2_model_attn, renormalize):
        low = _run(gpt2_model_attn, threshold=0.0, renormalize=renormalize)
        high = _run(gpt2_model_attn, threshold=0.1, renormalize=renormalize)
        for index in range(len(PROMPTS)):
            # Precondition: the higher threshold drops something.
            assert _edge_set(high, index) < _edge_set(low, index)
            if not renormalize:
                # Without renormalization each kept edge is a kept edge at tau=0.
                assert _edge_set(high, index) <= _edge_set(low, index)
        assert all(e.importance > 0.1 for edges in high.edges for e in edges)

    def test_mlp_frequency_counts_prompts_through_the_mlp(self, gpt2_model_attn):
        result = _run(gpt2_model_attn, threshold=0.02)
        for layer in range(gpt2_model_attn.n_layers):
            node = Node(layer, MLP)
            through = sum(
                any(e.source == node for e in edges) for edges in result.edges
            )
            assert result.contributions[node] == pytest.approx(through / len(PROMPTS))
        # Precondition: the threshold separates used from unused MLPs somewhere.
        assert (
            len(
                {
                    round(v, 3)
                    for n, v in result.contributions.items()
                    if n.module == MLP
                }
            )
            > 1
        )


class TestInformationFlowSave:
    def test_round_trip(self, gpt2_model_attn, tmp_path):
        result = _run(gpt2_model_attn)
        results = Results()
        results[keys.INFORMATION_FLOW] = result
        save_results(results, output_dir=str(tmp_path))
        loaded = load_information_flow(
            tmp_path / "information_flow" / "information_flow.json"
        )
        assert loaded.contributions == result.contributions
        assert loaded.head_importance == result.head_importance
        assert loaded.edges == result.edges
        assert loaded.tokens == result.tokens
        assert loaded.positions == result.positions
        assert loaded.threshold == result.threshold
        assert loaded.metadata == result.metadata


class TestInformationFlowPlot:
    def test_figure(self, gpt2_model_attn):
        pytest.importorskip("plotly")
        from murano.plotting import plot_information_flow

        result = _run(gpt2_model_attn)
        z = plot_information_flow(result).to_dict()["data"][0]["z"]
        assert len(z) == gpt2_model_attn.n_layers
        assert z[1][2] == pytest.approx(
            result.contributions[Node(1, SELF_ATTN, head=2)]
        )


class TestInformationFlowErrors:
    @pytest.mark.parametrize("threshold", [-0.1, 1.0])
    def test_bad_threshold(self, murano_model_attn, threshold):
        with pytest.raises(ValueError, match=r"threshold must be in \[0, 1\)"):
            InformationFlow(murano_model_attn, threshold=threshold)

    def test_bad_norm_order(self, murano_model_attn):
        with pytest.raises(ValueError, match="norm_order must be 1"):
            InformationFlow(murano_model_attn, norm_order=3)

    def test_padding_start_position(self, gpt2_model_attn):
        # The tokenizer pads on the left, so column 0 of the shorter prompt is padding.
        with pytest.raises(ValueError, match="is padding"):
            _run(gpt2_model_attn, positions=0)

    def test_needs_attention_probs(self, murano_model):
        with pytest.raises(RuntimeError, match="enable_attention_probs"):
            _run(murano_model)
