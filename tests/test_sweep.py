"""Tests for the Sweep step and the SweepResult artifact.

The swept chain is built from stub steps rather than a model, so the sweep's own
contract (forking, harvesting, pre-flight reads, leak isolation) is checked
directly. One integration test runs the tiny fixture model end to end.
"""

from __future__ import annotations

import json

import pytest

from murano import Node, NodeSet, Pipeline, keys
from murano.artifacts import MetricScore, SweepResult
from murano.io import load_sweep_result, save_sweep_result
from murano.results import Results
from murano.steps.base import Step
from murano.steps.logits import Logits
from murano.steps.metrics import LogitDiffStep
from murano.steps.prompts import LoadPrompts
from murano.steps.select import SelectComponents
from murano.steps.sweep import Sweep


class Seed(Step):
    """Write the baseline the swept chain reads."""

    reads: list[str] = []
    writes = ["base"]

    def __call__(self, results: Results) -> Results:
        results["base"] = 10.0
        return results


class Score(Step):
    """Score one item against the baseline, into a MetricScore and a raw float."""

    reads = ["base"]
    writes = ["score", "raw"]

    def __init__(self, value: float):
        self.value = value

    def __call__(self, results: Results) -> Results:
        results["score"] = MetricScore("score", value=self.value - results["base"])
        results["raw"] = self.value
        return results


class WritesSomethingElse(Step):
    reads = ["base"]
    writes = ["elsewhere"]

    def __call__(self, results: Results) -> Results:
        results["elsewhere"] = 1.0
        return results


class WritesAnObject(Step):
    reads: list[str] = []
    writes = ["thing"]

    def __call__(self, results: Results) -> Results:
        results["thing"] = object()
        return results


def _heads(n_layers: int, n_heads: int) -> NodeSet:
    return NodeSet.product(range(n_layers), ["self_attn"], heads=range(n_heads))


def _sweep(over, read="score") -> SweepResult:
    pipe = Pipeline(
        [Seed(), Sweep(over=over, steps=lambda i: [Score(_value(i))], read=read)]
    )
    return pipe.run()[keys.SWEEP]


def _value(item) -> float:
    """A per-item value that is distinct and cheap to predict in assertions."""
    if isinstance(item, Node):
        return item.layer * 10.0 + (item.head or 0)
    return float(len(str(item)))


# ── Sweep contract ────────────────────────────────────────────────────


class TestSweepContract:
    def test_reads_what_the_swept_chain_needs_from_outside(self):
        step = Sweep(over=[1], steps=lambda i: [Score(1.0)], read="score")
        assert step.reads == ["base"]
        assert step.writes == [keys.SWEEP]

    def test_a_key_the_chain_writes_itself_is_not_an_external_read(self):
        step = Sweep(over=[1], steps=lambda i: [Seed(), Score(1.0)], read="score")
        assert step.reads == []

    def test_pipeline_validation_rejects_a_missing_upstream_key(self):
        pipe = Pipeline([Sweep(over=[1], steps=lambda i: [Score(1.0)], read="score")])
        with pytest.raises(KeyError, match="requires 'base'"):
            pipe.validate()

    def test_pipeline_validation_passes_when_the_prefix_supplies_it(self):
        pipe = Pipeline(
            [Seed(), Sweep(over=[1], steps=lambda i: [Score(1.0)], read="score")]
        )
        assert keys.SWEEP in pipe.validate()

    def test_each_item_starts_from_the_same_baseline(self):
        result = _sweep([Node(0, "self_attn", head=1), Node(2, "self_attn", head=3)])
        assert result.scores[Node(0, "self_attn", head=1)] == 1.0 - 10.0
        assert result.scores[Node(2, "self_attn", head=3)] == 23.0 - 10.0

    def test_the_swept_chains_writes_do_not_leak_into_the_pipeline(self):
        out = Pipeline(
            [
                Seed(),
                Sweep(over=[1, 2], steps=lambda i: [Score(_value(i))], read="score"),
            ]
        ).run()
        assert "score" not in out and "raw" not in out
        assert out["base"] == 10.0

    def test_harvests_several_keys_in_one_pass(self):
        result = _sweep([1, 22], read=["score", "raw"])
        assert result.primary == "score"
        assert sorted(result.columns) == ["raw", "score"]
        assert result.column("raw")[22] == 2.0

    def test_a_metric_score_is_coerced_to_its_value(self):
        assert _sweep([1]).scores[1] == pytest.approx(1.0 - 10.0)

    def test_a_single_step_may_be_returned_unwrapped(self):
        pipe = Pipeline(
            [Seed(), Sweep(over=[1], steps=lambda i: Score(1.0), read="score")]
        )
        assert pipe.run()[keys.SWEEP].scores[1] == -9.0

    def test_metadata_records_the_chain_and_the_items(self):
        meta = _sweep(["a", "bb"]).metadata
        assert meta["n_items"] == 2
        assert meta["items"] == ["a", "bb"]
        assert meta["steps"] == ["Score"]
        assert meta["read"] == ["score"]


# ── Sweep guardrails ──────────────────────────────────────────────────


class TestSweepGuardrails:
    def test_empty_item_list_raises(self):
        with pytest.raises(ValueError, match="at least one item"):
            Sweep(over=[], steps=lambda i: [Score(1.0)], read="score")

    def test_empty_read_list_raises(self):
        with pytest.raises(ValueError, match="at least one key"):
            Sweep(over=[1], steps=lambda i: [Score(1.0)], read=[])

    def test_harvesting_a_key_the_chain_never_writes_raises(self):
        """Otherwise every item silently records the same baseline value."""
        with pytest.raises(ValueError, match="never writes"):
            Sweep(over=[1], steps=lambda i: [WritesSomethingElse()], read="score")

    def test_an_empty_chain_raises(self):
        with pytest.raises(ValueError, match="built no step"):
            Sweep(over=[1], steps=lambda i: [], read="score")

    def test_a_non_step_in_the_chain_raises(self):
        with pytest.raises(TypeError, match="must build Step objects"):
            Sweep(over=[1], steps=lambda i: ["not a step"], read="score")

    def test_harvesting_a_non_numeric_key_raises(self):
        pipe = Pipeline(
            [Sweep(over=[1], steps=lambda i: [WritesAnObject()], read="thing")]
        )
        with pytest.raises(TypeError, match="harvests 'thing' as a number"):
            pipe.run()


# ── SweepResult ───────────────────────────────────────────────────────


class TestSweepResult:
    def test_a_node_keyed_sweep_exposes_contributions(self):
        result = _sweep(_heads(2, 2))
        assert result.contributions is not None
        assert result.contributions is result.scores

    def test_contributions_accepts_node_shorthand(self):
        result = _sweep(_heads(2, 2))
        assert (
            result.scores["L1.self_attn.h1"]
            == result.scores[Node(1, "self_attn", head=1)]
        )

    def test_a_plain_sweep_has_no_contributions(self):
        assert _sweep(["zero", "mean"]).contributions is None

    def test_swept_preserves_the_sweep_order(self):
        assert _sweep(["b", "aa", "c"]).swept == ["b", "aa", "c"]

    def test_swept_is_not_named_items(self):
        """A mapping-shaped ``items`` would make the artifact duck-type as a dict."""
        result = _sweep([1])
        assert not hasattr(result, "items")

    def test_unknown_column_names_what_was_harvested(self):
        with pytest.raises(KeyError, match=r"it read \['score'\]"):
            _sweep([1]).column("nope")

    def test_primary_must_be_a_harvested_column(self):
        with pytest.raises(ValueError, match="not one of the harvested columns"):
            SweepResult(columns={"a": {1: 2.0}}, primary="b")

    def test_head_matrix_places_each_score_at_its_address(self):
        grid = _sweep(_heads(3, 2)).head_matrix()
        assert len(grid) == 3 and len(grid[0]) == 2
        assert grid[2][1] == 21.0 - 10.0

    def test_head_matrix_leaves_unswept_cells_blank(self):
        """NaN renders as a gap, so an unswept head never reads as a dead head."""
        grid = _sweep([Node(2, "self_attn", head=1)]).head_matrix()
        assert grid[2][1] == 11.0
        assert all(value != value for row in grid for value in row if value != 11.0)

    def test_head_matrix_accepts_explicit_dimensions(self):
        grid = _sweep([Node(0, "self_attn", head=0)]).head_matrix(n_layers=4, n_heads=3)
        assert len(grid) == 4 and len(grid[0]) == 3

    def test_head_matrix_rejects_dimensions_that_drop_a_score(self):
        with pytest.raises(ValueError, match="does not fit"):
            _sweep(_heads(3, 2)).head_matrix(n_layers=2, n_heads=2)

    def test_head_matrix_rejects_a_sweep_that_is_not_over_heads(self):
        with pytest.raises(TypeError, match="needs a sweep over attention heads"):
            _sweep(["zero"]).head_matrix()


# ── Composition with the rest of the library ──────────────────────────


class TestComposition:
    def test_select_components_ranks_a_component_sweep(self):
        out = Pipeline(
            [
                Seed(),
                Sweep(
                    over=_heads(3, 2), steps=lambda n: [Score(_value(n))], read="score"
                ),
                SelectComponents(source_key=keys.SWEEP, top_k=2, by="signed"),
            ]
        ).run()
        assert [str(n) for n in out[keys.SELECTION].nodes] == [
            "L2.self_attn.h1",
            "L2.self_attn.h0",
        ]

    def test_select_components_rejects_a_sweep_that_is_not_over_nodes(self):
        out = Pipeline(
            [Seed(), Sweep(over=["zero"], steps=lambda i: [Score(1.0)], read="score")]
        ).run()
        with pytest.raises(TypeError, match="not Node addresses"):
            SelectComponents(source_key=keys.SWEEP, top_k=1)(out)

    def test_select_components_validates_the_sweep_type(self):
        pipe = Pipeline(
            [
                Seed(),
                Sweep(
                    over=_heads(2, 2), steps=lambda n: [Score(_value(n))], read="score"
                ),
                SelectComponents(source_key=keys.SWEEP, top_k=1),
            ]
        )
        assert keys.SELECTION in pipe.validate()


# ── Persistence ───────────────────────────────────────────────────────


class TestPersistence:
    def test_component_sweep_round_trips_with_node_keys(self, tmp_path):
        original = _sweep(_heads(2, 2), read=["score", "raw"])
        path = tmp_path / "sweep.json"
        save_sweep_result(original, path)
        restored = load_sweep_result(path)

        assert restored.primary == original.primary
        assert restored.contributions is not None
        assert restored.scores == original.scores
        assert restored.column("raw") == original.column("raw")

    def test_plain_sweep_round_trips_with_string_labels(self, tmp_path):
        original = _sweep(["zero", "mean"])
        path = tmp_path / "sweep.json"
        save_sweep_result(original, path)
        restored = load_sweep_result(path)

        assert restored.contributions is None
        assert restored.scores == original.scores

    def test_saved_file_is_plain_json(self, tmp_path):
        path = tmp_path / "sweep.json"
        save_sweep_result(_sweep([1]), path)
        assert json.loads(path.read_text())["primary"] == "score"

    def test_save_results_writes_the_sweep(self, tmp_path):
        out = Pipeline(
            [Seed(), Sweep(over=[1], steps=lambda i: [Score(1.0)], read="score")]
        ).run()
        out.save(output_dir=str(tmp_path))
        assert (tmp_path / "sweep" / "sweep.json").exists()


# ── Integration ───────────────────────────────────────────────────────


class TestIntegration:
    def test_sweeping_an_intervened_forward_pass_over_layers(self, murano_model):
        """Edit each layer's residual in turn: one forward pass per layer, one artifact."""
        out = Pipeline(
            [
                LoadPrompts(["hello world", "good world"]),
                Sweep(
                    over=[0, 1],
                    steps=lambda layer: [
                        Logits(
                            murano_model,
                            fn=lambda activation, _node: activation + 1.0,
                            layers=[layer],
                            targets=None,
                        ),
                        LogitDiffStep(correct=5, incorrect=7),
                    ],
                    read=keys.LOGIT_DIFF,
                ),
            ]
        ).run()

        sweep = out[keys.SWEEP]
        assert sweep.swept == [0, 1]
        assert all(value == value for value in sweep.scores.values())
        # The edit lands in a different layer per item, so the scores must differ.
        assert sweep.scores[0] != sweep.scores[1]
        assert keys.FINAL_LOGITS not in out
