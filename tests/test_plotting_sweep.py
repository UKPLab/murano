"""Tests for plot_sweep, which picks its shape from the swept items."""

from __future__ import annotations

import math

from murano import Node
from murano.artifacts import SweepResult
from murano.plotting import plot_sweep
from murano.plotting.plotly_utils import BAR_COLOR, HIGHLIGHT_COLOR


def _heads(
    values: dict[tuple[int, int], float], name: str = "recovered"
) -> SweepResult:
    scores = {
        Node(layer, "self_attn", head=head): v for (layer, head), v in values.items()
    }
    return SweepResult(columns={name: scores}, primary=name)


def _plain(values: dict[str, float], name: str = "logit_diff") -> SweepResult:
    return SweepResult(columns={name: values}, primary=name)


class TestHeadSweep:
    def test_renders_a_heatmap(self):
        fig = plot_sweep(_heads({(0, 0): 1.0, (0, 1): 2.0}))
        assert fig.data[0].type == "heatmap"

    def test_signed_scores_center_the_color_scale_on_zero(self):
        """A signed statistic on a sequential scale shades zero as if it had a sign."""
        fig = plot_sweep(_heads({(0, 0): -1.0, (0, 1): 2.0}))
        assert fig.data[0].zmid == 0.0

    def test_one_sided_scores_stay_sequential(self):
        fig = plot_sweep(_heads({(0, 0): 1.0, (0, 1): 2.0}))
        assert fig.data[0].zmid is None

    def test_diverging_can_be_forced(self):
        fig = plot_sweep(_heads({(0, 0): 1.0, (0, 1): 2.0}), diverging=True)
        assert fig.data[0].zmid == 0.0

    def test_unswept_cells_are_blank_not_zero(self):
        fig = plot_sweep(_heads({(2, 1): 0.7}))
        z = fig.data[0].z
        assert z[2][1] == 0.7
        assert math.isnan(z[0][0])

    def test_title_defaults_to_the_column_name(self):
        assert plot_sweep(_heads({(0, 0): 1.0})).layout.title.text == "recovered"


class TestPlainSweep:
    def test_renders_one_bar_per_item(self):
        fig = plot_sweep(_plain({"zero": 1.0, "mean": 2.0}))
        assert fig.data[0].type == "bar"
        assert list(fig.data[0].x) == ["zero", "mean"]

    def test_signed_scores_color_by_sign(self):
        fig = plot_sweep(_plain({"zero": -0.3, "mean": 0.4}))
        assert list(fig.data[0].marker.color) == [HIGHLIGHT_COLOR, BAR_COLOR]

    def test_one_sided_scores_use_one_color(self):
        fig = plot_sweep(_plain({"zero": 0.3, "mean": 0.4}))
        assert fig.data[0].marker.color == BAR_COLOR

    def test_a_node_sweep_that_is_not_over_heads_falls_back_to_bars(self):
        scores = {Node(0, "mlp"): 1.0, Node(1, "mlp"): 2.0}
        fig = plot_sweep(SweepResult(columns={"x": scores}, primary="x"))
        assert fig.data[0].type == "bar"


class TestColumnSelection:
    def test_renders_a_named_column(self):
        result = SweepResult(
            columns={"logit_diff": {"a": 1.0}, "kl_div": {"a": 5.0}},
            primary="logit_diff",
        )
        fig = plot_sweep(result, column="kl_div")
        assert list(fig.data[0].y) == [5.0]
        assert fig.layout.title.text == "kl_div"
