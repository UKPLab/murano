"""Tests for stateless Plotly visualization utilities.

Adapted from the legacy ``test_visualization_lenses.py`` blueprint.
Calls pure functions instead of Lens classes, and preserves all
``fig.to_dict()`` assertions for CI safety.
"""

import pytest
import torch

from murano.plotting.plotly_utils import plot_heatmap, plot_line_chart


# ===========================================================================
# plot_heatmap
# ===========================================================================


class TestPlotHeatmap:
    """Tests for plot_heatmap."""

    @pytest.fixture
    def heatmap_data(self):
        """Minimal heatmap data mimicking LogitComputationLens output."""
        return {
            "x_labels": ["The", "quick", "brown"],
            "z_data": [[0.1, 0.5, 0.2], [0.8, 0.9, 0.6]],
            "hover_data": [["a", "fox", "dog"], ["The", "fast", "bear"]],
        }

    # ------------------------------------------------------------------

    def test_returns_plotly_figure(self, heatmap_data):
        """plot_heatmap must return a plotly Figure."""
        import plotly.graph_objects as go

        fig = plot_heatmap(
            z_data=heatmap_data["z_data"],
            x_labels=heatmap_data["x_labels"],
            title="Logit Lens Probabilities",
            color_scale="Viridis",
            hover_data=heatmap_data["hover_data"],
        )
        assert isinstance(fig, go.Figure)

    def test_trace_type_is_heatmap(self, heatmap_data):
        fig = plot_heatmap(
            z_data=heatmap_data["z_data"],
            x_labels=heatmap_data["x_labels"],
            title="Logit Lens Probabilities",
            color_scale="Viridis",
        )
        fig_dict = fig.to_dict()
        assert fig_dict["data"][0]["type"] == "heatmap"

    def test_layout_title(self, heatmap_data):
        fig = plot_heatmap(
            z_data=heatmap_data["z_data"],
            x_labels=heatmap_data["x_labels"],
            title="Logit Lens Probabilities",
            color_scale="Viridis",
        )
        assert fig.to_dict()["layout"]["title"]["text"] == "Logit Lens Probabilities"

    def test_x_axis_labels(self, heatmap_data):
        """X labels must exactly match the provided list."""
        fig_dict = plot_heatmap(
            z_data=heatmap_data["z_data"],
            x_labels=heatmap_data["x_labels"],
            title="Logit Lens Probabilities",
            color_scale="Viridis",
        ).to_dict()
        assert fig_dict["data"][0]["x"] == ["The", "quick", "brown"]

    def test_z_values_match_input(self, heatmap_data):
        """Z values must match the input data (within floating-point tolerance)."""
        fig_dict = plot_heatmap(
            z_data=heatmap_data["z_data"],
            x_labels=heatmap_data["x_labels"],
            title="Logit Lens Probabilities",
            color_scale="Viridis",
        ).to_dict()
        expected = [[0.1, 0.5, 0.2], [0.8, 0.9, 0.6]]
        for row_i, row in enumerate(expected):
            for col_i, val in enumerate(row):
                assert abs(fig_dict["data"][0]["z"][row_i][col_i] - val) < 1e-5

    def test_custom_hover_data(self, heatmap_data):
        """customdata must equal the provided hover_data."""
        fig_dict = plot_heatmap(
            z_data=heatmap_data["z_data"],
            x_labels=heatmap_data["x_labels"],
            title="Logit Lens Probabilities",
            color_scale="Viridis",
            hover_data=heatmap_data["hover_data"],
        ).to_dict()
        assert fig_dict["data"][0]["customdata"] == heatmap_data["hover_data"]

    def test_colorscale_starts_with_viridis_hex(self, heatmap_data):
        """The first stop of the Viridis colorscale is the dark-purple hex #440154."""
        fig_dict = plot_heatmap(
            z_data=heatmap_data["z_data"],
            x_labels=heatmap_data["x_labels"],
            title="Logit Lens Probabilities",
            color_scale="Viridis",
        ).to_dict()
        first_color = fig_dict["data"][0]["colorscale"][0][1].lower()
        assert first_color == "#440154"

    def test_auto_y_labels_when_not_provided(self, heatmap_data):
        """When y_labels is not set, Y labels default to ['Layer 0', 'Layer 1', …]."""
        fig_dict = plot_heatmap(
            z_data=heatmap_data["z_data"],
            x_labels=heatmap_data["x_labels"],
            title="Auto Y",
        ).to_dict()
        assert fig_dict["data"][0]["y"] == ["Layer 0", "Layer 1"]

    def test_custom_y_labels(self, heatmap_data):
        """When y_labels is provided, they must appear on the Y axis."""
        y_labels = ["Row A", "Row B"]
        fig_dict = plot_heatmap(
            z_data=heatmap_data["z_data"],
            x_labels=heatmap_data["x_labels"],
            y_labels=y_labels,
            title="Custom Y",
        ).to_dict()
        assert fig_dict["data"][0]["y"] == y_labels

    def test_accepts_tensor_converted_to_list(self):
        """Function must work when z_data is a list converted from a tensor."""
        z_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        fig_dict = plot_heatmap(
            z_data=z_tensor.tolist(),
            title="From Tensor",
        ).to_dict()
        assert fig_dict["data"][0]["z"] == [[1.0, 2.0], [3.0, 4.0]]

    def test_empty_data_does_not_crash(self):
        """Empty z_data should not raise (Plotly handles it gracefully)."""
        fig = plot_heatmap(z_data=[], title="Empty")
        fig_dict = fig.to_dict()
        assert len(fig_dict["data"]) == 1


# ===========================================================================
# plot_line_chart
# ===========================================================================


class TestPlotLineChart:
    """Tests for plot_line_chart."""

    @pytest.fixture
    def line_data(self):
        return {
            "x": [0, 1, 2, 3, 4],
            "train_loss": [2.5, 2.0, 1.6, 1.3, 1.1],
            "val_loss": [2.6, 2.1, 1.8, 1.5, 1.4],
        }

    # ------------------------------------------------------------------

    def test_returns_plotly_figure(self, line_data):
        import plotly.graph_objects as go

        fig = plot_line_chart(
            x_data=line_data["x"],
            y_series={"train_loss": line_data["train_loss"]},
            title="Training Loss",
        )
        assert isinstance(fig, go.Figure)

    def test_single_trace_type_is_scatter(self, line_data):
        fig_dict = plot_line_chart(
            x_data=line_data["x"],
            y_series={"train_loss": line_data["train_loss"]},
            title="Training Loss",
        ).to_dict()
        assert fig_dict["data"][0]["type"] == "scatter"

    def test_single_trace_mode_is_lines(self, line_data):
        fig_dict = plot_line_chart(
            x_data=line_data["x"],
            y_series={"train_loss": line_data["train_loss"]},
            title="Training Loss",
        ).to_dict()
        assert fig_dict["data"][0]["mode"] == "lines"

    def test_x_values_match(self, line_data):
        fig_dict = plot_line_chart(
            x_data=line_data["x"],
            y_series={"train_loss": line_data["train_loss"]},
            title="Training Loss",
        ).to_dict()
        assert list(fig_dict["data"][0]["x"]) == [0, 1, 2, 3, 4]

    def test_y_values_match(self, line_data):
        fig_dict = plot_line_chart(
            x_data=line_data["x"],
            y_series={"train_loss": line_data["train_loss"]},
            title="Training Loss",
        ).to_dict()
        expected = [2.5, 2.0, 1.6, 1.3, 1.1]
        actual = list(fig_dict["data"][0]["y"])
        for e, a in zip(expected, actual):
            assert abs(e - a) < 1e-5

    def test_multi_line_trace_count(self, line_data):
        """Two y_series entries must produce two traces."""
        fig_dict = plot_line_chart(
            x_data=line_data["x"],
            y_series={
                "train_loss": line_data["train_loss"],
                "val_loss": line_data["val_loss"],
            },
            title="Loss Curves",
        ).to_dict()
        assert len(fig_dict["data"]) == 2

    def test_multi_line_all_traces_are_scatter(self, line_data):
        fig_dict = plot_line_chart(
            x_data=line_data["x"],
            y_series={
                "train_loss": line_data["train_loss"],
                "val_loss": line_data["val_loss"],
            },
            title="Loss Curves",
        ).to_dict()
        for trace in fig_dict["data"]:
            assert trace["type"] == "scatter"

    def test_multi_line_y_data_correct(self, line_data):
        """Each trace's y data must exactly match its source list."""
        fig_dict = plot_line_chart(
            x_data=line_data["x"],
            y_series={
                "train_loss": line_data["train_loss"],
                "val_loss": line_data["val_loss"],
            },
            title="Loss Curves",
        ).to_dict()
        train_expected = [2.5, 2.0, 1.6, 1.3, 1.1]
        val_expected = [2.6, 2.1, 1.8, 1.5, 1.4]
        for e, a in zip(train_expected, fig_dict["data"][0]["y"]):
            assert abs(e - a) < 1e-5
        for e, a in zip(val_expected, fig_dict["data"][1]["y"]):
            assert abs(e - a) < 1e-5

    def test_layout_title(self, line_data):
        fig = plot_line_chart(
            x_data=line_data["x"],
            y_series={"train_loss": line_data["train_loss"]},
            title="Training Loss",
        )
        assert fig.to_dict()["layout"]["title"]["text"] == "Training Loss"

    def test_axis_labels(self, line_data):
        fig_dict = plot_line_chart(
            x_data=line_data["x"],
            y_series={"train_loss": line_data["train_loss"]},
            title="T",
            x_label="Step",
            y_label="Loss",
        ).to_dict()
        assert fig_dict["layout"]["xaxis"]["title"]["text"] == "Step"
        assert fig_dict["layout"]["yaxis"]["title"]["text"] == "Loss"

    def test_accepts_tensor_converted_to_list(self):
        """Function must work when y values come from tensor.tolist()."""
        x = [1, 2, 3]
        y = torch.tensor([10.0, 20.0, 30.0]).tolist()
        fig_dict = plot_line_chart(
            x_data=x,
            y_series={"data": y},
            title="Tensor Input",
        ).to_dict()
        assert list(fig_dict["data"][0]["y"]) == [10.0, 20.0, 30.0]

    def test_single_point(self):
        """A single data point should not crash."""
        fig = plot_line_chart(
            x_data=[0],
            y_series={"point": [42.0]},
            title="Single Point",
        )
        fig_dict = fig.to_dict()
        assert list(fig_dict["data"][0]["x"]) == [0]
        assert list(fig_dict["data"][0]["y"]) == [42.0]

    def test_trace_names_in_legend(self, line_data):
        """Trace names must match the y_series keys."""
        fig_dict = plot_line_chart(
            x_data=line_data["x"],
            y_series={
                "train_loss": line_data["train_loss"],
                "val_loss": line_data["val_loss"],
            },
            title="Loss Curves",
        ).to_dict()
        assert fig_dict["data"][0]["name"] == "train_loss"
        assert fig_dict["data"][1]["name"] == "val_loss"
