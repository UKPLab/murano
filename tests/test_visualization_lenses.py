"""
TDD test suite for visualization lenses.

Tests are intentionally self-contained: no mocking of Plotly and no
calls to ``fig.show()`` so the suite runs cleanly in CI/CD.

"""

import pytest
import torch

from murano.lenses.visualization_lenses import (
    HeatmapVisualizationLens,
    LineChartVisualizationLens,
)


# ===========================================================================
# HeatmapVisualizationLens
# ===========================================================================


class TestHeatmapVisualizationLens:
    """Tests for HeatmapVisualizationLens."""

    @pytest.fixture
    def heatmap_artifact(self):
        """A minimal artifact that mimics the output of a LogitComputationLens."""
        return {
            "input_words": ["The", "quick", "brown"],
            # 2 layers × 3 tokens
            "max_probs": torch.tensor([[0.1, 0.5, 0.2], [0.8, 0.9, 0.6]]),
            "predicted_words": [["a", "fox", "dog"], ["The", "fast", "bear"]],
        }

    @pytest.fixture
    def lens(self):
        return HeatmapVisualizationLens(
            name="LogitHeatmap",
            x_key="input_words",
            z_key="max_probs",
            hover_key="predicted_words",
            title="Logit Lens Probabilities",
            color_scale="Viridis",
        )

    # ------------------------------------------------------------------

    def test_returns_plotly_figure(self, lens, heatmap_artifact):
        """visualize() must return a plotly Figure."""
        import plotly.graph_objects as go

        fig = lens.visualize(heatmap_artifact)
        assert isinstance(fig, go.Figure)

    def test_trace_type_is_heatmap(self, lens, heatmap_artifact):
        fig = lens.visualize(heatmap_artifact)
        fig_dict = fig.to_dict()
        assert fig_dict["data"][0]["type"] == "heatmap"

    def test_layout_title(self, lens, heatmap_artifact):
        fig = lens.visualize(heatmap_artifact)
        assert fig.to_dict()["layout"]["title"]["text"] == "Logit Lens Probabilities"

    def test_x_axis_labels(self, lens, heatmap_artifact):
        """X labels must exactly match the list stored under x_key."""
        fig_dict = lens.visualize(heatmap_artifact).to_dict()
        assert fig_dict["data"][0]["x"] == ["The", "quick", "brown"]

    def test_z_values_match_tensor(self, lens, heatmap_artifact):
        """Z values must match the original tensor (within floating-point tolerance)."""
        fig_dict = lens.visualize(heatmap_artifact).to_dict()
        expected = [[0.1, 0.5, 0.2], [0.8, 0.9, 0.6]]
        for row_i, row in enumerate(expected):
            for col_i, val in enumerate(row):
                assert abs(fig_dict["data"][0]["z"][row_i][col_i] - val) < 1e-5

    def test_custom_hover_data(self, lens, heatmap_artifact):
        """customdata must equal the nested list stored under hover_key."""
        fig_dict = lens.visualize(heatmap_artifact).to_dict()
        assert fig_dict["data"][0]["customdata"] == heatmap_artifact["predicted_words"]

    def test_colorscale_starts_with_viridis_hex(self, lens, heatmap_artifact):
        """The first stop of the Viridis colorscale is the dark-purple hex #440154."""
        fig_dict = lens.visualize(heatmap_artifact).to_dict()
        first_color = fig_dict["data"][0]["colorscale"][0][1].lower()
        assert first_color == "#440154"

    def test_auto_y_labels_when_y_key_absent(self, heatmap_artifact):
        """When y_key is not set, Y labels default to ['Layer 0', 'Layer 1', …]."""
        lens = HeatmapVisualizationLens(
            x_key="input_words",
            z_key="max_probs",
            title="Auto Y",
        )
        fig_dict = lens.visualize(heatmap_artifact).to_dict()
        assert fig_dict["data"][0]["y"] == ["Layer 0", "Layer 1"]

    def test_missing_z_key_raises_value_error(self, heatmap_artifact):
        """Absent z_key must raise a descriptive ValueError."""
        lens = HeatmapVisualizationLens(z_key="nonexistent_key")
        with pytest.raises(ValueError, match="nonexistent_key"):
            lens.visualize(heatmap_artifact)

    def test_artifact_not_mutated(self, lens, heatmap_artifact):
        """visualize() must not modify the artifact."""
        import copy

        original = copy.deepcopy(heatmap_artifact)
        lens.visualize(heatmap_artifact)
        assert heatmap_artifact.keys() == original.keys()
        for k in original:
            if isinstance(original[k], torch.Tensor):
                assert torch.equal(heatmap_artifact[k], original[k])
            else:
                assert heatmap_artifact[k] == original[k]


# ===========================================================================
# LineChartVisualizationLens
# ===========================================================================


class TestLineChartVisualizationLens:
    """Tests for LineChartVisualizationLens."""

    @pytest.fixture
    def line_artifact(self):
        return {
            "steps": [0, 1, 2, 3, 4],
            "train_loss": torch.tensor([2.5, 2.0, 1.6, 1.3, 1.1]),
            "val_loss": torch.tensor([2.6, 2.1, 1.8, 1.5, 1.4]),
        }

    @pytest.fixture
    def single_line_lens(self):
        return LineChartVisualizationLens(
            name="LossChart",
            x_key="steps",
            y_keys=["train_loss"],
            title="Training Loss",
        )

    @pytest.fixture
    def multi_line_lens(self):
        return LineChartVisualizationLens(
            x_key="steps",
            y_keys=["train_loss", "val_loss"],
            title="Loss Curves",
        )

    # ------------------------------------------------------------------

    def test_returns_plotly_figure(self, single_line_lens, line_artifact):
        import plotly.graph_objects as go

        fig = single_line_lens.visualize(line_artifact)
        assert isinstance(fig, go.Figure)

    def test_single_trace_type_is_scatter(self, single_line_lens, line_artifact):
        fig_dict = single_line_lens.visualize(line_artifact).to_dict()
        assert fig_dict["data"][0]["type"] == "scatter"

    def test_single_trace_mode_is_lines(self, single_line_lens, line_artifact):
        fig_dict = single_line_lens.visualize(line_artifact).to_dict()
        assert fig_dict["data"][0]["mode"] == "lines"

    def test_x_values_match(self, single_line_lens, line_artifact):
        fig_dict = single_line_lens.visualize(line_artifact).to_dict()
        assert list(fig_dict["data"][0]["x"]) == [0, 1, 2, 3, 4]

    def test_y_values_match_tensor(self, single_line_lens, line_artifact):
        fig_dict = single_line_lens.visualize(line_artifact).to_dict()
        expected = [2.5, 2.0, 1.6, 1.3, 1.1]
        actual = list(fig_dict["data"][0]["y"])
        for e, a in zip(expected, actual):
            assert abs(e - a) < 1e-5

    def test_multi_line_trace_count(self, multi_line_lens, line_artifact):
        """Two y_keys must produce two traces."""
        fig_dict = multi_line_lens.visualize(line_artifact).to_dict()
        assert len(fig_dict["data"]) == 2

    def test_multi_line_all_traces_are_scatter(self, multi_line_lens, line_artifact):
        fig_dict = multi_line_lens.visualize(line_artifact).to_dict()
        for trace in fig_dict["data"]:
            assert trace["type"] == "scatter"

    def test_multi_line_y_data_correct(self, multi_line_lens, line_artifact):
        """Each trace's y data must exactly match its source tensor."""
        fig_dict = multi_line_lens.visualize(line_artifact).to_dict()
        train_expected = [2.5, 2.0, 1.6, 1.3, 1.1]
        val_expected = [2.6, 2.1, 1.8, 1.5, 1.4]
        for e, a in zip(train_expected, fig_dict["data"][0]["y"]):
            assert abs(e - a) < 1e-5
        for e, a in zip(val_expected, fig_dict["data"][1]["y"]):
            assert abs(e - a) < 1e-5

    def test_layout_title(self, single_line_lens, line_artifact):
        fig = single_line_lens.visualize(line_artifact)
        assert fig.to_dict()["layout"]["title"]["text"] == "Training Loss"

    def test_missing_x_key_raises(self, line_artifact):
        lens = LineChartVisualizationLens(x_key="missing_x", y_keys=["train_loss"])
        with pytest.raises(ValueError, match="missing_x"):
            lens.visualize(line_artifact)

    def test_missing_y_key_raises(self, line_artifact):
        lens = LineChartVisualizationLens(x_key="steps", y_keys=["nonexistent"])
        with pytest.raises(ValueError, match="nonexistent"):
            lens.visualize(line_artifact)

    def test_accepts_plain_python_lists(self):
        """Lens must work when artifact values are plain Python lists (not tensors)."""
        artifact = {
            "x": [1, 2, 3],
            "y": [10, 20, 30],
        }
        lens = LineChartVisualizationLens(x_key="x", y_keys=["y"], title="T")
        fig_dict = lens.visualize(artifact).to_dict()
        assert list(fig_dict["data"][0]["x"]) == [1, 2, 3]
        assert list(fig_dict["data"][0]["y"]) == [10, 20, 30]

    def test_single_string_y_key_convenience(self, line_artifact):
        """Passing a single string for y_keys must be equivalent to [string]."""
        lens = LineChartVisualizationLens(x_key="steps", y_keys="train_loss", title="T")
        fig_dict = lens.visualize(line_artifact).to_dict()
        assert len(fig_dict["data"]) == 1
