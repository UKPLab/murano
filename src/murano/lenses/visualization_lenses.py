"""
Generic Visualization Lenses for the Murano interpretability pipeline.

These lenses are BaseVisualizationLenses: they consume an already-enriched
artifact and produce Plotly figures without mutating any tensors.

Do NOT call ``fig.show()`` in production or CI , the caller is responsible
for rendering or persisting the returned Figure object.
"""

from __future__ import annotations

import torch
import plotly.graph_objects as go
from typing import Any, Dict, List, Optional, Union

from .base_lens import BaseVisualizationLens


# ---------------------------------------------------------------------------
# HeatmapVisualizationLens
# ---------------------------------------------------------------------------


class HeatmapVisualizationLens(BaseVisualizationLens):
    """
    Renders a ``go.Heatmap`` from data stored in the artifact.

    Artifact inputs
    ---------------
    ``<x_key>``     : list[str] | 1-D tensor   -> X-axis labels.
    ``<z_key>``     : 2-D tensor | nested list  -> Heatmap color values.
    ``<y_key>``     : (optional) list[str]      -> Y-axis labels; auto-generated
                      as ``['Layer 0', 'Layer 1', …]`` when omitted.
    ``<hover_key>`` : (optional) nested list    -> Custom hover data per cell.

    Parameters
    ----------
    name        : str  -> Lens identifier used in logging.
    x_key       : str  -> Artifact key for the X-axis labels.
    y_key       : str | None -> Artifact key for the Y-axis labels.
    z_key       : str  -> Artifact key for the heatmap values (Z / color).
    hover_key   : str | None -> Artifact key for custom hover text.
    title       : str  -> Figure title.
    color_scale : str  -> Plotly colorscale name (e.g. ``'Viridis'``).
    """

    def __init__(
        self,
        name: str = "HeatmapVisualization",
        x_key: str = "x_labels",
        y_key: Optional[str] = None,
        z_key: str = "z_values",
        hover_key: Optional[str] = None,
        title: str = "Heatmap",
        color_scale: str = "Viridis",
    ) -> None:
        super().__init__(name=name)
        self.x_key = x_key
        self.y_key = y_key
        self.z_key = z_key
        self.hover_key = hover_key
        self.title = title
        self.color_scale = color_scale

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_python(value: Any) -> Any:
        """Convert a PyTorch tensor to a nested Python list; pass-through otherwise."""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        return value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def visualize(self, artifact: Dict[str, Any]) -> go.Figure:
        """
        Build and return a Plotly Heatmap Figure from the artifact.

        Args:
            artifact: Enriched pipeline artifact. Must contain ``z_key``
                      and optionally ``x_key``, ``y_key``, ``hover_key``.

        Returns:
            A ``plotly.graph_objects.Figure`` containing a single Heatmap trace.

        Raises:
            ValueError: If ``z_key`` is absent from the artifact.
        """
        # ---- Z data (required) ----
        z_raw = artifact.get(self.z_key)
        if z_raw is None:
            raise ValueError(f"Artifact is missing the required z_key: '{self.z_key}'")
        z_data = self._to_python(z_raw)

        # ---- X labels ----
        x_data = self._to_python(artifact.get(self.x_key, []))

        # ---- Y labels ----
        if self.y_key and self.y_key in artifact:
            y_data = self._to_python(artifact[self.y_key])
        else:
            y_data = [f"Layer {i}" for i in range(len(z_data))]

        # ---- Custom hover data ----
        custom_data = None
        hover_template = (
            "<b>X:</b> %{x}<br><b>Y:</b> %{y}<br><b>Value:</b> %{z:.4f}<extra></extra>"
        )
        if self.hover_key and self.hover_key in artifact:
            custom_data = artifact[self.hover_key]
            hover_template = (
                "<b>Input:</b> %{x}<br>"
                "<b>Layer:</b> %{y}<br>"
                "<b>Prob:</b> %{z:.4f}<br>"
                "<b>Pred:</b> %{customdata}"
                "<extra></extra>"
            )

        # ---- Build figure ----
        fig = go.Figure(
            data=go.Heatmap(
                z=z_data,
                x=x_data,
                y=y_data,
                colorscale=self.color_scale,
                customdata=custom_data,
                hovertemplate=hover_template,
            )
        )
        fig.update_layout(
            title=self.title,
            xaxis_title=self.x_key.replace("_", " ").title(),
            yaxis_title=(
                "Layers" if not self.y_key else self.y_key.replace("_", " ").title()
            ),
        )
        return fig


# ---------------------------------------------------------------------------
# LineChartVisualizationLens
# ---------------------------------------------------------------------------


class LineChartVisualizationLens(BaseVisualizationLens):
    """
    Renders one or more line traces from 1-D data stored in the artifact.

    Artifact inputs
    ---------------
    ``<x_key>``          : list | 1-D tensor -> Shared X-axis values.
    ``<y_keys[i]>``      : list | 1-D tensor -> Y values for each line trace.

    Parameters
    ----------
    name   : str  -> Lens identifier.
    x_key  : str  -> Artifact key for the X-axis values.
    y_keys : list[str]  -> One or more artifact keys, each producing a line.
    title  : str  -> Figure title.
    """

    def __init__(
        self,
        name: str = "LineChartVisualization",
        x_key: str = "x_values",
        y_keys: Optional[Union[str, List[str]]] = None,
        title: str = "Line Chart",
    ) -> None:
        super().__init__(name=name)
        self.x_key = x_key
        # Accept a single string for convenience
        if y_keys is None:
            self.y_keys: List[str] = []
        elif isinstance(y_keys, str):
            self.y_keys = [y_keys]
        else:
            self.y_keys = list(y_keys)
        self.title = title

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_list(value: Any) -> list:
        """Flatten a 1-D tensor or pass-through a plain list."""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        return list(value)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def visualize(self, artifact: Dict[str, Any]) -> go.Figure:
        """
        Build and return a Plotly Figure containing one ``go.Scatter`` trace
        per key in ``y_keys``.

        Args:
            artifact: Enriched pipeline artifact. Must contain ``x_key``
                      and all keys specified in ``y_keys``.

        Returns:
            A ``plotly.graph_objects.Figure`` with line trace(s).

        Raises:
            ValueError: If ``x_key`` or any entry in ``y_keys`` is missing.
        """
        if self.x_key not in artifact:
            raise ValueError(f"Artifact is missing the required x_key: '{self.x_key}'")
        x_data = self._to_list(artifact[self.x_key])

        traces: List[go.Scatter] = []
        for key in self.y_keys:
            if key not in artifact:
                raise ValueError(f"Artifact is missing the y_key: '{key}'")
            y_data = self._to_list(artifact[key])
            traces.append(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode="lines",
                    name=key.replace("_", " ").title(),
                )
            )

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=self.title,
            xaxis_title=self.x_key.replace("_", " ").title(),
        )
        return fig
