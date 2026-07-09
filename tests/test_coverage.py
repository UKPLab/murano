"""Coverage for previously-untested helpers, plots, and plot steps."""

from __future__ import annotations

import json
import logging
import math

import pytest
import torch


def _reject(token: str):
    raise ValueError(f"non-finite token {token!r} is not strict JSON")


# ── logging.setup_logging ──────────────────────────────────────────────


def test_setup_logging_adds_one_console_handler():
    from murano.logging import logger, setup_logging

    def console_handlers():
        return sum(
            getattr(h, "_murano_console_handler", False) for h in logger.handlers
        )

    setup_logging(logging.DEBUG)
    assert console_handlers() == 1
    setup_logging(logging.INFO)
    assert console_handlers() == 1  # not duplicated on repeat calls
    assert logger.level == logging.INFO


# ── strict-JSON round-trip for the logit-attribution writer ────────────


def test_logit_attribution_nan_round_trips(tmp_path):
    from murano.io import load_logit_attribution, save_logit_attribution
    from murano.steps.logit_attribution import LogitAttributionResult

    result = LogitAttributionResult(
        contributions={},
        embed_contribution=float("nan"),
        other_contribution=0.5,
        target="logit",
        total=float("nan"),
        completeness_error=float("nan"),
    )
    path = tmp_path / "logit_attribution.json"
    save_logit_attribution(result, path)

    # Strict JSON: a bare NaN would trip parse_constant.
    json.loads(path.read_text(), parse_constant=_reject)

    loaded = load_logit_attribution(path)
    assert math.isnan(loaded.embed_contribution)
    assert math.isnan(loaded.total)
    assert math.isnan(loaded.completeness_error)
    assert loaded.other_contribution == 0.5


# ── plotting smoke tests ──────────────────────────────────────────────


def test_plot_probe_accuracy_returns_figure():
    pytest.importorskip("plotly")
    import numpy as np

    from murano.nodes import Node
    from murano.plotting import plot_probe_accuracy
    from murano.steps.probe import ProbeResult

    probe = ProbeResult(
        accuracy_per_layer={Node(0, "residual"): 0.9, Node(1, "residual"): 0.8},
        cv_scores={
            Node(0, "residual"): np.array([0.85, 0.95]),
            Node(1, "residual"): np.array([0.75, 0.85]),
        },
        best_layer=Node(0, "residual"),
        label_names=["neg", "pos"],
    )
    fig = plot_probe_accuracy(probe)
    assert fig.data[0].type == "bar"
    assert list(fig.data[0].y) == [0.9, 0.8]


# ── Plot step ─────────────────────────────────────────────────────────


def test_plot_step_writes_files(tmp_path):
    pytest.importorskip("plotly")
    from murano import keys
    from murano.results import Results
    from murano.steps.plot import Plot
    from murano.steps.train import SteeringResult

    results = Results()
    results[keys.STEERING] = SteeringResult(
        direction_per_layer={
            (0, "residual"): torch.ones(4),
            (1, "residual"): torch.ones(4),
        },
        separation_scores={(0, "residual"): 1.0, (1, "residual"): 0.5},
        best_layer=(0, "residual"),
    )
    Plot(output_dir=str(tmp_path))(results)
    assert list((tmp_path / "plots").glob("separation_scores.*"))


def test_plot_step_probe_writes_files(tmp_path):
    pytest.importorskip("plotly")
    import numpy as np

    from murano import keys
    from murano.nodes import Node
    from murano.results import Results
    from murano.steps.plot import Plot
    from murano.steps.probe import ProbeResult

    results = Results()
    results[keys.PROBE] = ProbeResult(
        accuracy_per_layer={Node(0, "residual"): 0.9},
        cv_scores={Node(0, "residual"): np.array([0.85, 0.95])},
        best_layer=Node(0, "residual"),
        label_names=["neg", "pos"],
    )
    Plot(output_dir=str(tmp_path))(results)
    assert list((tmp_path / "plots").glob("probe_accuracy.*"))


def test_plot_step_logit_lens_writes_files(tmp_path):
    pytest.importorskip("plotly")
    from murano import keys
    from murano.nodes import Node
    from murano.results import Results
    from murano.steps.logit_lens import LogitLensResult
    from murano.steps.plot import Plot

    results = Results()
    results[keys.LOGIT_LENS] = LogitLensResult(
        all_probs=torch.zeros(2, 1, 2, 3),
        max_probs=torch.rand(2, 1, 2),
        predicted_tokens=torch.zeros(2, 1, 2, dtype=torch.long),
        predicted_words=[[["a", "b"]], [["c", "d"]]],
        input_words=[["x", "y"]],
        attention_mask=torch.ones(1, 2),
        addresses=[Node(0, "residual"), Node(1, "residual")],
    )
    Plot(output_dir=str(tmp_path))(results)
    # save_figure writes .png with kaleido, else falls back to .html.
    assert list((tmp_path / "plots").glob("logit_lens.*"))


def test_plot_step_logit_attribution_writes_files(tmp_path):
    pytest.importorskip("plotly")
    from murano import keys
    from murano.nodes import Node
    from murano.results import Results
    from murano.steps.logit_attribution import LogitAttributionResult
    from murano.steps.plot import Plot

    results = Results()
    results[keys.LOGIT_ATTRIBUTION] = LogitAttributionResult(
        contributions={Node(0, "mlp"): 0.3, Node(1, "mlp"): -0.2},
        embed_contribution=0.1,
        other_contribution=0.05,
        target="logit",
        total=0.25,
        completeness_error=0.0,
    )
    Plot(output_dir=str(tmp_path))(results)
    assert list((tmp_path / "plots").glob("logit_attribution.*"))


def test_plot_step_html_format_writes_html(tmp_path):
    pytest.importorskip("plotly")
    from murano import keys
    from murano.results import Results
    from murano.steps.plot import Plot
    from murano.steps.train import SteeringResult

    results = Results()
    results[keys.STEERING] = SteeringResult(
        direction_per_layer={
            (0, "residual"): torch.ones(4),
            (1, "residual"): torch.ones(4),
        },
        separation_scores={(0, "residual"): 1.0, (1, "residual"): 0.5},
        best_layer=(0, "residual"),
    )
    Plot(output_dir=str(tmp_path), save_format="html")(results)
    assert (tmp_path / "plots" / "separation_scores.html").exists()


def test_plot_step_defaults_to_shared_output_root(tmp_path, monkeypatch):
    pytest.importorskip("plotly")
    from murano import keys
    from murano.results import Results
    from murano.steps.plot import Plot
    from murano.steps.train import SteeringResult

    monkeypatch.chdir(tmp_path)
    results = Results()
    results[keys.STEERING] = SteeringResult(
        direction_per_layer={
            (0, "residual"): torch.ones(4),
            (1, "residual"): torch.ones(4),
        },
        separation_scores={(0, "residual"): 1.0, (1, "residual"): 0.5},
        best_layer=(0, "residual"),
    )
    # No output_dir and no results['output_dir']: writes under the shared
    # murano_outputs/ root, not a bare ./plots/ in the CWD.
    Plot(save_format="html")(results)
    plots = tmp_path / keys.DEFAULT_OUTPUT_DIR / "plots"
    assert (plots / "separation_scores.html").exists()
    assert not (tmp_path / "plots").exists()
