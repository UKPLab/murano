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


# ── plotting smoke tests (matplotlib/seaborn) ──────────────────────────


def test_plot_probe_accuracy_writes_png(tmp_path):
    pytest.importorskip("seaborn")
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
    out = tmp_path / "probe.png"
    plot_probe_accuracy(probe, save_path=out)
    assert out.exists()


# ── Plot / ProbePlot steps ─────────────────────────────────────────────


def test_plot_step_writes_files(tmp_path):
    pytest.importorskip("seaborn")
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
    assert (tmp_path / "plots" / "separation_scores.png").exists()


def test_probe_plot_step_writes_files(tmp_path):
    pytest.importorskip("seaborn")
    import numpy as np

    from murano import keys
    from murano.nodes import Node
    from murano.results import Results
    from murano.steps.probe import ProbeResult
    from murano.steps.probing.plot import ProbePlot

    results = Results()
    results[keys.PROBE] = ProbeResult(
        accuracy_per_layer={Node(0, "residual"): 0.9},
        cv_scores={Node(0, "residual"): np.array([0.85, 0.95])},
        best_layer=Node(0, "residual"),
        label_names=["neg", "pos"],
    )
    ProbePlot(output_dir=str(tmp_path))(results)
    assert (tmp_path / "plots" / "probe_accuracy.png").exists()
