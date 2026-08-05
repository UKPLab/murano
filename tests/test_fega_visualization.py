"""Vital numerical and selection checks for FEGA visualization utilities."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from murano.fega.visualization import (
    ATLAS_VECTOR_KEYS,
    atlas_coordinates,
    project_directions,
    project_kernel,
    rank_family_candidates,
    surface_coordinates,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "fega"


@dataclass(frozen=True)
class _Candidate:
    """Small candidate record used to exercise source ranking."""

    feature_id: int
    primary_label: str
    n_valid: float | None
    m_median: float | None
    r_span_pr: float = 0.0
    c_ray: float = 0.0
    r_ctr_pr: float = 0.0


def test_projection_surface_and_family_ranking_are_deterministic() -> None:
    """Preserve FEGA geometry and source tie-breaking in one compact fixture."""

    # Project a small PSD geometry and verify sign-fixed, unit-surface output.
    directions = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 0.0]])
    projection = project_directions(directions, np.eye(2), dimensions=3)
    anchors = np.argmax(np.abs(projection.coordinates), axis=0)
    assert np.all(projection.coordinates[anchors, np.arange(3)] >= 0.0)
    surface, keep = surface_coordinates(
        np.vstack([projection.coordinates, np.zeros(3)])
    )
    np.testing.assert_allclose(np.linalg.norm(surface, axis=1), 1.0)
    assert surface.shape[0] == 3
    assert keep.tolist() == [True, True, True, False]

    # Rank normal and unresolved families by their distinct source tuples.
    normal = [
        _Candidate(3, "normal", 4, 2.0),
        _Candidate(1, "normal", 4, 2.0),
        _Candidate(2, "normal", 5, 1.0),
    ]
    unresolved = [
        _Candidate(
            4,
            "unresolved_high_dimensional_or_diffuse",
            3,
            0.0,
            r_span_pr=2.0,
            c_ray=0.4,
            r_ctr_pr=1.0,
        ),
        _Candidate(
            5,
            "unresolved_high_dimensional_or_diffuse",
            3,
            0.0,
            r_span_pr=2.0,
            c_ray=0.2,
            r_ctr_pr=0.5,
        ),
        _Candidate(
            6,
            "unresolved_high_dimensional_or_diffuse",
            4,
            0.0,
            r_span_pr=1.0,
            c_ray=0.8,
            r_ctr_pr=0.1,
        ),
    ]
    ranked = rank_family_candidates([*normal, *unresolved], top_k=2)
    assert [candidate.feature_id for candidate in ranked["normal"]] == [2, 1]
    assert [
        candidate.feature_id
        for candidate in ranked["unresolved_high_dimensional_or_diffuse"]
    ] == [6, 5]

    # Exercise finite zeroes against both None and NaN without widening fixtures.
    missing_normal = [
        _Candidate(1, "normal", 0, 1.0),
        _Candidate(2, "normal", None, 100.0),
        _Candidate(3, "normal", np.nan, 200.0),
        _Candidate(4, "normal", 1, None),
        _Candidate(5, "normal", 1, np.nan),
        _Candidate(6, "normal", 1, 2.0),
    ]
    missing_unresolved = [
        _Candidate(10, "unresolved_high_dimensional_or_diffuse", 0, 0.0, 1.0),
        _Candidate(11, "unresolved_high_dimensional_or_diffuse", None, 0.0, 100.0),
        _Candidate(12, "unresolved_high_dimensional_or_diffuse", np.nan, 0.0, 200.0),
    ]
    missing_ranked = rank_family_candidates(
        [*missing_normal, *missing_unresolved], top_k=10
    )
    assert [candidate.feature_id for candidate in missing_ranked["normal"]] == [
        6,
        4,
        5,
        1,
        3,
        2,
    ]
    assert [
        candidate.feature_id
        for candidate in missing_ranked["unresolved_high_dimensional_or_diffuse"]
    ] == [10, 12, 11]


def test_cached_real_kernel_reproduces_source_coordinates() -> None:
    """Preserve source eigensolver ordering/signs on one real FEGA feature."""
    # Reproject the compact source kernel without loading a model or full Gram.
    with np.load(FIXTURE_DIR / "real_feature_33760.npz") as fixture:
        projected = project_kernel(fixture["kernel"], dimensions=3)
        np.testing.assert_allclose(
            projected.coordinates, fixture["sphere_coordinates"], atol=1e-10
        )


def test_atlas_coordinates_match_source_robust_scaler_pca() -> None:
    """Preserve source preprocessing for missing and near-constant fields."""
    # Freeze one source-shaped case where sklearn treats a tiny IQR as constant.
    records: list[dict[str, float | None]] = [
        {
            key: float((row + 1) * (column + 1))
            for column, key in enumerate(ATLAS_VECTOR_KEYS)
        }
        for row in range(5)
    ]
    for row, value in enumerate(1.0 + np.arange(5) * 1.0e-15):
        records[row]["r2"] = float(value)
    records[1]["c_ray"] = None
    records[3]["c_ray"] = np.nan
    expected = np.array(
        [
            [-5.732887546931234, -0.3660606155280897],
            [-1.5681259792520406, 1.3382753502903768],
            [0.0, 0.0],
            [1.5681259792520408, -1.3382753502903786],
            [5.732887546931233, 0.36606061552808944],
        ]
    )
    np.testing.assert_allclose(atlas_coordinates(records), expected, atol=1.0e-12)
