"""Extract compact FEGA references from the completed source runs.

This is a regeneration tool, not a Murano runtime dependency. It reads the
completed comparison cache named in ``tasks/.executing/fega.md`` and writes
only derived arrays, metrics, and reference images used by local checks and
the reproduction notebook's reference-comparison section.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch


PRIMARY_RUN = (
    "results/fega/ravel/"
    "saebench_gemma-2-2b_width-2pow16_date-0107_"
    "gemma-2-2b_standard_new_width-2pow16_date-0107_"
    "resid_post_layer_12_trainer_2_custom_sae_eval_results.json/city_Country"
)
MULTI_MODE_RUN = (
    "results/fega/ravel/"
    "saebench_gemma-2-2b_width-2pow16_date-0107_"
    "gemma-2-2b_matryoshka_batch_top_k_width-2pow16_date-0107_"
    "resid_post_layer_12_trainer_2_custom_sae_eval_results.json/city_Country"
)
MULTI_MODE_FAMILY = "multi_mode_directional_geometry"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BUNDLE_DIR = REPOSITORY_ROOT / "notebooks/reproductions/artifacts/hoang2026_fega"
FIXTURE_DIR = REPOSITORY_ROOT / "tests/fixtures/fega"
PRIMARY_FAMILIES = {
    "axis_or_antipodal",
    "directed_ray",
    "global_2D_directional_subspace",
    "global_kD_directional_subspace",
    "oneD_diffuse",
    "residual_lowD_k",
    "unresolved_high_dimensional_or_diffuse",
}


def _load_json(path: Path) -> dict[str, Any]:
    """Load a trusted source JSON object from ``path``.

    Args:
        path: Existing JSON artifact path.

    Returns:
        The decoded top-level object.
    """
    # Decode the source artifact without inventing a parallel schema layer.
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    """Write one readable JSON artifact with stable key ordering.

    Args:
        path: Destination path below a generated output directory.
        payload: JSON-serializable derived reference data.
    """
    # Create the exact parent and write human-inspectable deterministic JSON.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _replace_generated_dir(path: Path) -> None:
    """Replace one script-owned generated directory.

    Args:
        path: Exact generated directory to replace.
    """
    # Constrain cleanup to the two explicitly owned FEGA output directories.
    path = path.resolve()
    if path not in {BUNDLE_DIR, FIXTURE_DIR}:
        raise ValueError(f"Refusing to replace unexpected output path: {path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _project(
    directions: torch.Tensor, gram: torch.Tensor, dimensions: int, *, centered: bool
) -> dict[str, torch.Tensor | int]:
    """Reproduce source FEGA's sign-fixed spectral projection.

    Args:
        directions: Ordered unit residual directions, shaped ``[rows, width]``.
        gram: Logit-equivalent residual Gram matrix, shaped ``[width, width]``.
        dimensions: Number of coordinate columns to retain.
        centered: Whether to double-center the exact kernel before decomposition.

    Returns:
        Kernel, coordinates, eigenvalues, explained ratios, and numerical rank.
    """
    # Form the exact symmetric logit-equivalent kernel in float64.
    rows = directions.detach().cpu().to(torch.float64)
    gram64 = gram.detach().cpu().to(torch.float64)
    kernel = rows @ gram64 @ rows.T
    kernel = (kernel + kernel.T) / 2.0
    if centered:
        row_mean = kernel.mean(dim=1, keepdim=True)
        kernel = kernel - row_mean - row_mean.T + kernel.mean()
        kernel = (kernel + kernel.T) / 2.0

    # Sort the spectrum, clamp numerical roundoff, and build principal coordinates.
    eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = torch.clamp(eigenvalues[order], min=0.0)
    eigenvectors = eigenvectors[:, order]
    take = min(dimensions, int(eigenvalues.numel()))
    coordinates = eigenvectors[:, :take] * torch.sqrt(eigenvalues[:take])
    for column_index in range(take):
        column = coordinates[:, column_index]
        anchor = int(torch.argmax(torch.abs(column)).item())
        if float(column[anchor]) < 0.0:
            coordinates[:, column_index] = -column
    if take < dimensions:
        coordinates = torch.cat(
            [coordinates, torch.zeros((len(rows), dimensions - take))], dim=1
        )

    # Match the source trace ratios and numerical-rank policy.
    total = float(eigenvalues.sum())
    ratios = eigenvalues / total if total > 0.0 else torch.zeros_like(eigenvalues)
    largest = float(eigenvalues[0]) if eigenvalues.numel() else 0.0
    rank = int((eigenvalues > max(1.0e-12, largest * 1.0e-12)).sum())
    return {
        "kernel": kernel,
        "coordinates": coordinates,
        "eigenvalues": eigenvalues,
        "explained_ratios": ratios,
        "numerical_rank": rank,
    }


def _display_coordinates(
    family: str,
    sphere_coordinates: torch.Tensor,
    directions: torch.Tensor,
    gram: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return source-equivalent sphere and 2D display coordinates.

    Args:
        family: Reported FEGA geometry family.
        sphere_coordinates: Sign-fixed uncentered three-dimensional coordinates.
        directions: Ordered unit residual directions.
        gram: Exact residual Gram matrix.

    Returns:
        Oriented sphere coordinates and family-specific 2D coordinates.
    """
    # Canonicalize the unsigned-axis display before deriving the 2D view.
    sphere = sphere_coordinates.clone()
    if family == "axis_or_antipodal":
        first = sphere[:, 0]
        if int((first > 0.0).sum()) > int((first < 0.0).sum()):
            sphere[:, 0] = -first
    # Use the source family's faithful or centered 2D view before display transforms.
    if family == "residual_lowD_k":
        plane = _project(directions, gram, 4, centered=True)["coordinates"]
        assert isinstance(plane, torch.Tensor)
    else:
        plane = sphere[:, :2]
    if family == "directed_ray":
        centered_plane = plane - plane.mean(dim=0)
        _, _, right_vectors = torch.linalg.svd(centered_plane, full_matrices=False)
        plane = centered_plane @ right_vectors.T
        for column_index in range(plane.shape[1]):
            column = plane[:, column_index]
            anchor = int(torch.argmax(torch.abs(column)).item())
            if float(column[anchor]) < 0.0:
                plane[:, column_index] = -column
    elif family == "unresolved_high_dimensional_or_diffuse":
        plane = plane - plane.mean(dim=0)
    return sphere, plane


def _feature_block(payload: dict[str, Any], feature_id: int) -> dict[str, Any]:
    """Slice one ordered feature block from a source effect shard.

    Args:
        payload: Decoded source effect-shard mapping.
        feature_id: Selected SAE feature identifier.

    Returns:
        Tensor and aligned-metadata slices for the requested feature.
    """
    # Resolve the source row interval from its compact feature-offset representation.
    feature_ids = payload["feature_ids"].tolist()
    feature_index = feature_ids.index(feature_id)
    start = int(payload["row_offsets"][feature_index])
    stop = int(payload["row_offsets"][feature_index + 1])

    # Slice only the arrays retained by the compact visual and CPU references.
    block: dict[str, Any] = {}
    for name in (
        "magnitude",
        "direction",
    ):
        block[name] = payload[name][start:stop]
    return block


def _curate_candidates(
    run_dir: Path,
    bundle_dir: Path,
    *,
    only_family: str | None,
    top_k: int,
    source_tag: str,
) -> list[dict[str, Any]]:
    """Extract selected candidate geometry and source images from one run.

    Args:
        run_dir: Completed source FEGA task directory.
        bundle_dir: Compact reproduction bundle root.
        only_family: Optional single-family supplement selector.
        top_k: Maximum available candidates retained per family.
        source_tag: Stable label distinguishing primary and supplement runs.

    Returns:
        Candidate summaries for the bundle index.
    """
    # Select only source-renderable candidates in the cached deterministic order.
    index = _load_json(run_dir / "visualizations" / "candidates.json")
    selected: dict[int, dict[str, Any]] = {}
    families = index["families"]
    for family, rows in families.items():
        if only_family is None and family not in PRIMARY_FAMILIES:
            continue
        if only_family is not None and family != only_family:
            continue
        available = [
            row for row in rows if row.get("visualization_status") == "available"
        ]
        for row in available[:top_k]:
            selected[int(row["feature_id"])] = row

    # Load compact summaries and the one Gram matrix needed for exact projections.
    manifest = _load_json(
        run_dir / "compute_effect" / "final_resid" / "effect_tensors_manifest.json"
    )
    gram = torch.load(
        run_dir / "data_prep" / "gram_cache" / "gram.pt",
        map_location="cpu",
        weights_only=True,
    )
    vmf = _load_json(run_dir / "vmf" / "pre_softcap_logits" / "vmf_scores.json")
    assignments_by_feature = {
        int(row["feature_id"]): (
            row["selected_fit"].get("hard_assignments", [])
            if isinstance(row.get("selected_fit"), dict)
            else []
        )
        for row in vmf["features"]
    }
    summaries: list[dict[str, Any]] = []

    # Read each relevant source shard once and immediately emit its compact feature blocks.
    for shard in manifest["shards"]:
        shard_ids = selected.keys() & {int(value) for value in shard["feature_ids"]}
        if not shard_ids:
            continue
        shard_path = (
            run_dir / "compute_effect" / "final_resid" / Path(shard["path"]).name
        )
        payload = torch.load(shard_path, map_location="cpu", weights_only=True)
        for feature_id in sorted(shard_ids):
            candidate = selected[feature_id]
            family = str(candidate["primary_label"])
            rank = int(candidate["rank"])
            block = _feature_block(payload, feature_id)
            sphere_projection = _project(block["direction"], gram, 3, centered=False)
            raw_sphere = sphere_projection["coordinates"]
            assert isinstance(raw_sphere, torch.Tensor)
            sphere, plane = _display_coordinates(
                family, raw_sphere, block["direction"], gram
            )

            # Store numeric geometry separately from inspectable scientific metadata.
            relative_dir = (
                Path("candidates") / family / f"rank_{rank:02d}_f{feature_id}"
            )
            candidate_dir = bundle_dir / relative_dir
            candidate_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                candidate_dir / "geometry.npz",
                kernel=np.asarray(sphere_projection["kernel"]),
                sphere_coordinates=np.asarray(sphere),
                projection_2d=np.asarray(plane),
                magnitudes=np.asarray(block["magnitude"]),
                hard_assignments=np.asarray(
                    assignments_by_feature.get(feature_id, []), dtype=np.int64
                ),
            )
            source_metrics = _load_json(run_dir / candidate["metrics_path"])
            projection = source_metrics.get("projection") or {}
            metadata = {
                "source_tag": source_tag,
                "feature_id": feature_id,
                "rank": rank,
                "family": family,
                "source_metrics": {
                    key: source_metrics.get(key)
                    for key in (
                        "color",
                        "evidence_status",
                        "family_metrics",
                        "feature_id",
                        "label_confidence",
                        "m_median",
                        "n_valid",
                        "primary_label",
                        "secondary_flags",
                        "terminal_reason",
                    )
                },
            }
            metadata["source_metrics"]["projection"] = {
                key: projection.get(key)
                for key in (
                    "axis_display_sign_flipped",
                    "mode_coloring",
                    "projection_2d_display_kind",
                    "projection_2d_display_transform",
                    "projection_2d_explained_ratios",
                    "projection_2d_kind",
                    "residual_selected_k",
                    "sphere_explained_ratios",
                    "sphere_kind",
                )
            }
            _write_json(candidate_dir / "metadata.json", metadata)

            # Copy only the two first-release figure types, never sphere-ball or cards.
            source_candidate_dir = (run_dir / candidate["metrics_path"]).parent
            for figure_name in ("sphere_surface.png", "projection_2d.png"):
                shutil.copy2(
                    source_candidate_dir / figure_name, candidate_dir / figure_name
                )
            summaries.append(
                {
                    "source_tag": source_tag,
                    "family": family,
                    "rank": rank,
                    "feature_id": feature_id,
                    "path": str(relative_dir),
                }
            )
        del payload
    return sorted(summaries, key=lambda row: (row["family"], row["rank"]))


def _curate_atlas(primary_dir: Path, bundle_dir: Path) -> None:
    """Extract the full compact atlas coordinates and copy its source PNG.

    Args:
        primary_dir: Primary completed source FEGA task directory.
        bundle_dir: Compact reproduction bundle root.
    """
    # Retain only points and fields actually drawn by the source atlas.
    atlas = _load_json(primary_dir / "geometry_reporting" / "geometry_map_data.json")
    figure_metadata = atlas["figure_metadata"]
    included_labels = set(figure_metadata["atlas_label_counts"])
    features = [
        {
            "feature_id": row["feature_id"],
            "label": row["atlas_label"],
            "x": row["embedding"]["x"],
            "y": row["embedding"]["y"],
            "m_median": row.get("m_median"),
        }
        for row in atlas["features"]
        if row["atlas_label"] in included_labels
    ]
    if len(features) != int(figure_metadata["atlas_point_count"]):
        raise ValueError("curated atlas point count does not match the source figure")
    _write_json(
        bundle_dir / "atlas.json",
        {
            "palette": {
                label: atlas["palette"][label] for label in sorted(included_labels)
            },
            "size_policy": atlas["size_policy"],
            "features": features,
        },
    )

    # Preserve the source atlas as the visual comparison target.
    shutil.copy2(
        primary_dir / "geometry_reporting" / "figures" / "geometry_atlas.png",
        bundle_dir / "geometry_atlas.png",
    )


def _write_test_fixture(bundle_dir: Path, fixture_dir: Path) -> None:
    """Write the smallest real-kernel CPU reference fixture.

    Args:
        bundle_dir: Completed compact reproduction bundle.
        fixture_dir: Script-owned local CPU fixture directory.
    """
    # Reuse one curated real feature rather than reopening or copying a source shard.
    real_candidate = bundle_dir / "candidates" / "directed_ray" / "rank_01_f33760"
    geometry = np.load(real_candidate / "geometry.npz")
    np.savez_compressed(
        fixture_dir / "real_feature_33760.npz",
        kernel=geometry["kernel"],
        sphere_coordinates=geometry["sphere_coordinates"],
        magnitudes=geometry["magnitudes"],
    )
    metadata = _load_json(real_candidate / "metadata.json")
    _write_json(
        fixture_dir / "real_feature_33760.json",
        {
            "source_run": PRIMARY_RUN,
            "feature_id": 33760,
            "family": "directed_ray",
            "source_metrics": metadata["source_metrics"],
        },
    )


def main() -> None:
    """Parse the cache root and regenerate the two repository-owned outputs."""
    # Keep the destructive destinations fixed while accepting one source cache root.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_root", type=Path)
    args = parser.parse_args()
    cache_root = args.cache_root.expanduser().resolve()
    primary_dir = cache_root / PRIMARY_RUN
    multi_mode_dir = cache_root / MULTI_MODE_RUN
    bundle_dir = BUNDLE_DIR
    fixture_dir = FIXTURE_DIR

    # Replace only generator-owned outputs and curate the primary plus one supplement.
    _replace_generated_dir(bundle_dir)
    _replace_generated_dir(fixture_dir)
    _curate_atlas(primary_dir, bundle_dir)
    candidates = _curate_candidates(
        primary_dir,
        bundle_dir,
        only_family=None,
        top_k=1,
        source_tag="standard_new_primary",
    )
    candidates.extend(
        _curate_candidates(
            multi_mode_dir,
            bundle_dir,
            only_family=MULTI_MODE_FAMILY,
            top_k=1,
            source_tag="matryoshka_multi_mode_supplement",
        )
    )

    # Publish compact provenance and local CPU fixtures after all candidates exist.
    _write_json(
        bundle_dir / "index.json",
        {
            "primary_run": PRIMARY_RUN,
            "multi_mode_supplement_run": MULTI_MODE_RUN,
            "representatives_per_family": 1,
            "candidates": sorted(
                candidates, key=lambda row: (row["family"], row["rank"])
            ),
        },
    )
    _write_test_fixture(bundle_dir, fixture_dir)


if __name__ == "__main__":
    main()
