"""Build the compact FEGA inputs and original-result comparisons.

Run from the Murano repository root with the historical FEGA source/cache root::

    uv run python notebooks/reproductions/fega/curate_reference.py \
        /path/to/FEGA

The tool reads completed artifacts even when ``run_status.json`` is stale. It
replaces the notebook bundle only after all eight raw clouds, answer records,
and sixteen source-rendered thumbnails validate.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


STANDARD_RUN = (
    "results/fega/ravel/"
    "saebench_gemma-2-2b_width-2pow16_date-0107_"
    "gemma-2-2b_standard_new_width-2pow16_date-0107_"
    "resid_post_layer_12_trainer_2_custom_sae_eval_results.json/city_Country"
)
MATRYOSHKA_RUN = (
    "results/fega/ravel/"
    "saebench_gemma-2-2b_width-2pow16_date-0107_"
    "gemma-2-2b_matryoshka_batch_top_k_width-2pow16_date-0107_"
    "resid_post_layer_12_trainer_2_custom_sae_eval_results.json/city_Country"
)
BUNDLE_DIR = Path(__file__).resolve().parent / "artifacts"
FEATURES = (
    (33760, "directed_ray", "standard_new", "relu"),
    (14513, "axis_or_antipodal", "standard_new", "relu"),
    (54361, "global_2D_directional_subspace", "standard_new", "relu"),
    (32542, "global_kD_directional_subspace", "standard_new", "relu"),
    (59154, "oneD_diffuse", "standard_new", "relu"),
    (19224, "residual_lowD_k", "standard_new", "relu"),
    (34636, "unresolved_high_dimensional_or_diffuse", "standard_new", "relu"),
    (1425, "multi_mode_directional_geometry", "matryoshka", "matryoshka"),
)
PRIVATE_PROVENANCE_KEYS = frozenset(
    {
        "executed_plan_digests",
        "plan_digest",
        "point_record_sha256",
        "schedule_digest",
        "source_paths",
    }
)

_SOURCE_CAPTURE_SCRIPT = r"""
import copy
import json
import pathlib
import sys

import numpy as np
import torch

request_path, response_path = map(pathlib.Path, sys.argv[1:3])
request = json.loads(request_path.read_text())
source_root = pathlib.Path(request["source_root"]).resolve()
external_root = source_root / "external"
sys.path[:0] = [str(source_root), str(external_root)]

import fega.core.visualizations.runner as runner

for name, module in tuple(sys.modules.items()):
    if name == "fega" or name.startswith("fega."):
        if getattr(module, "__file__", None) is None:
            continue
        origin = pathlib.Path(module.__file__).resolve()
        if not origin.is_relative_to(source_root):
            raise RuntimeError(f"non-source FEGA import: {name}={origin}")
    if name == "sae_bench" or name.startswith("sae_bench."):
        if getattr(module, "__file__", None) is None:
            continue
        origin = pathlib.Path(module.__file__).resolve()
        if not origin.is_relative_to(external_root):
            raise RuntimeError(f"non-vendored sae_bench import: {name}={origin}")


def encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pathlib.Path):
        return value.name
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return copy.deepcopy(value)


def load_json(path):
    return json.loads(path.read_text())


def feature_rows(run_dir, feature_id):
    manifest = load_json(
        run_dir / "compute_effect/final_resid/effect_tensors_manifest.json"
    )
    shard = next(row for row in manifest["shards"] if feature_id in row["feature_ids"])
    payload = torch.load(
        run_dir / "compute_effect/final_resid" / pathlib.Path(shard["path"]).name,
        map_location="cpu",
        weights_only=True,
    )
    index = payload["feature_ids"].tolist().index(feature_id)
    start = int(payload["row_offsets"][index])
    stop = int(payload["row_offsets"][index + 1])
    return payload["direction"][start:stop]


responses = []
originals = {
    name: getattr(runner, name)
    for name in (
        "render_sphere",
        "render_projection_2d",
        "render_subspace_plane",
        "render_residual_view",
    )
}
runner._relative = lambda path, run_dir: pathlib.Path(path).name
for item in request["features"]:
    run_dir = pathlib.Path(item["run_dir"])
    feature_id = int(item["feature_id"])
    family = item["family"]
    records = load_json(
        run_dir / "geometry_reporting/geometry_feature_records.json"
    )["features"]
    record = next(row for row in records if int(row["feature_id"]) == feature_id)
    vmf_rows = load_json(run_dir / "vmf/pre_softcap_logits/vmf_scores.json")["features"]
    vmf = next(row for row in vmf_rows if int(row["feature_id"]) == feature_id)
    assignments = (vmf.get("selected_fit") or {}).get("hard_assignments")
    captured = {}

    def wrap(name):
        original = originals[name]

        def delegated(*args, **kwargs):
            destination = pathlib.Path(args[0])
            if destination.name == "sphere_surface.png":
                captured["sphere_surface"] = {
                    "coordinates": encode(args[1]),
                    "kwargs": encode(kwargs),
                }
            elif destination.name == "projection_2d.png":
                captured["projection_2d"] = {
                    "coordinates": encode(args[1]),
                    "kwargs": encode(kwargs),
                }
            return original(*args, **kwargs)

        return delegated

    for name in originals:
        setattr(runner, name, wrap(name))
    output_dir = pathlib.Path(item["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        runner._render_candidate(
            output_dir,
            {
                "feature_id": feature_id,
                "primary_label": family,
                "rank": 1,
                "record": record,
            },
            directions=feature_rows(run_dir, feature_id),
            gram=torch.load(
                run_dir / "data_prep/gram_cache/gram.pt",
                map_location="cpu",
                weights_only=True,
            ),
            color=item["color"],
            dpi=int(item["dpi"]),
            run_dir=run_dir,
            mode_assignments=assignments,
        )
    finally:
        for name, original in originals.items():
            setattr(runner, name, original)
    if set(captured) != {"sphere_surface", "projection_2d"}:
        raise RuntimeError(f"incomplete renderer capture for feature {feature_id}")
    responses.append(
        {
            "feature_id": feature_id,
            "render_inputs": captured,
        }
    )

response_path.write_text(
    json.dumps(
        {
            "source_imports_verified": True,
            "entrypoint": "fega.core.visualizations.runner._render_candidate",
            "features": responses,
        },
        separators=(",", ":"),
    )
)
"""


def _load_json(path: Path) -> dict[str, Any]:
    """Load one trusted source JSON object.

    Args:
        path: Existing source artifact.

    Returns:
        Decoded top-level mapping.
    """
    # Decode only the existing source artifact format.
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    """Write compact portable JSON suitable for the committed bundle.

    Args:
        path: Output artifact path.
        payload: JSON-compatible content.
    """
    # Keep large captured coordinate arrays compact while retaining exact values.
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")


def _source_block(run_dir: Path, feature_id: int) -> dict[str, Any]:
    """Load one literal delta block and its ordered row identity.

    Args:
        run_dir: Completed source task directory.
        feature_id: Feature to extract.

    Returns:
        Source shard, row bounds, literal deltas, and aligned metadata.
    """
    # Resolve the shard containing the requested feature.
    manifest = _load_json(
        run_dir / "compute_effect/final_resid/effect_tensors_manifest.json"
    )
    shard = next(row for row in manifest["shards"] if feature_id in row["feature_ids"])
    payload = torch.load(
        run_dir / "compute_effect/final_resid" / Path(shard["path"]).name,
        map_location="cpu",
        weights_only=True,
    )
    index = payload["feature_ids"].tolist().index(feature_id)
    start = int(payload["row_offsets"][index])
    stop = int(payload["row_offsets"][index + 1])
    delta = payload["delta"][start:stop]
    if delta.dtype != torch.float32:
        raise ValueError(f"feature {feature_id} source delta is not float32")

    # Prove the retained candidate identity is the exact persisted shard order.
    identity = payload["candidate_identity"][index]
    retained = [
        row
        for row, keep in zip(identity, payload["retained_mask"][index], strict=True)
        if keep
    ]
    rows = [
        {
            "source_context_index": int(payload["context_indices"][row]),
            "attribute_label": str(payload["attribute_labels"][row]),
            "pair_role": str(payload["pair_roles"][row]),
            "pair_index": int(payload["pair_indices"][row]),
        }
        for row in range(start, stop)
    ]
    if retained != [
        {key: row[key] for key in ("attribute_label", "pair_role", "pair_index")}
        for row in rows
    ]:
        raise ValueError(f"feature {feature_id} retained identity is misaligned")
    return {"delta": delta.numpy(), "rows": rows}


def _group_label(record: dict[str, Any]) -> str | None:
    """Resolve the same first available stability grouping field as FEGA.

    Args:
        record: Source prompt metadata.

    Returns:
        Dimension-qualified group label, or ``None``.
    """
    # Match the published stability metadata precedence exactly.
    for key in (
        "context_split",
        "entity_split",
        "group",
        "split",
        "pair_role",
        "attribute_label",
        "entity_label",
        "attribute_type",
    ):
        value = record.get(key)
        if value not in (None, ""):
            return f"{key}={value}"
    return None


def _row_metadata(
    run_dir: Path, feature_id: int, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Attach resolved groups and exact live replay fields to source rows.

    Args:
        run_dir: Completed source task directory.
        feature_id: Feature to extract.
        rows: Ordered effect-shard identities.

    Returns:
        Complete computation-only row metadata.
    """
    # Reuse the source grouping precedence for the saved contexts.
    contexts = _load_json(run_dir / "data_prep/select/feature_contexts.json")
    selected = {int(row["index"]): row for row in contexts[str(feature_id)]}
    for row in rows:
        context = selected[row["source_context_index"]]
        row["stability_group"] = _group_label(context)
    if feature_id != 33760:
        return rows

    # Join the live feature to full tokens and unpadded logical target positions.
    pairs = _load_json(run_dir / "data_prep/collect/pairs_full.json")
    activation_rows: dict[int, dict[str, Any]] = {}
    activation_dir = run_dir / "data_prep/collect/activations"
    for path in sorted(activation_dir.glob("activations_meta_*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            activation_rows[int(record["index"])] = record
    for row in rows:
        context = selected[row["source_context_index"]]
        pair = pairs[context["attribute_type"]][row["pair_role"]][row["pair_index"]]
        activation = activation_rows[row["source_context_index"]]
        if not (context["prompt"] == pair["text"] == activation["prompt"]):
            raise ValueError("33760 prompt identity differs across source artifacts")
        if len(pair["input_ids"]) != int(activation["prompt_length"]):
            raise ValueError(
                "33760 token length differs from source activation metadata"
            )
        row.update(
            {
                "prompt": context["prompt"],
                "source_token_ids": [int(token) for token in pair["input_ids"]],
                "logical_target_position": int(activation["unpadded_target_position"]),
            }
        )
    return rows


def _without_private_provenance(value: Any) -> Any:
    """Remove source paths and fingerprint-only fields from answer records.

    Args:
        value: Nested source answer value.

    Returns:
        Portable claim-level value with no maintainer paths or hashes.
    """
    # Recursively retain scientific answers while dropping known source provenance.
    if isinstance(value, dict):
        return {
            key: _without_private_provenance(item)
            for key, item in value.items()
            if key not in PRIVATE_PROVENANCE_KEYS
        }
    if isinstance(value, list):
        return [_without_private_provenance(item) for item in value]
    if isinstance(value, str) and value.startswith("/"):
        raise ValueError("private absolute path reached the compact answer bundle")
    return value


def _answers(run_dir: Path, feature_id: int) -> dict[str, Any]:
    """Extract claim-level source geometry, vMF, stability, and reporting answers.

    Args:
        run_dir: Completed source task directory.
        feature_id: Feature to extract.

    Returns:
        Portable answer mapping without source paths or fingerprints.
    """
    # Select the exact source records used by downstream claim comparison.
    geometry = _load_json(
        run_dir / "geometry_metrics/final_resid/geometry_metrics_scores.json"
    )["per_feature"][str(feature_id)]
    reporting = next(
        row
        for row in _load_json(
            run_dir / "geometry_reporting/geometry_feature_records.json"
        )["features"]
        if int(row["feature_id"]) == feature_id
    )
    vmf = next(
        row
        for row in _load_json(run_dir / "vmf/pre_softcap_logits/vmf_scores.json")[
            "features"
        ]
        if int(row["feature_id"]) == feature_id
    )
    stability = _load_json(run_dir / "stability/stability_scores.json")[
        "effect_spaces"
    ]["final_resid"]["per_feature"][str(feature_id)]
    return _without_private_provenance(
        {
            "geometry_metrics": geometry,
            "vmf": vmf,
            "stability": stability,
            "reporting": reporting,
        }
    )


def _source_capture(
    generated_dir: Path, historical_root: Path, feature_requests: list[dict[str, Any]]
) -> dict[str, Any]:
    """Run the original FEGA renderer in its existing environment.

    Args:
        generated_dir: Temporary artifact directory.
        historical_root: Source FEGA checkout containing the completed runs.
        feature_requests: Figures to render.

    Returns:
        Captured exact renderer arrays, kwargs, and import proof.
    """
    # Use the source checkout's environment and implementation.
    request_path = generated_dir / ".capture-request.json"
    response_path = generated_dir / ".capture-response.json"
    request_path.write_text(
        json.dumps({"source_root": str(historical_root), "features": feature_requests})
    )
    environment = {
        **os.environ,
        "MPLCONFIGDIR": str(generated_dir / ".matplotlib"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    subprocess.run(
        [
            str(historical_root / ".venv/bin/python"),
            "-I",
            "-c",
            _SOURCE_CAPTURE_SCRIPT,
            str(request_path),
            str(response_path),
        ],
        check=True,
        env=environment,
    )
    response = _load_json(response_path)
    request_path.unlink()
    response_path.unlink()
    shutil.rmtree(generated_dir / ".matplotlib", ignore_errors=True)
    return response


def _thumbnail(source: Path, destination: Path) -> None:
    """Write one review-quality thumbnail from a source-rendered figure.

    Args:
        source: Fresh source renderer output.
        destination: Final compact artifact path.
    """
    # Bound display size without changing the scientific point population.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.thumbnail((900, 900), Image.Resampling.LANCZOS)
        image.save(destination, optimize=True)


def _validate(
    generated_dir: Path, inputs: dict[str, Any], expected: dict[str, Any]
) -> None:
    """Validate the compact artifacts before replacing the existing bundle.

    Args:
        generated_dir: Complete temporary bundle.
        inputs: Computation-only metadata.
        expected: Answer-only source records.
    """
    # Check the eight feature slices and raw effect array.
    with np.load(generated_dir / "effect_clouds.npz") as clouds:
        if set(clouds.files) != {"delta", "row_offsets"}:
            raise ValueError("effect_clouds.npz contains derived arrays")
        if clouds["delta"].dtype != np.float32 or clouds["row_offsets"].shape != (9,):
            raise ValueError("effect cloud has unexpected dtype or row offsets")
        offsets = clouds["row_offsets"].tolist()
        delta_count = len(clouds["delta"])
    if [row["feature_id"] for row in inputs["features"]] != [
        row[0] for row in FEATURES
    ]:
        raise ValueError("input feature order differs from the saved metadata")
    if len(expected["features"]) != 8 or offsets[-1] != delta_count:
        raise ValueError("answer inventory or effect row count is incomplete")
    if len(inputs["features"][0]["rows"]) != 64:
        raise ValueError("feature 33760 does not contain 64 ordered rows")

    # Keep original answers out of the computation inputs.
    forbidden_keys = (
        "expected_family",
        "geometry_metrics",
        "vmf",
        "stability",
        "reporting",
        "render_inputs",
        "coordinates",
        "image_paths",
        "kernel",
        "magnitude",
        "direction",
    )
    input_text = json.dumps(inputs).lower()
    if any(f'"{key}"' in input_text for key in forbidden_keys) or "/mnt/" in input_text:
        raise ValueError("computation inputs contain answer or private-path data")
    expected_text = json.dumps(expected)
    if "/mnt/" in expected_text:
        raise ValueError("expected answers contain a private absolute path")
    for _, family, _, _ in FEATURES:
        for name in ("sphere_surface.png", "projection_2d.png"):
            path = generated_dir / "source_figures" / family / name
            with Image.open(path) as image:
                if image.width < 100 or image.height < 100:
                    raise ValueError(f"source thumbnail is unreadable: {path}")


def main() -> None:
    """Regenerate the compact bundle from a completed source FEGA run."""
    # Read the source checkout and build into a temporary sibling directory.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("historical_root", type=Path)
    args = parser.parse_args()
    historical_root = args.historical_root.expanduser().resolve()
    generated_dir = Path(
        tempfile.mkdtemp(prefix=".fega-artifacts-", dir=BUNDLE_DIR.parent)
    )
    try:
        deltas: list[np.ndarray] = []
        offsets = [0]
        input_features = []
        expected_features = []
        render_requests = []
        render_root = generated_dir / ".source-renders"
        for feature_id, family, source_tag, sae_kind in FEATURES:
            run_dir = historical_root / (
                STANDARD_RUN if source_tag == "standard_new" else MATRYOSHKA_RUN
            )
            block = _source_block(run_dir, feature_id)
            rows = _row_metadata(run_dir, feature_id, block["rows"])
            deltas.append(block["delta"])
            offsets.append(offsets[-1] + len(block["delta"]))
            input_features.append(
                {
                    "feature_id": feature_id,
                    "source_run": source_tag,
                    "sae_kind": sae_kind,
                    "row_start": offsets[-2],
                    "row_stop": offsets[-1],
                    "row_count": len(rows),
                    "rows": rows,
                }
            )
            answer = _answers(run_dir, feature_id)
            answer.update({"feature_id": feature_id, "expected_family": family})
            expected_features.append(answer)
            candidate_index = _load_json(run_dir / "visualizations/candidates.json")
            candidate = next(
                row
                for row in candidate_index["families"][family]
                if int(row["feature_id"]) == feature_id
            )
            render_requests.append(
                {
                    "feature_id": feature_id,
                    "family": family,
                    "run_dir": str(run_dir),
                    "color": candidate["color"],
                    "dpi": 180,
                    "output_dir": str(render_root / family),
                }
            )

        # Save the raw effects and their feature boundaries.
        np.savez_compressed(
            generated_dir / "effect_clouds.npz",
            delta=np.concatenate(deltas).astype(np.float32, copy=False),
            row_offsets=np.asarray(offsets, dtype=np.int64),
        )
        inputs = {"features": input_features}
        capture = _source_capture(generated_dir, historical_root, render_requests)
        captures = {int(row["feature_id"]): row for row in capture["features"]}
        for answer in expected_features:
            captured = captures[int(answer["feature_id"])]
            answer["render_inputs"] = captured["render_inputs"]
            family = answer["expected_family"]
            for name in ("sphere_surface.png", "projection_2d.png"):
                _thumbnail(
                    render_root / family / name,
                    generated_dir / "source_figures" / family / name,
                )
        expected = {
            "capture_provenance": {
                "source_imports_verified": capture["source_imports_verified"],
                "entrypoint": capture["entrypoint"],
            },
            "features": expected_features,
        }
        shutil.rmtree(render_root)
        _write_json(generated_dir / "inputs.json", inputs)
        _write_json(generated_dir / "expected.json", expected)
        _validate(generated_dir, inputs, expected)

        # Replace the bundle only after every new artifact validates.
        if BUNDLE_DIR.exists():
            shutil.rmtree(BUNDLE_DIR)
        generated_dir.replace(BUNDLE_DIR)
        total_bytes = sum(
            path.stat().st_size for path in BUNDLE_DIR.rglob("*") if path.is_file()
        )
        print(
            f"wrote 8 features, 16 thumbnails, {offsets[-1]} rows, {total_bytes} bytes"
        )
        print("source FEGA imports verified under the historical checkout")
    finally:
        if generated_dir.exists():
            shutil.rmtree(generated_dir)


if __name__ == "__main__":
    main()
