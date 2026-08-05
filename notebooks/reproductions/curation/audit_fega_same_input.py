"""Run three bounded FEGA maintainer comparisons against one external cache."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from murano import keys
from murano.fega.artifacts import (
    FEGAEffectStore,
    FEGAFeatureEffects,
    FEGAVMFResult,
)
from murano.fega.config import FEGAConfig
from murano.fega.contexts import FEGAContext
from murano.fega.visualization import project_directions, surface_coordinates
from murano.fega.vmf import (
    _derived_seed,
    assignment_stability,
    feature_seed,
    select_vmf,
)
from murano.model import MuranoModel
from murano.results import Results
from murano.steps.fega import (
    _materialize_vmf_coordinates,
    _unembedding_fingerprint,
)
from murano.steps.fega_analysis import (
    FEGAGeometryMetrics,
    FEGAGeometryReporting,
    FEGAStability,
    FEGAVisualize,
)


FEATURE_ID = 33760
MODEL_ID = "google/gemma-2-2b"
SOURCE_TOLERANCES = {
    "reconstruction": {"rtol": 1.0e-6, "atol": 1.0e-6},
    "norms": {"rtol": 1.0e-5, "atol": 1.0e-6},
    "inner_products": {"rtol": 1.0e-5, "atol": 1.0e-5},
    "geometry": {"rtol": 1.0e-4, "atol": 1.0e-4},
}


class MissingPrerequisite(RuntimeError):
    """Mark unavailable external data or hardware without hiding code failures."""


def _load_json(path: Path) -> dict[str, Any]:
    """Load one trusted source JSON mapping."""
    # Require the expected top level because every caller reads named source fields.
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return payload


def _comparison(
    actual: torch.Tensor | np.ndarray,
    expected: torch.Tensor | np.ndarray,
    *,
    rtol: float | None = None,
    atol: float | None = None,
) -> dict[str, Any]:
    """Return path-free numerical diagnostics for two arrays."""
    # Compare in float64 and report a decision only when a tolerance is authorized.
    left = torch.as_tensor(actual, dtype=torch.float64).cpu()
    right = torch.as_tensor(expected, dtype=torch.float64).cpu()
    if left.shape != right.shape:
        return {"status": "mismatch", "shape_match": False}
    error = (left - right).abs()
    result: dict[str, Any] = {
        "shape_match": True,
        "count": left.numel(),
        "max_absolute_error": float(error.max()) if error.numel() else 0.0,
    }
    if rtol is None or atol is None:
        result["status"] = "not_comparable"
        return result
    close = torch.isclose(left, right, rtol=rtol, atol=atol, equal_nan=False)
    result.update(
        status="match" if bool(close.all()) else "mismatch",
        rtol=rtol,
        atol=atol,
        exceedance_count=int((~close).sum()),
    )
    return result


def _source_rows(run_dir: Path) -> dict[str, Any]:
    """Read normalized activation rows and feature selection in source order."""
    # Stream source chunks only to establish ordered activation metadata identity.
    activation_dir = run_dir / "data_prep" / "collect" / "activations"
    manifest = _load_json(activation_dir / "activations_manifest.json")
    metadata: list[dict[str, Any]] = []
    for chunk in manifest["chunks"]:
        tensors = torch.load(
            activation_dir / Path(chunk["tensors"]).name,
            map_location="cpu",
            weights_only=True,
        )
        rows = [
            json.loads(line)
            for line in (activation_dir / Path(chunk["meta"]).name)
            .read_text()
            .splitlines()
        ]
        if [int(row["index"]) for row in rows] != [
            int(value) for value in tensors["index"]
        ]:
            raise ValueError("Activation tensor and metadata indices disagree")
        metadata.extend(rows)
    if [int(row["index"]) for row in metadata] != list(range(len(metadata))):
        raise ValueError("Source activation rows are not in normalized index order")

    # Preserve only metadata and the exact cached selection used by later checks.
    selected = _load_json(run_dir / "data_prep/select/feature_contexts.json")[
        str(FEATURE_ID)
    ]
    return {
        "metadata": metadata,
        "metadata_by_index": {int(row["index"]): row for row in metadata},
        "selected_rows": selected,
    }


def _effect_block(run_dir: Path) -> dict[str, Any]:
    """Load the exact cached feature block and candidate identities."""
    # Slice only the audited feature range declared by the completed effect summary.
    effect_dir = run_dir / "compute_effect/final_resid"
    manifest = _load_json(effect_dir / "effect_tensors_manifest.json")
    summary = _load_json(effect_dir / "effect_summary.json")["per_feature"][
        str(FEATURE_ID)
    ]
    declared = next(
        row for row in manifest["shards"] if FEATURE_ID in row["feature_ids"]
    )
    if Path(declared["path"]).name != summary["tensor_shard"]:
        raise ValueError("Effect manifest and feature summary disagree")
    payload = torch.load(
        effect_dir / str(summary["tensor_shard"]),
        map_location="cpu",
        weights_only=True,
    )
    start, stop = int(summary["row_start"]), int(summary["row_end"])
    return {
        "summary": summary,
        "context_indices": tuple(
            int(value) for value in payload["context_indices"][start:stop].tolist()
        ),
        "pair_indices": tuple(
            int(value) for value in payload["pair_indices"][start:stop].tolist()
        ),
        "attribute_labels": tuple(payload["attribute_labels"][start:stop]),
        "pair_roles": tuple(payload["pair_roles"][start:stop]),
        "feature_activations": payload["feature_activations"][start:stop].float(),
        "delta": payload["delta"][start:stop].float(),
        "directions": payload["direction"][start:stop].float(),
        "magnitudes": payload["magnitude"][start:stop].float(),
        "retained_mask": tuple(bool(value) for value in summary["retained_mask"]),
    }


def _identity_rows(rows: list[dict[str, Any]]) -> list[tuple[str, str, int]]:
    """Return complete ordered source pair identities."""
    # Keep attribute, role, and pair index together to prevent local-index collisions.
    return [
        (str(row["attribute_label"]), str(row["pair_role"]), int(row["pair_index"]))
        for row in rows
    ]


def _extract_group_label(record: dict[str, Any]) -> str | None:
    """Apply source FEGA's ordered group-dimension precedence."""
    # Prefix the chosen dimension so equal values from different fields never merge.
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


def _group_data(
    run_dir: Path, source: dict[str, Any], cached: dict[str, Any]
) -> dict[str, Any]:
    """Build pair-first and context-fallback group lookup data."""
    # Reproduce source lookup keys and retain pair records needed for token identities.
    pairs = _load_json(run_dir / "data_prep/collect/pairs_full.json")
    pair_records: dict[tuple[str, str, int], dict[str, Any]] = {}
    pair_labels: dict[tuple[str, str, int], str | None] = {}
    for attribute_label, roles in pairs.items():
        for raw_role, prompts in roles.items():
            role = str(raw_role).removesuffix("_prompts")
            for pair_index, prompt in enumerate(prompts):
                if isinstance(prompt, dict):
                    pair_labels[(str(attribute_label), role, pair_index)] = (
                        _extract_group_label(prompt)
                    )
                    pair_records[
                        (str(prompt.get("attribute_label")), str(raw_role), pair_index)
                    ] = prompt

    # Prefer pair identity and use ordered activation metadata only as fallback.
    context_labels: dict[int, str | None] = {}
    for row in [*source["selected_rows"], *source["metadata"]]:
        label = _extract_group_label(row)
        if label is not None:
            context_labels.setdefault(int(row["index"]), label)
    labels = []
    for context_index, pair_index, attribute, role in zip(
        cached["context_indices"],
        cached["pair_indices"],
        cached["attribute_labels"],
        cached["pair_roles"],
        strict=True,
    ):
        label = pair_labels.get((attribute, role, pair_index))
        labels.append(context_labels.get(context_index) if label is None else label)
    return {
        "labels": labels,
        "pair_records": pair_records,
        "pair_labels": pair_labels,
        "context_labels": context_labels,
    }


def _cached_contexts(
    source: dict[str, Any], cached: dict[str, Any], groups: dict[str, Any]
) -> tuple[FEGAContext, ...]:
    """Construct retained Murano contexts from cached source metadata."""
    # Bind each effect row to one exact selected prompt and token identity.
    selected = {int(row["index"]): row for row in source["selected_rows"]}
    contexts = []
    for context_index, attribute, role, pair_index, label in zip(
        cached["context_indices"],
        cached["attribute_labels"],
        cached["pair_roles"],
        cached["pair_indices"],
        groups["labels"],
        strict=True,
    ):
        row = selected[context_index]
        activation = source["metadata_by_index"][context_index]
        pair = groups["pair_records"][(attribute, role, pair_index)]
        if (
            str(row["attribute_label"]),
            str(row["pair_role"]),
            int(row["pair_index"]),
        ) != (attribute, role, pair_index):
            raise ValueError("Selected context and cached effect identity disagree")
        contexts.append(
            FEGAContext(
                index=context_index,
                prompt=str(row["prompt"]),
                input_ids=tuple(int(token) for token in pair["input_ids"]),
                target_position=int(activation["unpadded_target_position"]),
                attribute_label=attribute,
                pair_role=role,
                pair_index=pair_index,
                group_label=label,
            )
        )
    return tuple(contexts)


def _source_downstream(run_dir: Path) -> dict[str, Any]:
    """Read the completed cached downstream records for the audited feature."""
    # Slice only feature 33760 from each authoritative completed-phase artifact.
    key = str(FEATURE_ID)
    vmf_payload = _load_json(run_dir / "vmf/pre_softcap_logits/vmf_scores.json")
    reporting_payload = _load_json(
        run_dir / "geometry_reporting/geometry_feature_records.json"
    )
    candidates = _load_json(run_dir / "visualizations/candidates.json")
    candidate = next(
        row
        for rows in candidates["families"].values()
        for row in rows
        if row["feature_id"] == FEATURE_ID
    )
    return {
        "geometry": _load_json(
            run_dir / "geometry_metrics/final_resid/geometry_metrics_scores.json"
        )["per_feature"][key],
        "vmf": next(
            row for row in vmf_payload["features"] if row["feature_id"] == FEATURE_ID
        ),
        "stability": _load_json(run_dir / "stability/stability_scores.json")[
            "effect_spaces"
        ]["final_resid"]["per_feature"][key],
        "reporting": next(
            row
            for row in reporting_payload["features"]
            if row["feature_id"] == FEATURE_ID
        ),
        "visualization": _load_json(run_dir / str(candidate["metrics_path"])),
    }


def _cached_runtime(
    run_dir: Path,
    model: MuranoModel,
    source: dict[str, Any],
    cached: dict[str, Any],
) -> dict[str, Any]:
    """Run Murano downstream helpers once on the exact cached effect cloud."""
    # Bind the cached Gram to the live canonical unembedding before any analysis.
    gram = torch.load(
        run_dir / "data_prep/gram_cache/gram.pt",
        map_location="cpu",
        weights_only=True,
    )
    gram_meta = _load_json(run_dir / "data_prep/gram_cache/gram_meta.json")
    if gram.dtype != torch.float64 or gram_meta["gram_dtype"] != "float64":
        raise ValueError("Cached comparison requires the source float64 Gram")
    fingerprint = _unembedding_fingerprint(model.unembed_weight)
    if fingerprint != gram_meta["unembedding_fingerprint"]:
        raise ValueError("Canonical unembedding does not match cached Gram")

    # Materialize one transient vocabulary matrix and reuse it for both current runs.
    groups = _group_data(run_dir, source, cached)
    contexts = _cached_contexts(source, cached, groups)
    effects = FEGAEffectStore(
        features={
            FEATURE_ID: FEGAFeatureEffects(
                feature_id=FEATURE_ID,
                directions=cached["directions"],
                magnitudes=cached["magnitudes"],
                context_indices=cached["context_indices"],
                feature_activations=cached["feature_activations"],
                retained_mask=cached["retained_mask"],
                contexts=contexts,
            )
        },
        gram=gram,
        unembedding_fingerprint=fingerprint,
        metadata={"effect_sign": "ablated-minus-baseline"},
        analysis_id="maintainer-cache-comparison",
    )
    config = FEGAConfig(seed=42)
    coordinates = _materialize_vmf_coordinates(
        cached["directions"], model.unembed_weight.detach()
    )
    cloud = coordinates.numpy()
    seed = feature_seed(config.seed, FEATURE_ID)
    selection = select_vmf(
        cloud.copy(),
        config.vmf_k_values,
        seed,
        config.vmf_n_init,
        config.vmf_max_iter,
        config.vmf_bic_tolerance,
        n_jobs=1,
        warn_large=False,
    )
    stability = assignment_stability(
        cloud.copy(),
        selection.selected,
        seed,
        config.vmf_resample_fraction,
        config.vmf_resample_rounds,
        1,
        config.vmf_n_init,
        config.vmf_max_iter,
    )
    results = Results()
    results[keys.FEGA_EFFECTS] = effects
    FEGAGeometryMetrics(config)(results)
    results[keys.FEGA_VMF] = FEGAVMFResult(
        {FEATURE_ID: selection},
        {FEATURE_ID: stability},
        fingerprint,
        effects.analysis_id,
        {FEATURE_ID: "fitted"},
    )
    FEGAStability(config)(results)
    FEGAGeometryReporting()(results)
    return {
        "config": config,
        "coordinates": coordinates,
        "contexts": contexts,
        "effects": effects,
        "groups": groups,
        "results": results,
        "selection": selection,
    }


def _geometry_checks(
    cached: dict[str, Any], gram: torch.Tensor, actual: Any, source: dict[str, Any]
) -> dict[str, Any]:
    """Check cached factorization, Gram geometry, and published point metrics."""
    # Compare only quantities whose historical cache retains both sides.
    reconstructed = cached["directions"] * cached["magnitudes"][:, None]
    delta = cached["delta"].to(torch.float64)
    directions = cached["directions"].to(torch.float64)
    magnitudes = cached["magnitudes"].to(torch.float64)
    delta_kernel = delta @ gram @ delta.T
    direction_kernel = directions @ gram @ directions.T
    checks = {
        "reconstruction": _comparison(
            reconstructed, cached["delta"], **SOURCE_TOLERANCES["reconstruction"]
        ),
        "norms": _comparison(
            torch.sqrt(torch.clamp_min(torch.diag(delta_kernel), 0.0)),
            magnitudes,
            **SOURCE_TOLERANCES["norms"],
        ),
        "inner_products": _comparison(
            direction_kernel,
            delta_kernel / (magnitudes[:, None] * magnitudes[None, :]),
            **SOURCE_TOLERANCES["inner_products"],
        ),
    }
    for field, value in (
        ("c_ray", actual.c_ray),
        ("s_span_1", actual.s_span.get(1)),
        ("s_res_1", actual.s_res.get(1)),
    ):
        checks[field] = _comparison(
            np.asarray([value]),
            np.asarray([source[field]]),
            **SOURCE_TOLERANCES["geometry"],
        )
    checks["status"] = (
        "match"
        if all(row["status"] == "match" for row in checks.values())
        else "mismatch"
    )
    return checks


def _historical_vmf(actual: Any, source: dict[str, Any], n_rows: int) -> dict[str, Any]:
    """Compare only vMF fields persisted with historical authority."""
    # Compare the selected partition and persisted assignment-resampling identities.
    selected = actual.selected
    expected = source["selected_fit"]
    expected_labels = np.asarray(expected["hard_assignments"], dtype=np.int64)
    subset_size = max(
        selected.n_components, math.ceil(FEGAConfig().vmf_resample_fraction * n_rows)
    )
    fit_seed = feature_seed(FEGAConfig().seed, FEATURE_ID)
    subsets_match = True
    for replicate in source["assignment_stability"]["replicates"]:
        replicate_id = int(replicate["replicate_id"])
        subset_seed = _derived_seed(
            fit_seed, selected.n_components, replicate_id, "subset"
        )
        refit_seed = _derived_seed(
            fit_seed, selected.n_components, replicate_id, "refit"
        )
        indices = np.sort(
            np.random.default_rng(subset_seed).choice(
                n_rows, subset_size, replace=False
            )
        ).tolist()
        subsets_match &= (
            subset_seed == int(replicate["subset_seed"])
            and refit_seed == int(replicate["refit_seed"])
            and indices == replicate["subset_indices"]
        )
    selected_k_match = selected.n_components == int(
        source["metrics"]["selected_mode_count"]
    )
    assignments_match = adjusted_rand_score(expected_labels, selected.labels) == 1.0
    counts_match = sorted(np.bincount(selected.labels).tolist()) == sorted(
        expected["hard_mode_counts"]
    )
    return {
        "status": (
            "match"
            if selected_k_match and assignments_match and counts_match and subsets_match
            else "mismatch"
        ),
        "selected_k_match": selected_k_match,
        "partition_match": assignments_match,
        "mode_counts_match": counts_match,
        "assignment_resampling_subsets_match": subsets_match,
        "responsibilities": {
            "status": "not_comparable",
            "reason": "artifact_not_persisted",
        },
        "kappas": {
            "status": "not_comparable",
            "reason": "historical_run_not_governed_by_current_dense_tolerance",
        },
    }


def _historical_stability(
    actual: dict[str, Any], source: dict[str, Any], report: Any
) -> dict[str, Any]:
    """Compare cached stability decisions while rejecting absent memberships."""
    # Use published availability and decision fields, never plan digests as membership.
    expected_report = source["reporting"]
    expected_decision = str(expected_report["label_confidence"])
    expected_availability = str(expected_report["evidence_status"])
    actual_availability = (
        "unavailable"
        if "selected_family_evidence_unavailable" in report.secondary_flags
        else "available"
    )
    availability_match = actual_availability == expected_availability
    decision_match = str(actual.get("decision")) == expected_decision
    return {
        "status": "match" if availability_match and decision_match else "mismatch",
        "availability_match": availability_match,
        "decision_match": decision_match,
        "memberships": {
            "status": "not_comparable",
            "reason": "explicit_indices_not_persisted",
        },
    }


def _visualization_checks(
    cached: dict[str, Any], gram: torch.Tensor, source: dict[str, Any], files: list[str]
) -> dict[str, Any]:
    """Check the retained sphere-surface and two-dimensional plot contracts."""
    # Recompute source-signed projections and validate only the two approved figures.
    directions = cached["directions"].numpy()
    metric = gram.numpy()
    full = project_directions(directions, metric, dimensions=len(directions))
    sphere = project_directions(directions, metric, dimensions=3)
    surface, _ = surface_coordinates(sphere.coordinates)
    source_projection = source["visualization"]["projection"]
    checks = {
        "coordinate_gram": _comparison(
            full.coordinates @ full.coordinates.T,
            full.kernel,
            rtol=0.0,
            atol=1.0e-10,
        ),
        "eigenvalues": _comparison(
            sphere.eigenvalues[:10],
            np.asarray(source_projection["sphere_top_eigenvalues"]),
            rtol=0.0,
            atol=1.0e-10,
        ),
        "sphere_norms": _comparison(
            np.linalg.norm(surface, axis=1),
            np.ones(len(surface)),
            rtol=0.0,
            atol=1.0e-12,
        ),
    }
    names_match = files == ["projection_2d.png", "sphere_surface.png"]
    contract_match = (
        source_projection["sphere_kind"] == "uncentered_logit_equivalent"
        and source_projection["projection_2d_kind"] == "uncentered_logit_equivalent"
        and {"projection_2d", "sphere_surface"}
        <= set(source["visualization"]["image_paths"])
    )
    return {
        "status": (
            "match"
            if names_match
            and contract_match
            and all(row["status"] == "match" for row in checks.values())
            else "mismatch"
        ),
        "render_names_match": names_match,
        "source_contract_match": contract_match,
        **checks,
    }


def _frozen_cache(
    run_dir: Path,
    cached: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """Compare Murano with fields actually retained by the historical cache."""
    # Render from the cached cloud, then compare every historically authoritative field.
    source = _source_downstream(run_dir)
    results = runtime["results"]
    with tempfile.TemporaryDirectory(prefix="murano-fega-frozen-cache-") as directory:
        FEGAVisualize(
            directory,
            top_k_per_family=1,
            figures=("sphere_surface", "projection_2d"),
            dpi=100,
        )(results)
        files = sorted(path.name for path in results[keys.FEGA_VISUALIZATION].files)
    geometry = _geometry_checks(
        cached,
        runtime["effects"].gram,
        results[keys.FEGA_GEOMETRY].features[FEATURE_ID],
        source["geometry"],
    )
    vmf = _historical_vmf(runtime["selection"], source["vmf"], len(runtime["contexts"]))
    reporting = results[keys.FEGA_REPORTING].features[FEATURE_ID]
    stability = _historical_stability(
        results[keys.FEGA_STABILITY].features[FEATURE_ID], source, reporting
    )
    reporting_match = (
        reporting.primary_label == source["reporting"]["primary_label"]
        and reporting.selected_k == source["reporting"]["selected_k"]
        and list(reporting.secondary_flags) == source["reporting"]["secondary_flags"]
        and list(reporting.global_flags) == source["reporting"]["global_flags"]
    )
    visualization = _visualization_checks(
        cached, runtime["effects"].gram, source, files
    )
    row_identity_match = (
        _identity_rows(cached["summary"]["candidate_identity"])
        == list(
            zip(
                cached["attribute_labels"],
                cached["pair_roles"],
                cached["pair_indices"],
                strict=True,
            )
        )
        and tuple(row.index for row in runtime["contexts"]) == cached["context_indices"]
    )
    matched = (
        row_identity_match
        and geometry["status"] == "match"
        and vmf["status"] == "match"
        and stability["status"] == "match"
        and reporting_match
        and visualization["status"] == "match"
    )
    return {
        "status": "match" if matched else "mismatch",
        "claim": "historical fields persisted for one cached feature cloud",
        "counts": {"retained_rows": len(runtime["contexts"])},
        "row_identity_and_order_match": row_identity_match,
        "targeted_unembedding_match": True,
        "geometry": geometry,
        "vmf": vmf,
        "stability": stability,
        "reporting": {
            "status": "match" if reporting_match else "mismatch",
            "family": reporting.primary_label,
            "selected_k": reporting.selected_k,
        },
        "visualization": visualization,
        "not_claimed": [
            "generation of the cached effect cloud",
            "historical responsibility or kappa agreement",
            "paper-dataset reproduction",
        ],
    }


def _current_source_process(
    source_python: Path,
    source_root: Path,
    cached: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """Run current source and Murano numerics in the source interpreter."""
    # Exchange only transient coordinates, a row kernel, and non-prompt identities.
    if not source_python.is_file() or not (source_root / "fega").is_dir():
        raise MissingPrerequisite("source interpreter or checkout is unavailable")
    helper = Path(__file__).with_name("audit_fega_current_source.py")
    repository = Path(__file__).resolve().parents[3]
    directions = cached["directions"].to(torch.float64)
    gram_kernel = directions @ runtime["effects"].gram @ directions.T
    groups = runtime["groups"]
    metadata = {
        "feature_id": FEATURE_ID,
        "context_indices": list(cached["context_indices"]),
        "pair_indices": list(cached["pair_indices"]),
        "attribute_labels": list(cached["attribute_labels"]),
        "pair_roles": list(cached["pair_roles"]),
        "murano_labels": groups["labels"],
        "context_labels": list(groups["context_labels"].items()),
        "pair_labels": [
            [attribute, role, index, label]
            for (attribute, role, index), label in groups["pair_labels"].items()
        ],
    }
    with tempfile.TemporaryDirectory(prefix="murano-fega-current-source-") as directory:
        work = Path(directory)
        coordinates_path = work / "coordinates.npy"
        kernel_path = work / "gram_kernel.npy"
        metadata_path = work / "metadata.json"
        result_path = work / "result.json"
        np.save(coordinates_path, runtime["coordinates"].numpy(), allow_pickle=False)
        np.save(kernel_path, gram_kernel.numpy(), allow_pickle=False)
        metadata_path.write_text(json.dumps(metadata))
        environment = os.environ.copy()
        python_path = [
            str(repository / "src"),
            str(source_root),
            str(source_root / "external"),
        ]
        if environment.get("PYTHONPATH"):
            python_path.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(python_path)
        subprocess.run(
            [
                str(source_python),
                str(helper),
                str(coordinates_path),
                str(kernel_path),
                str(metadata_path),
                str(source_root),
                str(result_path),
            ],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if not result_path.is_file():
            raise RuntimeError("current-source comparison process failed")
        return _load_json(result_path)


def _exact_effect_process(
    source_python: Path, source_root: Path, run_dir: Path
) -> dict[str, Any]:
    """Run the shared-object live oracle in FEGA's source interpreter."""
    # Validate the three explicit external inputs before launching the helper.
    if not source_python.is_file():
        raise FileNotFoundError(source_python)
    if not (source_root / "fega").is_dir():
        raise FileNotFoundError(source_root / "fega")
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)

    # Reuse the source dependency stack and read only its compact path-free verdict.
    repository = Path(__file__).resolve().parents[3]
    helper = Path(__file__).with_name("audit_fega_effect_call_path.py")
    with tempfile.TemporaryDirectory() as temporary_directory:
        result_path = Path(temporary_directory) / "effect-call-path.json"
        environment = os.environ.copy()
        python_paths = [
            str(repository / "src"),
            str(source_root),
            str(source_root / "external"),
        ]
        if environment.get("PYTHONPATH"):
            python_paths.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(python_paths)
        subprocess.run(
            [
                str(source_python),
                str(helper),
                "--run-dir",
                str(run_dir),
                "--source-root",
                str(source_root),
                "--result-json",
                str(result_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        if not result_path.is_file():
            raise RuntimeError(
                "shared-object live oracle failed before writing a result"
            )
        return _load_json(result_path)


def _section_error(claim: str, error: Exception) -> dict[str, Any]:
    """Return a path-free section result with honest availability semantics."""
    # Preserve diagnostics on stderr while keeping the stored report path-free.
    traceback.print_exception(error)
    return {
        "status": (
            "not_comparable" if isinstance(error, MissingPrerequisite) else "mismatch"
        ),
        "claim": claim,
        "error_type": type(error).__name__,
    }


def main() -> None:
    """Run independent cache, same-coordinate, and live-upstream sections."""
    # Parse explicit external roots and keep the report local to the requested output.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("source_python", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    output = args.output.expanduser().resolve()

    # Load shared external inputs once without allowing one section to mask another.
    try:
        if not run_dir.is_dir():
            raise MissingPrerequisite("completed source cache is unavailable")
        source = _source_rows(run_dir)
        cached = _effect_block(run_dir)
    except Exception as error:
        unavailable = {
            name: _section_error(name, error)
            for name in (
                "frozen_cache",
                "current_source_same_coordinate",
                "live_upstream",
            )
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(unavailable, indent=2, sort_keys=True) + "\n")
        raise SystemExit(1) from error

    try:
        if not torch.cuda.is_available():
            raise MissingPrerequisite("canonical model comparison requires CUDA")
        model = MuranoModel(MODEL_ID, dtype=torch.bfloat16, attn_implementation="eager")
        runtime = _cached_runtime(run_dir, model, source, cached)
        frozen = _frozen_cache(run_dir, cached, runtime)
    except Exception as error:
        model = None
        runtime = None
        frozen = _section_error("historical cache-supported fields", error)

    try:
        if runtime is None:
            raise MissingPrerequisite("Murano coordinate runtime is unavailable")
        current = _current_source_process(
            args.source_python.expanduser(), source_root, cached, runtime
        )
    except Exception as error:
        current = _section_error("current source on one coordinate matrix", error)

    # Release the parent model before the source interpreter loads the shared bundle.
    if model is not None:
        del model
        model = None
        torch.cuda.empty_cache()
    try:
        live = _exact_effect_process(
            args.source_python.expanduser(),
            source_root,
            run_dir,
        )
    except Exception as error:
        live = _section_error("shared live effect and downstream chain", error)

    # Emit exactly three independent statuses and fail only performed mismatches.
    report = {
        "frozen_cache": frozen,
        "current_source_same_coordinate": current,
        "live_upstream": live,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "FEGA maintainer audit: "
        + " ".join(f"{name}={section['status']}" for name, section in report.items())
    )
    if any(section["status"] == "mismatch" for section in report.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
