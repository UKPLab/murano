"""Audit source FEGA and Murano on one shared live feature cloud."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from audit_fega_current_source import _compare as compare_downstream
from fega.core.compute_effect.effects import (
    EffectContextRecord,
    _build_final_resid_rows,
    _empty_compute_stats,
    run_ablation_readouts_batch,
)
from fega.core.compute_effect.prompting import AblationSpec
from fega.core.data_prep.collection import run_sae_reconstruction
from fega.core.data_prep.gram_cache import (
    canonical_unembedding,
    unembedding_fingerprint,
)
from fega.core.positioning import POSITIONING_SCHEMA_VERSION, build_padded_prompt_batch
from fega.core.utils.models import load_model_and_sae
from fega.core.utils.ravel import ReplayContext
from fega.core.vmf.runner import materialize_linear_coordinates
from murano.fega.effects import normalize_effect_rows, run_reconstruction_readout_batch
from murano.steps.fega import _materialize_vmf_coordinates
from sae_bench.evals.ravel.instance import Prompt
from sae_bench.sae_bench_utils import activation_collection


FEATURE_ID = 33760
TAU_ZERO = 1.0e-12
NORMALIZATION_RTOL = 1.0e-5
NORMALIZATION_ATOL = 1.0e-6


def _load_json(path: Path) -> dict[str, Any]:
    """Load one trusted source JSON mapping.

    Args:
        path: Existing source artifact path.

    Returns:
        The decoded top-level mapping.
    """
    # Require the mapping shape consumed by every source-artifact lookup below.
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return payload


def _selected_prompts(run_dir: Path) -> tuple[list[dict[str, Any]], list[Prompt]]:
    """Resolve feature 33760's selected source prompts in canonical order.

    Args:
        run_dir: Completed FEGA task directory containing data-prep artifacts.

    Returns:
        The selected context records and their exact source ``Prompt`` objects.
    """
    # Follow source selection order and recover prompts by their pair identity.
    selected = _load_json(run_dir / "data_prep/select/feature_contexts.json")[
        str(FEATURE_ID)
    ]
    pairs = _load_json(run_dir / "data_prep/collect/pairs_full.json")["Country"]
    prompts = [
        Prompt(**pairs[str(row["pair_role"])][int(row["pair_index"])])
        for row in selected
    ]
    return selected, prompts


def _target_rows(readout: torch.Tensor, positions: list[int]) -> torch.Tensor:
    """Select one target-position row from every rank-three readout.

    Args:
        readout: Model readout tensor shaped ``[batch, sequence, width]``.
        positions: One physical padded position per batch row.

    Returns:
        The ordered rank-two target rows.
    """
    # Gather the same physical rows patched by the source positioning helper.
    rows = torch.arange(readout.shape[0], device=readout.device)
    columns = torch.as_tensor(positions, device=readout.device, dtype=torch.long)
    return readout[rows, columns]


def _exact(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    """Compare two shared-object tensors without serializing their values.

    Args:
        left: Source tensor.
        right: Murano tensor.

    Returns:
        A compact exact-equality result and maximum-error diagnostic.
    """
    # Canonicalize only device placement; the scientific comparison stays exact.
    source = left.detach().cpu()
    murano = right.detach().cpu()
    shape_match = source.shape == murano.shape
    matched = shape_match and torch.equal(source, murano)
    maximum = None
    if shape_match and source.numel():
        maximum = float(
            (source.to(torch.float64) - murano.to(torch.float64)).abs().max()
        )
    return {
        "status": "match" if matched else "mismatch",
        "count": source.numel(),
        "max_absolute_error": maximum,
        "shape_match": shape_match,
    }


def _close(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    rtol: float = NORMALIZATION_RTOL,
    atol: float = NORMALIZATION_ATOL,
) -> dict[str, Any]:
    """Compare independently computed numerical outputs at method tolerance.

    Args:
        left: Source result.
        right: Murano result.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        A compact numerical comparison.
    """
    # Use float64 only for diagnostics while preserving each implementation's result.
    source = left.detach().cpu().to(torch.float64)
    murano = right.detach().cpu().to(torch.float64)
    if source.shape != murano.shape:
        return {"status": "mismatch", "shape_match": False}
    close = torch.isclose(source, murano, rtol=rtol, atol=atol, equal_nan=False)
    error = (source - murano).abs()
    return {
        "status": "match" if bool(close.all()) else "mismatch",
        "shape_match": True,
        "count": source.numel(),
        "max_absolute_error": float(error.max()) if error.numel() else 0.0,
        "rtol": rtol,
        "atol": atol,
        "exceedance_count": int((~close).sum()),
    }


def _group_label(record: dict[str, Any]) -> str | None:
    """Resolve source FEGA's first available stability grouping field.

    Args:
        record: Serialized source prompt.

    Returns:
        A dimension-qualified label or ``None``.
    """
    # Preserve the source precedence so equal text from different dimensions cannot merge.
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


def _downstream_metadata(
    selected: list[dict[str, Any]], prompts: list[Prompt]
) -> dict[str, Any]:
    """Build path-free identities for the shared live downstream comparison.

    Args:
        selected: Ordered source-selected contexts.
        prompts: Exact prompts resolved for those contexts.

    Returns:
        Metadata accepted by the current-source downstream comparator.
    """
    # Use context labels as the source fallback used by the cached RAVEL run.
    prompt_dicts = [vars(prompt) for prompt in prompts]
    labels = [_group_label(record) for record in prompt_dicts]
    return {
        "feature_id": FEATURE_ID,
        "context_indices": [int(row["index"]) for row in selected],
        "pair_indices": [int(row["pair_index"]) for row in selected],
        "attribute_labels": [str(row["attribute_label"]) for row in selected],
        "pair_roles": [str(row["pair_role"]) for row in selected],
        "murano_labels": labels,
        "context_labels": [
            [int(row["index"]), label]
            for row, label in zip(selected, labels, strict=True)
        ],
        "pair_labels": [],
    }


def run_audit(run_dir: Path, source_root: Path) -> dict[str, Any]:
    """Run source and Murano FEGA on shared model, SAE, prompts, and rows.

    Args:
        run_dir: Completed source task supplying selected prompts and its Gram.
        source_root: Current FEGA checkout used for direct downstream imports.

    Returns:
        One path-free compact report spanning effects and downstream decisions.
    """
    # Load the canonical source model bundle once using recorded run metadata.
    run_metadata = _load_json(run_dir / "run_metadata.json")
    resolved = run_metadata["resolved_config"]
    replay = ReplayContext.from_file(Path(resolved["reference_json"]))
    model, tokenizer, sae = load_model_and_sae(
        replay.eval_config,
        "cuda:0",
        sae_repo_id=str(resolved["sae_repo_id"]),
        sae_release_id=replay.sae_lens_release_id,
        sae_id_override=replay.sae_lens_id,
        download_location=Path(resolved["download_saes_dir"]),
        sae_cfg_dict=replay.sae_cfg_dict,
        sae_source="auto",
    )
    model.eval()
    sae.eval()

    # Build one source-owned padded batch and share its exact tensors with Murano.
    selected, prompts = _selected_prompts(run_dir)
    model_device = next(model.parameters()).device
    padded = build_padded_prompt_batch(
        prompts,
        device=model_device,
        pad_token_id=int(tokenizer.pad_token_id or 0),
        original_indices=[int(row["index"]) for row in selected],
        positioning_schema_version=POSITIONING_SCHEMA_VERSION,
    )
    tokens = {
        "input_ids": padded.input_ids,
        "attention_mask": padded.attention_mask,
        "position_ids": padded.position_ids,
    }
    layer = activation_collection.get_module(model, int(sae.cfg.hook_layer))
    output_embedding = model.get_output_embeddings()

    # Run both reconstruction baselines against the same live objects and inputs.
    source_x, source_z, source_readouts = run_sae_reconstruction(
        model,
        sae,
        padded.input_ids,
        padded.attention_mask,
        padded.target_positions,
        position_ids=padded.position_ids,
        readouts=["final_resid"],
    )
    murano_x, murano_z, murano_readout = run_reconstruction_readout_batch(
        model,
        layer,
        output_embedding,
        sae,
        tokens,
        padded.target_positions,
    )
    source_x_tensor = torch.stack(source_x)
    source_z_tensor = torch.stack(source_z)
    source_baseline = torch.stack(source_readouts["final_resid"])
    murano_baseline = _target_rows(murano_readout, padded.target_positions)

    # Zero feature 33760 from one canonical latent tensor in both implementations.
    ablation = AblationSpec(
        feature_ids=torch.full(
            (len(selected),),
            FEATURE_ID,
            device=source_z_tensor.device,
            dtype=torch.long,
        )
    )
    source_ablated_rows = run_ablation_readouts_batch(
        model,
        sae,
        padded.input_ids,
        padded.attention_mask,
        padded.target_positions,
        source_z_tensor,
        ablation,
        position_ids=padded.position_ids,
        requested_readouts=["final_resid"],
    )["final_resid"]
    murano_ablated_readout = run_reconstruction_readout_batch(
        model,
        layer,
        output_embedding,
        sae,
        tokens,
        padded.target_positions,
        feature_ids=[FEATURE_ID] * len(selected),
        z_batch=source_z_tensor,
    )
    source_ablated = torch.stack(source_ablated_rows).to(torch.float32)
    murano_ablated = _target_rows(murano_ablated_readout, padded.target_positions).to(
        torch.float32
    )
    source_baseline_float = source_baseline.detach().cpu().to(torch.float32)
    murano_baseline_float = murano_baseline.detach().cpu().to(torch.float32)
    murano_ablated_float = murano_ablated.detach().cpu()

    # Establish the exact intervention seam before any numerical normalization.
    exact = {
        "original_rows": _exact(source_x_tensor, murano_x),
        "latent_rows": _exact(source_z_tensor, murano_z),
        "baseline_readouts": _exact(source_baseline, murano_baseline),
        "ablated_readouts": _exact(source_ablated, murano_ablated),
        "effect_deltas": _exact(
            source_ablated - source_baseline_float,
            murano_ablated_float - murano_baseline_float,
        ),
    }

    # Bind the source Gram to the same live unembedding used for vocabulary coordinates.
    gram = torch.load(
        run_dir / "data_prep/gram_cache/gram.pt",
        map_location="cpu",
        weights_only=True,
    )
    gram_metadata = _load_json(run_dir / "data_prep/gram_cache/gram_meta.json")
    unembedding = canonical_unembedding(model)
    unembedding_matches = (
        unembedding_fingerprint(unembedding) == gram_metadata["unembedding_fingerprint"]
    )

    # Let each implementation derive retained directions and magnitudes independently.
    records = [
        EffectContextRecord(
            index=int(row["index"]),
            pair_index=int(row["pair_index"]),
            pair_role=str(row["pair_role"]),
            attribute_label=str(row["attribute_label"]),
            feature_activation=float(source_z_tensor[index, FEATURE_ID]),
            prompt=prompts[index],
        )
        for index, row in enumerate(selected)
    ]
    source_rows, source_mask = _build_final_resid_rows(
        records=records,
        bases=list(source_baseline),
        ablated=list(source_ablated),
        gram=gram,
        tau_zero=TAU_ZERO,
        stats=_empty_compute_stats(),
    )
    source_directions = torch.stack([row["direction"] for row in source_rows])
    source_magnitudes = torch.tensor(
        [row["magnitude"] for row in source_rows], dtype=torch.float32
    )
    murano_directions, murano_magnitudes, murano_mask = normalize_effect_rows(
        source_baseline_float,
        source_ablated,
        gram,
        tau_zero=TAU_ZERO,
    )
    normalization = {
        "retained_mask": {
            "status": ("match" if source_mask == murano_mask.tolist() else "mismatch"),
            "retained_rows": int(murano_mask.sum()),
        },
        "directions": _close(source_directions, murano_directions),
        "magnitudes": _close(source_magnitudes, murano_magnitudes),
    }

    # Compare vocabulary materialization, then share one live matrix downstream.
    source_coordinates = materialize_linear_coordinates(source_directions, unembedding)
    murano_coordinates = _materialize_vmf_coordinates(murano_directions, unembedding)
    coordinate_comparison = _close(source_coordinates, murano_coordinates)
    gram_kernel = (
        source_directions.to(torch.float64)
        @ gram.to(torch.float64)
        @ source_directions.to(torch.float64).T
    )
    downstream = compare_downstream(
        source_coordinates.numpy(),
        gram_kernel.numpy(),
        _downstream_metadata(selected, prompts),
        source_root,
        residual_rows=source_directions.numpy(),
        residual_gram=gram.numpy(),
        magnitudes=source_magnitudes.numpy(),
        source_config_path=run_dir / "config_used.yaml",
    )

    # Collapse only the section decisions; retain no model- or prompt-scale values.
    exact_match = all(item["status"] == "match" for item in exact.values())
    normalization_match = all(
        item["status"] == "match" for item in normalization.values()
    )
    matched = (
        exact_match
        and unembedding_matches
        and normalization_match
        and coordinate_comparison["status"] == "match"
        and downstream["status"] == "match"
    )
    return {
        "status": "match" if matched else "mismatch",
        "claim": "shared_live_feature_cloud_from_effect_calls_through_downstream",
        "configuration": {
            "feature_id": FEATURE_ID,
            "dtype": str(next(model.parameters()).dtype),
            "effect_batch_size": len(selected),
            "seed": 42,
        },
        "counts": {
            "selected_rows": len(selected),
            "retained_rows": len(source_rows),
            "vocabulary_dimensions": int(source_coordinates.shape[1]),
        },
        "effect_calls": exact,
        "unembedding_matches_gram": unembedding_matches,
        "normalization": normalization,
        "vocabulary_coordinates": coordinate_comparison,
        "downstream": downstream,
        "out_of_scope": [
            "RAVEL collection and selection regeneration",
            "paper-scale experiment matrix",
            "cross-hardware bitwise equality",
        ],
    }


def main() -> None:
    """Parse external roots, run the live oracle, and write compact JSON."""
    # Keep every external path in CLI arguments and out of the serialized report.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(args.run_dir.resolve(), args.source_root.resolve())
    args.result_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["status"] != "match":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
