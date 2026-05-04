"""I/O utilities — saving and loading Murano results."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING, Callable

import torch

from murano import __version__
from murano.artifacts import GenerationComparison, MetricResult, PromptBatch
from murano.logging import logger

if TYPE_CHECKING:
    from murano.model import MuranoModel


ArtifactSerializer = Callable[[str, Any, Path, Any, dict[str, Any]], None]


def save_steering(steering_result: Any, path: Path) -> None:
    """Save a SteeringResult to a .pt file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "direction_per_layer": steering_result.direction_per_layer,
            "separation_scores": steering_result.separation_scores,
            "best_layer": steering_result.best_layer,
        },
        path,
    )
    logger.info("Saved steering result to %s", path)


def load_steering(path: str | Path) -> Any:
    """Load a SteeringResult from a .pt file.

    Args:
        path: Path to the steering.pt file.

    Returns:
        SteeringResult ready for use with model.generate(ablate=...).
    """
    from murano.steps.train import SteeringResult

    data = torch.load(path, weights_only=False)
    return SteeringResult(
        direction_per_layer=data["direction_per_layer"],
        separation_scores=data["separation_scores"],
        best_layer=data["best_layer"],
    )


def save_generations(
    intervene_result: Any, path: Path, prompts: list[str] | None = None
) -> None:
    """Save an InterveneResult to a JSON file, paired per prompt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline_label = getattr(intervene_result, "baseline_label", "clean")
    modified_label = getattr(intervene_result, "modified_label", "modified")
    prompts = (
        prompts if prompts is not None else getattr(intervene_result, "prompts", None)
    )
    n = len(intervene_result.clean_generations)
    data = []
    for i in range(n):
        entry: dict[str, str] = {}
        if prompts and i < len(prompts):
            entry["prompt"] = prompts[i]
        entry[baseline_label] = intervene_result.clean_generations[i]
        entry[modified_label] = intervene_result.modified_generations[i]
        data.append(entry)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logger.info("Saved %d generation pairs to %s", n, path)


def save_eval(eval_result: Any, path: Path) -> None:
    """Save an EvalResult to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "metric_name": getattr(eval_result, "metric_name", "metric"),
        "baseline_label": getattr(eval_result, "baseline_label", "clean"),
        "modified_label": getattr(eval_result, "modified_label", "modified"),
        "baseline_score": eval_result.baseline_score,
        "modified_score": eval_result.modified_score,
        "baseline_scores": eval_result.baseline_scores,
        "modified_scores": eval_result.modified_scores,
        "metadata": getattr(eval_result, "metadata", {}),
    }
    if hasattr(eval_result, "clean_compliance"):
        data["clean_compliance"] = eval_result.clean_compliance
    if hasattr(eval_result, "ablated_compliance"):
        data["ablated_compliance"] = eval_result.ablated_compliance
    path.write_text(json.dumps(data, indent=2))
    logger.info("Saved eval result to %s", path)


def save_ablated_model(model: MuranoModel, save_dir: str | Path) -> Path:
    """Save the current (ablated) model weights in HF format.

    Saves model weights and tokenizer so the ablated model can be
    reloaded with transformers or MuranoModel.

    Args:
        model: MuranoModel with ablated weights (call after ablate_model_weights).
        save_dir: Directory to save the model.

    Returns:
        Path to the saved model directory.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # nnsight Envoy forwards .save_pretrained to the underlying HF model at runtime;
    # its static type signature doesn't reflect that.
    hf_model: Any = model._lm.model
    hf_model.save_pretrained(str(save_dir))
    model.tokenizer.save_pretrained(str(save_dir))
    logger.info("Saved ablated model to %s", save_dir)
    return save_dir


def save_probe(probe_result: Any, path: Path) -> None:
    """Save a ProbeResult to a JSON file (without classifiers)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "accuracy_per_layer": {
            str(k): v for k, v in probe_result.accuracy_per_layer.items()
        },
        "cv_scores": {str(k): v.tolist() for k, v in probe_result.cv_scores.items()},
        "best_layer": probe_result.best_layer,
        "label_names": probe_result.label_names,
    }
    path.write_text(json.dumps(data, indent=2))
    logger.info("Saved probe result to %s", path)


def save_prompts(prompt_batch: PromptBatch, path: Path) -> None:
    """Save a PromptBatch to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "source": prompt_batch.source,
        "prompts": prompt_batch.prompts,
        "raw_prompts": prompt_batch.raw_prompts,
        "metadata": prompt_batch.metadata,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logger.info("Saved %d prompts to %s", len(prompt_batch), path)


def save_metadata(metadata: dict, path: Path) -> None:
    """Save pipeline metadata to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata.setdefault("murano_version", __version__)
    metadata.setdefault("timestamp", datetime.now().isoformat())
    path.write_text(json.dumps(metadata, indent=2, default=str))


def _resolve_output_dir(output_dir: str | Path, run_name: str | None) -> Path:
    out = Path(output_dir)
    if run_name is not None:
        out = out / run_name
    return out


def _generation_prompts(dataset: Any) -> list[str] | None:
    raw_positive = getattr(dataset, "raw_positive", None)
    if raw_positive is not None:
        return raw_positive

    positive_texts = getattr(dataset, "positive_texts", None)
    if positive_texts is not None:
        return positive_texts

    raw_texts = getattr(dataset, "raw_texts", None)
    if raw_texts is not None:
        return raw_texts

    return getattr(dataset, "texts", None)


def register_artifact_serializer(
    registry: list[tuple[type, ArtifactSerializer]],
    artifact_type: type,
    serializer: ArtifactSerializer,
) -> None:
    registry.append((artifact_type, serializer))


def _serializer_registry() -> list[tuple[type, ArtifactSerializer]]:
    from murano.steps.probe import ProbeResult
    from murano.steps.train import SteeringResult

    registry: list[tuple[type, ArtifactSerializer]] = []

    def serialize_prompts(
        key: str,
        prompts: PromptBatch,
        out: Path,
        _results: Any,
        metadata: dict[str, Any],
    ) -> None:
        save_prompts(prompts, out / "prompts" / f"{key}.json")
        metadata["prompts"] = {
            "source": prompts.source,
            "n_prompts": len(prompts),
        }

    def serialize_steering(
        key: str, steering: Any, out: Path, _results: Any, metadata: dict[str, Any]
    ) -> None:
        filename = "steering.pt" if key == "steering" else f"{key}.pt"
        save_steering(steering, out / "direction" / filename)
        metadata[key] = {
            "best_layer": steering.best_layer,
            "separation_scores": steering.separation_scores,
        }

    def serialize_generations(
        key: str,
        comparison: GenerationComparison,
        out: Path,
        results: Any,
        metadata: dict[str, Any],
    ) -> None:
        prompts = comparison.prompts
        if prompts is None and "prompts" in results:
            prompt_batch = results["prompts"]
            prompts = (
                prompt_batch.raw_prompts
                if prompt_batch.raw_prompts is not None
                else prompt_batch.prompts
            )
        if prompts is None and "dataset" in results:
            prompts = _generation_prompts(results["dataset"])

        filename = "generations.json" if key == "intervene" else f"{key}.json"
        save_generations(comparison, out / "evaluation" / filename, prompts=prompts)
        metadata[key] = {
            "baseline_label": comparison.baseline_label,
            "modified_label": comparison.modified_label,
            "n_examples": len(comparison),
            "metadata": comparison.metadata,
        }

    def serialize_metric(
        key: str,
        metric: MetricResult,
        out: Path,
        _results: Any,
        metadata: dict[str, Any],
    ) -> None:
        filename = "eval.json" if key == "eval" else f"{key}.json"
        folder = "evaluation" if key == "eval" else "metrics"
        save_eval(metric, out / folder / filename)
        metadata[key] = {
            "metric_name": metric.metric_name,
            "baseline_label": metric.baseline_label,
            "modified_label": metric.modified_label,
            "baseline_score": metric.baseline_score,
            "modified_score": metric.modified_score,
            "metadata": metric.metadata,
        }
        if key == "eval":
            metadata["evaluation"] = {
                "metric_name": metric.metric_name,
                "baseline_score": metric.baseline_score,
                "modified_score": metric.modified_score,
            }

    def serialize_probe(
        key: str, probe: Any, out: Path, _results: Any, metadata: dict[str, Any]
    ) -> None:
        filename = "probe.json" if key == "probe" else f"{key}.json"
        save_probe(probe, out / "probe" / filename)
        metadata[key] = {
            "best_layer": probe.best_layer,
            "accuracy_per_layer": probe.accuracy_per_layer,
        }

    register_artifact_serializer(registry, PromptBatch, serialize_prompts)
    register_artifact_serializer(registry, SteeringResult, serialize_steering)
    register_artifact_serializer(registry, GenerationComparison, serialize_generations)
    register_artifact_serializer(registry, MetricResult, serialize_metric)
    register_artifact_serializer(registry, ProbeResult, serialize_probe)
    return registry


def _find_serializer(
    artifact: Any,
    registry: list[tuple[type, ArtifactSerializer]],
) -> ArtifactSerializer | None:
    for artifact_type, serializer in registry:
        if isinstance(artifact, artifact_type):
            return serializer
    return None


def save_results(
    results: Any,
    output_dir: str = "murano_outputs",
    model_id: str = "",
    run_name: str | None = None,
) -> Path:
    """Save all available results to organized subdirectories.

    Output structure:
        output_dir/
        ├── direction/           # steering vectors
        │   └── steering.pt
        ├── evaluation/          # generations + metrics
        │   ├── generations.json
        │   └── eval.json
        └── metadata.json

    Args:
        results: Results object from a pipeline run.
        output_dir: Base directory for outputs.
        model_id: HuggingFace model identifier (e.g. "meta-llama/Llama-3.2-1B-Instruct").
        run_name: Optional subdirectory name inside ``output_dir``.

    Returns:
        Path to the output directory.
    """
    out = _resolve_output_dir(output_dir, run_name)
    out.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "base_model": model_id,
        "pipeline_steps": list(results.keys()),
    }

    # Record dataset provenance
    if "dataset" in results:
        ds = results["dataset"]
        from murano.dataset import LabeledDataset

        if isinstance(ds, LabeledDataset):
            metadata["dataset"] = {
                "type": "labeled",
                "n_examples": len(ds.texts),
                "n_classes": len(set(ds.labels)),
                "label_names": ds.label_names,
            }
        else:
            metadata["dataset"] = {
                "type": "contrastive",
                "n_positive": len(ds.positive_texts),
                "n_negative": len(ds.negative_texts),
                "chat_templated": ds.raw_positive is not None,
            }

    registry = _serializer_registry()
    serialized_ids: set[int] = set()
    for key in results.keys():
        artifact = results[key]
        if id(artifact) in serialized_ids:
            continue
        serializer = _find_serializer(artifact, registry)
        if serializer is None:
            continue
        serializer(key, artifact, out, results, metadata)
        serialized_ids.add(id(artifact))

    save_metadata(metadata, out / "metadata.json")
    logger.info("Results saved to %s", out)
    return out
