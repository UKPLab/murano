"""I/O utilities: saving and loading Murano results."""

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
    """Save a SteeringResult to a .pt file.

    Args:
        steering_result: SteeringResult to serialize.
        path: Output path. Parent directory is created if missing.
    """
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
    """Save an InterveneResult to a JSON file, paired per prompt.

    Args:
        intervene_result: GenerationComparison-compatible artifact with
            ``clean_generations`` and ``modified_generations`` lists.
        path: Output path for the JSON file. Parent directory is created
            if missing.
        prompts: Prompts to pair each entry with. Falls back to
            ``intervene_result.prompts`` when None.
    """
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
    """Save an EvalResult to a JSON file.

    Args:
        eval_result: MetricResult or EvalResult to serialize.
        path: Output path. Parent directory is created if missing.
    """
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
    """Save a ProbeResult to a JSON file (without classifiers).

    Fitted classifier objects are not serialized; only per-layer accuracy,
    CV scores, and metadata are persisted.

    Args:
        probe_result: ProbeResult to serialize.
        path: Output path. Parent directory is created if missing.
    """
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


def save_logit_lens(logit_lens_result: Any, path: Path) -> None:
    """Save a LogitLensResult to a .pt file.

    Persists every field on the dataclass. Per-layer probability tensors,
    argmax tokens, decoded words, input words, attention mask, and layer
    indices, so the result can be reloaded for downstream plotting or
    analysis without re-running the trace.

    Args:
        logit_lens_result: LogitLensResult to serialize.
        path: Output path. Parent directory is created if missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "all_probs": logit_lens_result.all_probs,
            "max_probs": logit_lens_result.max_probs,
            "predicted_tokens": logit_lens_result.predicted_tokens,
            "predicted_words": logit_lens_result.predicted_words,
            "input_words": logit_lens_result.input_words,
            "attention_mask": logit_lens_result.attention_mask,
            "layer_indices": logit_lens_result.layer_indices,
        },
        path,
    )
    logger.info("Saved logit lens result to %s", path)


def load_logit_lens(path: str | Path) -> Any:
    """Load a LogitLensResult from a .pt file.

    Args:
        path: Path to the logit_lens.pt file.

    Returns:
        LogitLensResult reconstructed from the saved tensors and lists,
        ready for downstream plotting or analysis.
    """
    from murano.steps.logit_lens import LogitLensResult

    data = torch.load(path, weights_only=False)
    return LogitLensResult(
        all_probs=data["all_probs"],
        max_probs=data["max_probs"],
        predicted_tokens=data["predicted_tokens"],
        predicted_words=data["predicted_words"],
        input_words=data["input_words"],
        attention_mask=data["attention_mask"],
        layer_indices=data["layer_indices"],
    )


def save_activation_store(activation_store: Any, path: Path) -> None:
    """Save an ActivationStore to a .pt file.

    Interim on-disk format; the layout may change.

    Args:
        activation_store: ActivationStore to serialize.
        path: Output path. Parent directory is created if missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "positive": activation_store.positive,
            "negative": activation_store.negative,
        },
        path,
    )
    logger.info("Saved activation store to %s", path)


def load_activation_store(path: str | Path) -> Any:
    """Load an ActivationStore from a .pt file.

    Args:
        path: Path to the saved activation-store .pt file.

    Returns:
        ActivationStore with ``positive`` and ``negative`` activation dicts.
    """
    from murano.steps.record import ActivationStore

    data = torch.load(path, weights_only=False)
    return ActivationStore(positive=data["positive"], negative=data["negative"])


def save_labeled_activation_store(labeled_store: Any, path: Path) -> None:
    """Save a LabeledActivationStore to a .pt file.

    Interim on-disk format; the layout may change.

    Args:
        labeled_store: LabeledActivationStore to serialize.
        path: Output path. Parent directory is created if missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "activations": labeled_store.activations,
            "labels": labeled_store.labels,
        },
        path,
    )
    logger.info("Saved labeled activation store to %s", path)


def load_labeled_activation_store(path: str | Path) -> Any:
    """Load a LabeledActivationStore from a .pt file.

    Args:
        path: Path to the saved labeled-activation-store .pt file.

    Returns:
        LabeledActivationStore with ``activations`` dict and ``labels`` tensor.
    """
    from murano.steps.record import LabeledActivationStore

    data = torch.load(path, weights_only=False)
    return LabeledActivationStore(
        activations=data["activations"], labels=data["labels"]
    )


def save_prompts(prompt_batch: PromptBatch, path: Path) -> None:
    """Save a PromptBatch to JSON.

    Args:
        prompt_batch: Prompts plus optional raw versions and metadata.
        path: Output path. Parent directory is created if missing.
    """
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
    """Save pipeline metadata to a JSON file.

    ``murano_version`` and ``timestamp`` keys are filled in if absent.

    Args:
        metadata: Metadata dict to serialize. Mutated in place to add the
            two default keys above.
        path: Output path for the JSON file.
    """
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
    """Append a (type, serializer) pair to a serializer registry.

    Used by the Save step to dispatch each result-dict value to the right
    serializer based on its type.

    Args:
        registry: Mutable list of (type, serializer) pairs.
        artifact_type: Class to dispatch on. ``isinstance`` is used at lookup
            time, so subclasses match.
        serializer: Callable that writes the artifact to disk. Signature:
            ``(key, artifact, out_dir, results, metadata) -> None``.
    """
    registry.append((artifact_type, serializer))


def _serializer_registry() -> list[tuple[type, ArtifactSerializer]]:
    from murano.steps.logit_lens import LogitLensResult
    from murano.steps.probe import ProbeResult
    from murano.steps.record import ActivationStore, LabeledActivationStore
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

    def serialize_logit_lens(
        key: str,
        logit_lens: Any,
        out: Path,
        _results: Any,
        metadata: dict[str, Any],
    ) -> None:
        filename = "logit_lens.pt" if key == "logit_lens" else f"{key}.pt"
        save_logit_lens(logit_lens, out / "logit_lens" / filename)
        metadata[key] = {
            "layer_indices": logit_lens.layer_indices,
            "n_layers": logit_lens.all_probs.shape[0],
            "n_inputs": logit_lens.all_probs.shape[1],
        }

    def serialize_activation_store(
        key: str,
        store: Any,
        out: Path,
        _results: Any,
        metadata: dict[str, Any],
    ) -> None:
        filename = "record.pt" if key == "record" else f"{key}.pt"
        save_activation_store(store, out / "activations" / filename)
        pos = next(iter(store.positive.values()), None)
        neg = next(iter(store.negative.values()), None)
        metadata[key] = {
            "kind": "contrastive",
            "keys": [str(k) for k in store.positive],
            "n_positive": pos.shape[0] if pos is not None else 0,
            "n_negative": neg.shape[0] if neg is not None else 0,
        }

    def serialize_labeled_activation_store(
        key: str,
        store: Any,
        out: Path,
        _results: Any,
        metadata: dict[str, Any],
    ) -> None:
        filename = "record.pt" if key == "record" else f"{key}.pt"
        save_labeled_activation_store(store, out / "activations" / filename)
        metadata[key] = {
            "kind": "labeled",
            "keys": [str(k) for k in store.activations],
            "n_examples": store.labels.shape[0],
        }

    register_artifact_serializer(registry, PromptBatch, serialize_prompts)
    register_artifact_serializer(registry, SteeringResult, serialize_steering)
    register_artifact_serializer(registry, GenerationComparison, serialize_generations)
    register_artifact_serializer(registry, MetricResult, serialize_metric)
    register_artifact_serializer(registry, ProbeResult, serialize_probe)
    register_artifact_serializer(registry, LogitLensResult, serialize_logit_lens)
    register_artifact_serializer(registry, ActivationStore, serialize_activation_store)
    register_artifact_serializer(
        registry, LabeledActivationStore, serialize_labeled_activation_store
    )
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
        ├── logit_lens/          # logit-lens probabilities + decoded words
        │   └── logit_lens.pt
        ├── activations/         # recorded activation stores (transitional format)
        │   └── record.pt
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
        "result_keys": list(results.keys()),
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
            # "dataset" is recorded as provenance above; "output_dir" and other
            # transient values aren't artifacts. Warn on anything else so a real
            # artifact with no serializer isn't dropped silently.
            if key not in metadata and not isinstance(
                artifact, (str, bytes, int, float, bool, Path, type(None))
            ):
                logger.warning(
                    "No serializer registered for results[%r] (type %s); it was "
                    "not saved. Register one in _serializer_registry().",
                    key,
                    type(artifact).__name__,
                )
            continue
        serializer(key, artifact, out, results, metadata)
        serialized_ids.add(id(artifact))

    save_metadata(metadata, out / "metadata.json")
    logger.info("Results saved to %s", out)
    return out
