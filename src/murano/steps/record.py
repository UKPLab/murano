"""Record step: captures activations via nnsight trace."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING, cast

from torch import Tensor, arange, cat, full_like, long, tensor  # pyright: ignore[reportPrivateImportUsage]

from murano.logging import logger
from murano.results import Results
from murano.steps.base import Step

if TYPE_CHECKING:
    from murano.model import MuranoModel


# Key type for activation dictionaries.
# When a single module is targeted, keys are plain layer indices (int).
# When multiple modules are targeted, keys are (layer_idx, module_name) tuples.
ActivationKey = int | tuple[int, str]


@dataclass
class ActivationStore:
    """Stores per-layer/module activations for contrastive dataset splits.

    Attributes:
        positive: {key: tensor [N, d_model]} for positive texts.
                  Keys are ``int`` (layer index) when a single module is
                  targeted, or ``(int, str)`` (layer, module_name) when
                  multiple modules are targeted.
        negative: {key: tensor [N, d_model]} for negative texts.
    """

    positive: dict[ActivationKey, Tensor]
    negative: dict[ActivationKey, Tensor]


@dataclass
class LabeledActivationStore:
    """Stores per-layer/module activations with associated labels for probing.

    Attributes:
        activations: {key: tensor [N, d_model]} token-position activations.
                     Keys are ``int`` (layer index) when a single module is
                     targeted, or ``(int, str)`` (layer, module_name) when
                     multiple modules are targeted.
        labels: tensor [N] integer labels.
    """

    activations: dict[ActivationKey, Tensor]
    labels: Tensor


def _batched(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch


def _rank_positions(attention_mask: Tensor) -> Tensor:
    """Return token ranks within each sequence, ignoring padding."""
    return attention_mask.cumsum(dim=1) - 1


def _select_token_activations(
    output: Tensor,
    attention_mask: Tensor,
    position: str | int,
) -> Tensor:
    """Select per-sequence activations according to the requested position."""
    if output.dim() != 3:
        raise ValueError(
            f"Expected [batch, seq, d_model] output, got {tuple(output.shape)}"
        )

    batch_indices = arange(output.shape[0], device=output.device)
    mask_bool = attention_mask.bool()
    seq_len = attention_mask.shape[1]

    if position == "mean":
        mask = attention_mask.unsqueeze(-1).to(output.dtype)
        return (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    if position == "first":
        first_pos = mask_bool.int().argmax(dim=1)
        return output[batch_indices, first_pos, :]

    indices = (
        arange(seq_len, device=output.device).unsqueeze(0).expand_as(attention_mask)
    )
    masked_indices = indices.masked_fill(~mask_bool, -1)

    if position == "last":
        selected_positions = masked_indices.max(dim=1).values
        return output[batch_indices, selected_positions, :]

    if isinstance(position, int):
        lengths = attention_mask.sum(dim=1)
        target_rank = full_like(lengths, position)
        if position < 0:
            target_rank = lengths + position
        invalid = (target_rank < 0) | (target_rank >= lengths)
        if invalid.any():
            raise ValueError(
                f"Requested position {position} is out of bounds for at least one "
                f"sequence with lengths {lengths.tolist()}."
            )
        ranks = _rank_positions(attention_mask)
        target_mask = mask_bool & (ranks == target_rank.unsqueeze(1))
        selected_positions = target_mask.int().argmax(dim=1)
        return output[batch_indices, selected_positions, :]

    raise ValueError(
        "position must be one of 'last', 'first', 'mean', or an integer token index"
    )


class Record(Step):
    """Capture residual-stream activations via nnsight.

    Reads from results:
        results['dataset']: MuranoDataset or LabeledDataset

    Writes to results:
        results['record']: ActivationStore or LabeledActivationStore

    Args:
        model: Wrapped model to record from.
        layers: Layer indices to record, or ``"all"`` for every layer.
        position: Token position to record at. One of ``"last"``,
            ``"first"``, ``"mean"``, or an integer token index.
        batch_size: Forward-pass batch size; must be ``>= 1``.

    Raises:
        ValueError: If ``position`` or ``batch_size`` is invalid, or
            ``layers`` is a string other than ``"all"``.
    """

    reads = ["dataset"]
    writes = ["record"]
    read_types = {}

    def __init__(
        self,
        model: MuranoModel,
        layers: list[int] | str = "all",
        modules: str | list[str] = "residual",
        position: str | int = "last",
        batch_size: int = 8,
    ):
        if not (isinstance(position, int) or position in {"last", "first", "mean"}):
            raise ValueError(
                "position must be 'last', 'first', 'mean', or an integer token index"
            )
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.model = model
        if isinstance(layers, str):
            if layers != "all":
                raise ValueError(f"layers as string must be 'all', got {layers!r}")
            self.layers: list[int] = list(range(model.n_layers))
        else:
            self.layers = list(layers)
        self.modules: list[str] = [modules] if isinstance(modules, str) else modules
        self.position = position
        self.batch_size = batch_size

    def expected_read_types(self, results=None, available_types=None):
        """Return ``{"dataset": (MuranoDataset, LabeledDataset)}``."""
        from murano.dataset import LabeledDataset, MuranoDataset

        return {"dataset": (MuranoDataset, LabeledDataset)}

    def expected_write_types(self, results=None, available_types=None):
        """Return the write type for ``record``, narrowed by the upstream dataset type.

        The output store type mirrors the input dataset type:
        ``MuranoDataset`` produces ``ActivationStore``; ``LabeledDataset``
        produces ``LabeledActivationStore``. Falls back to the union when
        the dataset type is not yet known.
        """
        from murano.dataset import LabeledDataset, MuranoDataset

        dataset_type = None
        if results is not None and "dataset" in results:
            dataset_type = type(results["dataset"])
        elif available_types is not None:
            dataset_type = available_types.get("dataset")

        if dataset_type is None:
            return {"record": (ActivationStore, LabeledActivationStore)}

        candidate_types = (
            dataset_type if isinstance(dataset_type, tuple) else (dataset_type,)
        )
        is_labeled = [issubclass(t, LabeledDataset) for t in candidate_types]
        is_contrastive = [
            issubclass(t, MuranoDataset) and not issubclass(t, LabeledDataset)
            for t in candidate_types
        ]

        if all(is_labeled):
            return {"record": LabeledActivationStore}
        if all(is_contrastive):
            return {"record": ActivationStore}
        return {"record": (ActivationStore, LabeledActivationStore)}

    def __call__(self, results: Results) -> Results:
        from murano.dataset import LabeledDataset

        dataset = results["dataset"]

        if isinstance(dataset, LabeledDataset):
            logger.info(
                "Recording: %d labeled texts, %d layers",
                len(dataset.texts),
                len(self.layers),
            )
            acts = self._collect(dataset.texts)
            labels_tensor = tensor(dataset.labels, dtype=long)
            results["record"] = LabeledActivationStore(
                activations=acts,
                labels=labels_tensor,
            )
        else:
            logger.info(
                "Recording: %d pos, %d neg texts, %d layers",
                len(dataset.positive_texts),
                len(dataset.negative_texts),
                len(self.layers),
            )
            pos_acts = (
                self._collect(dataset.positive_texts) if dataset.positive_texts else {}
            )
            neg_acts = (
                self._collect(dataset.negative_texts) if dataset.negative_texts else {}
            )
            results["record"] = ActivationStore(positive=pos_acts, negative=neg_acts)

        return results

    def _collect(self, texts: list[str]) -> dict[ActivationKey, Tensor]:
        """Run texts through model and capture activations per layer/module.

        Runs a separate nnsight trace per module to avoid ``OutOfOrderError``
        when saving outputs from nested submodules (e.g. ``layer.output`` and
        ``layer.mlp.output``) in the same trace.

        Returns:
            {key: tensor [N, d_model]} with selected-token activations.
            Keys are ``int`` (layer index) when a single module is targeted,
            or ``(int, str)`` (layer, module_name) when multiple modules
            are targeted.
        """
        all_acts: dict[ActivationKey, list[Tensor]] = {}
        for layer in self.layers:
            for mod_str in self.modules:
                key: ActivationKey = (
                    layer if len(self.modules) == 1 else (layer, mod_str)
                )
                all_acts[key] = []

        # Run a separate trace per module to avoid nnsight conflicts when
        # saving outputs from nested submodules in the same trace.
        for mod_str in self.modules:
            for batch in _batched(texts, self.batch_size):
                tokens = self.model.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    return_token_type_ids=False,
                )
                attention_mask = cast(Tensor, tokens["attention_mask"])

                saved = {}
                with self.model._lm.trace(tokens):
                    for layer in self.layers:
                        mod_proxy = self.model._resolve_module(
                            self.model.layer(layer), mod_str
                        )
                        saved[layer] = mod_proxy.output.save()

                for layer in self.layers:
                    key: ActivationKey = (
                        layer if len(self.modules) == 1 else (layer, mod_str)
                    )
                    output = (
                        saved[layer].value
                        if hasattr(saved[layer], "value")
                        else saved[layer]
                    )
                    # Some transformers versions return (hidden_states, ...) tuples
                    # from decoder layers; older (<5.0) Llama is one. Unwrap here.
                    if isinstance(output, tuple):
                        output = output[0]
                    # output: [batch, seq, d_model]
                    selected_acts = _select_token_activations(
                        output=output,
                        attention_mask=attention_mask,
                        position=self.position,
                    )
                    all_acts[key].append(selected_acts.detach().cpu())

        return {key: cat(all_acts[key]) for key in all_acts}
