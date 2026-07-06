"""Record step: captures activations via nnsight trace."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING, cast

from torch import Tensor, arange, cat, full_like, long, tensor  # pyright: ignore[reportPrivateImportUsage]

from murano import keys
from murano._proxy import unwrap_traced
from murano.logging import logger
from murano.nodes import Node, NodeDict
from murano.results import Results
from murano.steps.base import Step

if TYPE_CHECKING:
    from murano.backend import ModelBackend


@dataclass
class ActivationStore:
    """Stores per-layer/module activations for contrastive dataset splits.

    Attributes:
        positive: ``{Node: tensor}`` for positive texts, keyed by component
            address. Accepts shorthand on lookup (``store.positive[5]``).
        negative: ``{Node: tensor}`` for negative texts.
        position: Token-position selection applied at record time
            (``"last"`` / ``"first"`` / ``"mean"`` / int, or ``"none"`` to keep
            every position). Sets the tensor rank: reduced modes give
            ``[N, d_model]``; ``"none"`` gives ``[N, seq, d_model]``.
        per_head: Whether activations are split per attention head, which adds a
            trailing ``[..., n_heads, head_dim]`` pair of dims.
        positive_token_mask: ``[N, seq]`` valid-token mask for ``positive`` when
            ``position="none"``; ``None`` otherwise.
        negative_token_mask: ``[N, seq]`` valid-token mask for ``negative`` when
            ``position="none"``; ``None`` otherwise.
    """

    positive: dict[Node, Tensor]
    negative: dict[Node, Tensor]
    position: str | int = "last"
    per_head: bool = False
    positive_token_mask: Tensor | None = None
    negative_token_mask: Tensor | None = None

    def __post_init__(self) -> None:
        self.positive = NodeDict(self.positive)
        self.negative = NodeDict(self.negative)


@dataclass
class LabeledActivationStore:
    """Stores per-layer/module activations with associated per-example labels.

    Attributes:
        activations: ``{Node: tensor}`` token-position activations, keyed by
            component address. Accepts shorthand on lookup.
        labels: tensor [N] integer labels.
        position: Token-position selection applied at record time (see
            :class:`ActivationStore`).
        per_head: Whether activations are split per attention head.
        token_mask: ``[N, seq]`` valid-token mask when ``position="none"``;
            ``None`` otherwise.
    """

    activations: dict[Node, Tensor]
    labels: Tensor
    position: str | int = "last"
    per_head: bool = False
    token_mask: Tensor | None = None

    def __post_init__(self) -> None:
        self.activations = NodeDict(self.activations)


def _require_reduced_store(
    store: ActivationStore | LabeledActivationStore, step_name: str
) -> None:
    """Raise if the store holds full-position or per-head activations.

    Steps that consume reduced ``[N, d_model]`` activations call this to reject
    stores recorded with ``position="none"`` or ``per_head=True``: those carry
    extra dimensions that would silently produce wrong results. The caller
    identifies itself via ``step_name`` for the error message.

    Args:
        store: The activation store to validate.
        step_name: Name of the calling step, used in the error message.

    Raises:
        ValueError: If ``store.position == "none"`` or ``store.per_head``.
    """
    if store.position == "none" or store.per_head:
        raise ValueError(
            f"{step_name} requires a reduced activation store "
            f"(position != 'none' and per_head=False); got "
            f"position={store.position!r}, per_head={store.per_head}. "
            f"Record with a reduced position (e.g. 'last' or 'mean') and "
            f"per_head=False."
        )


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
    """Select per-sequence activations according to the requested position.

    ``output`` is ``[batch, seq, d_model]`` for whole-module activations, or
    ``[batch, seq, n_heads, head_dim]`` when recording per head. Reductions
    operate over the sequence dimension and preserve any trailing dims;
    ``position="none"`` returns the full positional tensor unchanged.
    """
    if output.dim() not in (3, 4):
        raise ValueError(
            f"Expected [batch, seq, ...] output of rank 3 or 4, "
            f"got {tuple(output.shape)}"
        )

    # Keep mask and indexing tensors on the activation's device.
    attention_mask = attention_mask.to(device=output.device)
    batch_indices = arange(output.shape[0], device=output.device)
    mask_bool = attention_mask.bool()
    seq_len = attention_mask.shape[1]

    if position == "none":
        return output

    if position == "mean":
        mask = mask_bool.to(output.dtype)
        while mask.dim() < output.dim():
            mask = mask.unsqueeze(-1)
        return (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    if position == "first":
        first_pos = mask_bool.int().argmax(dim=1)
        return output[batch_indices, first_pos]

    indices = (
        arange(seq_len, device=output.device).unsqueeze(0).expand_as(attention_mask)
    )
    masked_indices = indices.masked_fill(~mask_bool, -1)

    if position == "last":
        selected_positions = masked_indices.max(dim=1).values
        return output[batch_indices, selected_positions]

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
        return output[batch_indices, selected_positions]

    raise ValueError(
        "position must be one of 'last', 'first', 'mean', 'none', or an "
        "integer token index"
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
        position: Token position to record at. One of ``"last"``, ``"first"``,
            ``"mean"``, an integer token index, or ``"none"`` to keep every
            position (full-position recording).
        batch_size: Forward-pass batch size; must be ``>= 1``.
        per_head: If True, split attention activations per head, producing a
            trailing ``[..., n_heads, head_dim]`` pair of dims. Only valid for
            attention modules; the per-head signal is the input to the
            attention output projection (the concatenated head outputs).

    Raises:
        ValueError: If ``position`` or ``batch_size`` is invalid, or
            ``layers`` is a string other than ``"all"``.
        NotImplementedError: If ``per_head`` is set for a module whose
            architecture's attention output projection is not recognized.

    Note:
        Full-position (``position="none"``) and per-head recording accumulate
        larger tensors in CPU memory than reduced recording; combining them with
        ``layers="all"``, multiple modules, and large batches is memory-bound.
    """

    reads = [keys.DATASET]
    writes = [keys.RECORD]
    read_types = {}

    def __init__(
        self,
        model: ModelBackend,
        layers: list[int] | str = "all",
        modules: str | list[str] = "residual",
        position: str | int = "last",
        batch_size: int = 8,
        per_head: bool = False,
    ):
        if not (
            isinstance(position, int) or position in {"last", "first", "mean", "none"}
        ):
            raise ValueError(
                "position must be 'last', 'first', 'mean', 'none', or an "
                "integer token index"
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
        self.per_head = per_head

    def expected_read_types(self, results=None, available_types=None):
        """Return ``{"dataset": (MuranoDataset, LabeledDataset)}``."""
        from murano.dataset import LabeledDataset, MuranoDataset

        return {keys.DATASET: (MuranoDataset, LabeledDataset)}

    def expected_write_types(self, results=None, available_types=None):
        """Return the write type for ``record``, narrowed by the upstream dataset type.

        The output store type mirrors the input dataset type:
        ``MuranoDataset`` produces ``ActivationStore``; ``LabeledDataset``
        produces ``LabeledActivationStore``. Falls back to the union when
        the dataset type is not yet known.
        """
        from murano.dataset import LabeledDataset, MuranoDataset

        dataset_type = None
        if results is not None and keys.DATASET in results:
            dataset_type = type(results[keys.DATASET])
        elif available_types is not None:
            dataset_type = available_types.get(keys.DATASET)

        if dataset_type is None:
            return {keys.RECORD: (ActivationStore, LabeledActivationStore)}

        candidate_types = (
            dataset_type if isinstance(dataset_type, tuple) else (dataset_type,)
        )
        is_labeled = [issubclass(t, LabeledDataset) for t in candidate_types]
        is_contrastive = [
            issubclass(t, MuranoDataset) and not issubclass(t, LabeledDataset)
            for t in candidate_types
        ]

        if all(is_labeled):
            return {keys.RECORD: LabeledActivationStore}
        if all(is_contrastive):
            return {keys.RECORD: ActivationStore}
        return {keys.RECORD: (ActivationStore, LabeledActivationStore)}

    def __call__(self, results: Results) -> Results:
        from murano.dataset import LabeledDataset

        dataset = results[keys.DATASET]

        if isinstance(dataset, LabeledDataset):
            logger.info(
                "Recording: %d labeled texts, %d layers",
                len(dataset.texts),
                len(self.layers),
            )
            acts, mask = self._collect(dataset.texts)
            labels_tensor = tensor(dataset.labels, dtype=long)
            results[keys.RECORD] = LabeledActivationStore(
                activations=acts,
                labels=labels_tensor,
                position=self.position,
                per_head=self.per_head,
                token_mask=mask,
            )
        else:
            logger.info(
                "Recording: %d pos, %d neg texts, %d layers",
                len(dataset.positive_texts),
                len(dataset.negative_texts),
                len(self.layers),
            )
            pos_acts, pos_mask = (
                self._collect(dataset.positive_texts)
                if dataset.positive_texts
                else ({}, None)
            )
            neg_acts, neg_mask = (
                self._collect(dataset.negative_texts)
                if dataset.negative_texts
                else ({}, None)
            )
            results[keys.RECORD] = ActivationStore(
                positive=pos_acts,
                negative=neg_acts,
                position=self.position,
                per_head=self.per_head,
                positive_token_mask=pos_mask,
                negative_token_mask=neg_mask,
            )

        return results

    def _collect(self, texts: list[str]) -> tuple[dict[Node, Tensor], Tensor | None]:
        """Run texts through model and capture activations per layer/module.

        Runs a separate nnsight trace per module to avoid ``OutOfOrderError``
        when saving outputs from nested submodules (e.g. ``layer.output`` and
        ``layer.mlp.output``) in the same trace.

        Returns:
            A pair ``(activations, mask)``. ``activations`` is ``{Node: tensor}``
            keyed by canonical component address; the tensor rank follows
            ``position`` / ``per_head`` (see :class:`Record`). ``mask`` is the
            ``[N, seq]`` valid-token mask when ``position="none"`` (so positions
            stay interpretable across the concatenated batches), else ``None``.
        """
        keep_positions = self.position == "none"
        padding: bool | str = True
        max_length: int | None = None
        if keep_positions:
            # Pad every batch to one global width so full-position tensors from
            # different batches concatenate along the example dim.
            raw = self.model.tokenizer(
                texts, truncation=True, return_token_type_ids=False
            )
            raw_ids = cast("list[list[int]]", raw["input_ids"])
            padding = "max_length"
            max_length = max(len(ids) for ids in raw_ids)

        n_heads = self.model.n_heads

        if self.per_head:
            # Fail fast (outside any trace) if a module lacks a known
            # attention output projection, so the error propagates cleanly.
            for mod_str in self.modules:
                self.model.attn_out_proj(self.layers[0], mod_str)

        all_acts: dict[Node, list[Tensor]] = {}
        for layer in self.layers:
            for mod_str in self.modules:
                all_acts[Node(layer, mod_str)] = []
        masks: list[Tensor] = []

        # Run a separate trace per module to avoid nnsight conflicts when
        # saving outputs from nested submodules in the same trace.
        for mod_str in self.modules:
            for batch in _batched(texts, self.batch_size):
                tokens = self.model.tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    return_token_type_ids=False,
                    padding=padding,
                    max_length=max_length,
                )
                attention_mask = cast(Tensor, tokens["attention_mask"])
                # Masks are identical across modules; collect them once.
                if keep_positions and mod_str == self.modules[0]:
                    masks.append(attention_mask.detach().cpu())

                saved = {}
                with self.model.trace(tokens):
                    for layer in self.layers:
                        if self.per_head:
                            proj = self.model.attn_out_proj(layer, mod_str)
                            saved[layer] = proj.input.save()
                        else:
                            mod_proxy = self.model.resolve_module(layer, mod_str)
                            saved[layer] = mod_proxy.output.save()

                for layer in self.layers:
                    key = Node(layer, mod_str)
                    output = unwrap_traced(saved[layer])
                    if self.per_head:
                        # [batch, seq, n_heads * head_dim] -> per-head split.
                        if output.dim() != 3:
                            raise ValueError(
                                f"Per-head capture expects a rank-3 projection "
                                f"input [batch, seq, n_heads*head_dim] for module "
                                f"{mod_str!r}, got shape {tuple(output.shape)}."
                            )
                        b, s, d = output.shape
                        if d % n_heads != 0:
                            raise ValueError(
                                f"Cannot split module {mod_str!r} per head: "
                                f"projection input width {d} is not divisible by "
                                f"n_heads={n_heads}."
                            )
                        output = output.reshape(b, s, n_heads, d // n_heads)
                    selected_acts = _select_token_activations(
                        output=output,
                        attention_mask=attention_mask,
                        position=self.position,
                    )
                    all_acts[key].append(selected_acts.detach().cpu())

        acts = {key: cat(all_acts[key]) for key in all_acts}
        mask = cat(masks) if keep_positions else None
        return acts, mask
