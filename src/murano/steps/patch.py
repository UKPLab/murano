"""Patch step: cross-run activation patching (interchange intervention).

Activation patching runs one input while injecting another input's activations at
chosen sites, then reads the logits, isolating how much those sites carry the
behaviour. This is the primitive under circuit discovery and interchange analysis.

The default direction is denoising: run the corrupt prompts and patch in the
clean activations at the target sites, so the metric recovers toward the clean
run. Compose with three metric runs and :class:`~murano.steps.metrics.RecoveredMetricStep`
to read the recovered fraction, and sweep ``targets`` (per layer, per head, or per
``positions``) to localize the behaviour. Swapping ``base_key`` and ``source_key``
flips the direction to noising (run clean, patch in corrupt).

``Patch`` is a thin preset over :class:`~murano.steps.ablate.Ablate`'s resample
engine: it captures the source batch's activations at the targets and blends them
into the base run, reusing the same per-head dispatch and ``positions`` selection.
The base and source prompts must be token-length-matched per pair so positions
align (the step raises if they are not), which a :class:`~murano.dataset.CleanCorruptDataset`
of equal-length pairs satisfies.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from murano import keys
from murano.nodes import AddressLike, NodeSet
from murano.steps.ablate import Ablate

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

    from murano.backend import ModelBackend


class Patch(Ablate):
    """Patch source-run activations into a base run and write the logits.

    Runs the base prompts (``base_key``) through one forward pass while replacing
    each target component's activation with the source prompts' (``source_key``)
    activation at the same site, then stores the resulting logits. With the
    defaults (base = corrupt, source = clean) this is denoising activation
    patching; a metric step scores how far the patched run recovers toward clean.

    Reads from results:
        results[base_key]: PromptBatch (default ``corrupt_prompts``), the run to
            patch into.
        results[source_key]: PromptBatch (default ``prompts``), the run supplying
            the replacement activations.

    Writes to results:
        results[logits_key]: Tensor [B, S, vocab] of patched logits.
        results[mask_key]: Tensor [B, S] marking real (1) vs padding (0) for the
            base batch, so a downstream metric step can locate the answer position.

    Args:
        model: Model backend to run.
        targets: Sites to patch: a :class:`~murano.nodes.NodeSet`, a single
            address, or an iterable of addresses. A whole-component target patches
            the module output; a head target (``Node(layer, "self_attn", head=h)``)
            patches that head. One call is a single mode (all whole-component or
            all per-head).
        base_key: Results key of the batch to patch into (default
            ``corrupt_prompts``).
        source_key: Results key of the batch supplying replacement activations
            (default ``prompts``, the clean side).
        positions: Token position(s) to patch, in the
            :func:`~murano.steps.metrics._answer_positions` form (an int or
            per-example sequence, negatives allowed); ``None`` patches every
            position.
        logits_key: Results key to write the patched logits under.
        mask_key: Results key to write the base attention mask under.

    Raises:
        ValueError: If ``targets`` is empty or mixes modes, or the base and source
            prompts are not token-length-matched per pair.
    """

    def __init__(
        self,
        model: ModelBackend,
        targets: NodeSet | AddressLike | Iterable[AddressLike],
        *,
        base_key: str = keys.CORRUPT_PROMPTS,
        source_key: str = keys.PROMPTS,
        positions: int | Sequence[int] | torch.Tensor | None = None,
        logits_key: str = keys.PATCHED_LOGITS,
        mask_key: str = keys.PATCHED_MASK,
    ):
        super().__init__(
            model,
            targets,
            method="resample",
            source_key=source_key,
            positions=positions,
            prompts_key=base_key,
            logits_key=logits_key,
            mask_key=mask_key,
        )
