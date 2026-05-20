"""SAE steps: encode residual activations and rank features by top-activating contexts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import Tensor  # pyright: ignore[reportPrivateImportUsage]

from murano.artifacts import PromptBatch
from murano.logging import logger
from murano.results import Results
from murano.steps.base import Step

if TYPE_CHECKING:
    from murano.model import MuranoModel


@dataclass
class SAEActivationStore:
    """Per-token SAE encodings of one layer's residual stream.

    Attributes:
        activations: Tensor [N, seq, n_features] of SAE encoder outputs.
        tokens: Tensor [N, seq] of input token ids.
        attention_mask: Tensor [N, seq] marking real (1) vs padding (0) positions.
        texts: Input texts paired by index with the N dimension.
        layer: Layer index where the SAE was applied.
        sae_repo: HuggingFace repo id of the SAE weights.
        n_features: SAE width (equals ``activations.shape[-1]``).
    """

    activations: Tensor
    tokens: Tensor
    attention_mask: Tensor
    texts: list[str]
    layer: int
    sae_repo: str
    n_features: int


@dataclass
class SAEFeatureExamples:
    """Top-K activating contexts per SAE feature, sorted descending by activation.

    Attributes:
        feat_ids: Features included.
        contexts: ``{feat_id: list[str]}`` the K context strings per feature.
        tokens: ``{feat_id: list[str]}`` the K triggering tokens per feature.
        act_vals: ``{feat_id: list[float]}`` the K activation values per feature.
        layer: Layer index where the SAE was applied.
        sae_repo: HuggingFace repo id of the SAE weights.
        k: Top-K cap per feature.
    """

    feat_ids: list[int]
    contexts: dict[int, list[str]]
    tokens: dict[int, list[str]]
    act_vals: dict[int, list[float]]
    layer: int
    sae_repo: str
    k: int


class SAEEncode(Step):
    """Encode residual-stream activations through an SAE loaded from HuggingFace.

    Runs an nnsight trace over the prompts at the target layer, applies
    the SAE encoder, and stores the per-token sparse features alongside
    the tokens and attention mask.

    Reads from results:
        results['prompts']: PromptBatch

    Writes to results:
        results['sae_record']: SAEActivationStore

    Args:
        model: MuranoModel to record from.
        sae_repo: HuggingFace repo id where SAE encoder weights live.
        layer: Layer index whose residual stream the SAE was trained on.

    Raises:
        ValueError: If ``layer`` is out of bounds for ``model``.
    """

    reads = ["prompts"]
    writes = ["sae_record"]
    read_types = {"prompts": PromptBatch}
    write_types = {"sae_record": SAEActivationStore}

    def __init__(self, model: MuranoModel, sae_repo: str, layer: int):
        if layer < 0 or layer >= model.n_layers:
            raise ValueError(
                f"layer {layer} out of bounds for model with {model.n_layers} layers"
            )
        self.model = model
        self.sae_repo = sae_repo
        self.layer = layer

    def __call__(self, results: Results) -> Results:
        raise NotImplementedError


class SAETopKContexts(Step):
    """Rank top-K activating contexts per SAE feature.

    For each feature, scans every ``(text, token)`` position in the
    ``SAEActivationStore`` and keeps the K positions with the largest
    activation. Padded tokens are excluded.

    Reads from results:
        results['sae_record']: SAEActivationStore

    Writes to results:
        results['feature_examples']: SAEFeatureExamples

    Args:
        model: MuranoModel, used to decode triggering tokens.
        k: Number of top contexts per feature; must be ``>= 1``.
        feat_ids: Specific features to rank. ``None`` ranks every feature.

    Raises:
        ValueError: If ``k < 1``, the SAE activations are not
            ``[N, seq, n_features]``-shaped, or any requested ``feat_id``
            is out of range.
    """

    reads = ["sae_record"]
    writes = ["feature_examples"]
    read_types = {"sae_record": SAEActivationStore}
    write_types = {"feature_examples": SAEFeatureExamples}

    def __init__(
        self,
        model: MuranoModel,
        k: int = 10,
        feat_ids: list[int] | None = None,
    ):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.model = model
        self.k = k
        self.feat_ids = list(feat_ids) if feat_ids is not None else None

    def __call__(self, results: Results) -> Results:
        store: SAEActivationStore = results["sae_record"]
        acts = store.activations
        if acts.dim() != 3:
            raise ValueError(
                f"Expected [N, seq, n_features] activations, got {tuple(acts.shape)}"
            )
        _, seq, n_features = acts.shape

        if self.feat_ids is None:
            feat_ids = list(range(n_features))
        else:
            feat_ids = self.feat_ids
            invalid = [f for f in feat_ids if f < 0 or f >= n_features]
            if invalid:
                raise ValueError(
                    f"feat_ids {invalid} out of range for SAE with "
                    f"{n_features} features"
                )

        flat_mask = store.attention_mask.bool().reshape(-1)
        k_used = min(self.k, int(flat_mask.sum().item()))

        contexts: dict[int, list[str]] = {}
        tokens: dict[int, list[str]] = {}
        act_vals: dict[int, list[float]] = {}
        for f in feat_ids:
            feat_flat = acts[..., f].reshape(-1).masked_fill(~flat_mask, float("-inf"))
            topk_vals, topk_idx = feat_flat.topk(k_used)

            contexts[f] = []
            tokens[f] = []
            act_vals[f] = []
            for v, flat_i in zip(topk_vals.tolist(), topk_idx.tolist()):
                n, j = divmod(int(flat_i), seq)
                contexts[f].append(store.texts[n])
                tokens[f].append(self.model.tokenizer.decode([int(store.tokens[n, j])]))
                act_vals[f].append(float(v))

        logger.info(
            "SAETopKContexts: %d features, k=%d (capped at %d), layer=%d",
            len(feat_ids),
            self.k,
            k_used,
            store.layer,
        )

        results["feature_examples"] = SAEFeatureExamples(
            feat_ids=feat_ids,
            contexts=contexts,
            tokens=tokens,
            act_vals=act_vals,
            layer=store.layer,
            sae_repo=store.sae_repo,
            k=self.k,
        )
        return results
