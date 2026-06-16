"""SAE steps: encode residual activations and rank features by top-activating contexts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from torch import Tensor, no_grad  # pyright: ignore[reportPrivateImportUsage]

from murano import keys
from murano.artifacts import PromptBatch
from murano.logging import logger
from murano.nodes import Node
from murano.results import Results
from murano.steps.base import Step

if TYPE_CHECKING:
    from murano.backend import ModelBackend


_LAYER_RE = re.compile(r"(?:blocks|layer)[._](\d+)")


def _parse_layer(s: str | None) -> int | None:
    """Pull a layer index from strings like ``blocks.8.hook_resid_pre`` or ``layer_8/...``."""
    if s is None:
        return None
    m = _LAYER_RE.search(s)
    return int(m.group(1)) if m else None


def _unwrap(saved: Any) -> Tensor:
    val = saved.value if hasattr(saved, "value") else saved
    if isinstance(val, tuple):
        val = val[0]
    return val


def _resolve_hook(sae_model: SAEModel) -> tuple[int, str]:
    """Resolve (layer, hook_kind) from a loaded SAE's metadata.

    Falls back to parsing the layer index from the sae_id or hook_name when
    ``cfg.metadata.hook_layer`` is unset (some releases like gemma-scope do
    not populate it).
    """
    meta = sae_model._sae.cfg.metadata
    hook_name = meta.hook_name
    hook_layer = meta.hook_layer
    if hook_layer is None:
        hook_layer = _parse_layer(hook_name) or _parse_layer(sae_model.sae_id)
    if hook_layer is None:
        raise ValueError(
            f"Could not determine SAE layer (hook_layer is None, "
            f"hook_name={hook_name!r}, sae_id={sae_model.sae_id!r})."
        )

    if hook_name is None or "resid_post" in hook_name:
        hook_kind = "resid_post"
    elif "resid_pre" in hook_name:
        hook_kind = "resid_pre"
    elif "resid_mid" in hook_name:
        hook_kind = "resid_mid"
    elif "mlp_out" in hook_name:
        hook_kind = "mlp_out"
    elif "attn_out" in hook_name:
        hook_kind = "attn_out"
    elif "hook_z" in hook_name:
        raise NotImplementedError(
            "hook_z (per-head attention) SAEs need release-specific reshape "
            "handling; SAEEncode supports resid_pre, resid_post, resid_mid, "
            "mlp_out, attn_out."
        )
    else:
        raise NotImplementedError(
            f"SAE hook point {hook_name!r} not recognized by SAEEncode."
        )
    return int(hook_layer), hook_kind


def _capture_residual(
    model: ModelBackend, tokens: Any, hook_layer: int, hook_kind: str
) -> Tensor:
    """Trace ``tokens`` through ``model`` and return the residual the SAE expects."""
    saved: dict[str, Any] = {}
    with model.trace(tokens):
        layer = model.layer(hook_layer)
        if hook_kind == "resid_pre":
            saved["main"] = layer.input.save()
        elif hook_kind == "resid_post":
            saved["main"] = layer.output.save()
        elif hook_kind == "mlp_out":
            saved["main"] = model.resolve_module(hook_layer, "mlp").output.save()
        elif hook_kind == "attn_out":
            saved["main"] = model.resolve_module(hook_layer, "self_attn").output.save()
        else:  # resid_mid
            # resid_mid = resid_pre + attn_out (post-attention, pre-MLP).
            saved["main"] = layer.input.save()
            saved["extra"] = model.resolve_module(hook_layer, "self_attn").output.save()

    residual = _unwrap(saved["main"])
    if "extra" in saved:
        residual = residual + _unwrap(saved["extra"])
    return residual


@dataclass
class SAEActivationStore:
    """Per-token SAE encodings of one layer's residual stream.

    Attributes:
        activations: Tensor [N, seq, n_features] of SAE encoder outputs.
        tokens: Tensor [N, seq] of input token ids.
        attention_mask: Tensor [N, seq] marking real (1) vs padding (0) positions.
        texts: Input texts paired by index with the N dimension.
        hook: Component address (:class:`Node`) the SAE was applied at. Feature
            indices are a separate space and stay plain ints.
        release: HuggingFace SAE release identifier.
        sae_id: SAE id within the release.
        n_features: SAE width (equals ``activations.shape[-1]``).
    """

    activations: Tensor
    tokens: Tensor
    attention_mask: Tensor
    texts: list[str]
    hook: Node
    release: str
    sae_id: str
    n_features: int

    def __post_init__(self) -> None:
        self.hook = Node.coerce(self.hook)


@dataclass
class SAEFeatureExamples:
    """Top-K activating contexts per SAE feature, sorted descending by activation.

    Attributes:
        feat_ids: Features included.
        contexts: ``{feat_id: list[str]}`` the K context strings per feature.
        tokens: ``{feat_id: list[str]}`` the K triggering tokens per feature.
        act_vals: ``{feat_id: list[float]}`` the K activation values per feature.
        hook: Component address (:class:`Node`) the SAE was applied at.
        release: HuggingFace SAE release identifier.
        sae_id: SAE id within the release.
        k: Top-K cap per feature.
    """

    feat_ids: list[int]
    contexts: dict[int, list[str]]
    tokens: dict[int, list[str]]
    act_vals: dict[int, list[float]]
    hook: Node
    release: str
    sae_id: str
    k: int

    def __post_init__(self) -> None:
        self.hook = Node.coerce(self.hook)


class SAEModel:
    """Loaded SAE encoder, applied to a model's residual stream.

    Weights are pulled lazily from HuggingFace via ``sae-lens`` on first
    use. Sharing one instance across pipelines avoids repeated loads.

    Requires the ``[sae]`` extra: ``pip install -e ".[sae]"``.

    Attributes:
        release: sae-lens release identifier.
        sae_id: SAE id within the release.
        device: Torch device the encoder runs on.
    """

    def __init__(self, release: str, sae_id: str, device: str = "cpu"):
        self.release = release
        self.sae_id = sae_id
        self.device = device
        self._sae: Any = None

    def _ensure_loaded(self) -> None:
        if self._sae is None:
            # sae-lens is the optional [sae] extra; absent in the base install.
            from sae_lens import SAE  # pyright: ignore[reportMissingImports]

            self._sae = SAE.from_pretrained(
                release=self.release,
                sae_id=self.sae_id,
                device=self.device,
            )

    @property
    def n_features(self) -> int:
        """SAE width (encoder output dimension)."""
        self._ensure_loaded()
        return int(self._sae.cfg.d_sae)

    def encode(self, residual: Tensor) -> Tensor:
        """Encode residual ``[N, seq, d_model]`` to SAE codes ``[N, seq, n_features]``."""
        self._ensure_loaded()
        return self._sae.encode(residual.to(self.device))


class SAEEncode(Step):
    """Encode residual-stream activations through an SAE loaded from HuggingFace.

    Constructs an ``SAEModel`` internally from ``release`` + ``sae_id`` and
    auto-detects the target layer and hook point from the SAE's own config,
    so the same step works against any sae-lens release without the caller
    having to know its training-time conventions. The loaded SAE is reachable
    via ``self.sae_model`` for reuse or inspection.

    Supported hook points: ``resid_pre``, ``resid_post``, ``resid_mid``,
    ``mlp_out``, ``attn_out``. Per-head ``hook_z`` SAEs are not handled
    because reshape semantics vary per release.

    Reads from results:
        results['prompts']: PromptBatch

    Writes to results:
        results['sae_record']: SAEActivationStore

    Args:
        model: MuranoModel to record from.
        release: HuggingFace SAE release identifier.
        sae_id: SAE id within the release.
        max_length: Truncate prompts to this many tokens before tracing.
            ``None`` disables truncation (default; suitable for short prompts).

    Raises:
        ValueError: If the SAE's hook layer is out of bounds for ``model``,
            or the SAE layer cannot be determined from the config or sae_id.
        NotImplementedError: If the SAE was trained on an unsupported hook
            point (e.g. ``hook_z``).
    """

    reads = [keys.PROMPTS]
    writes = [keys.SAE_RECORD]
    read_types = {keys.PROMPTS: PromptBatch}
    write_types = {keys.SAE_RECORD: SAEActivationStore}

    def __init__(
        self,
        model: ModelBackend,
        release: str,
        sae_id: str,
        max_length: int | None = None,
    ):
        self.model = model
        self.max_length = max_length
        self.sae_model = SAEModel(release=release, sae_id=sae_id)

    def __call__(self, results: Results) -> Results:
        prompts = results[keys.PROMPTS].prompts

        self.sae_model._ensure_loaded()
        hook_layer, hook_kind = _resolve_hook(self.sae_model)
        if hook_layer < 0 or hook_layer >= self.model.n_layers:
            raise ValueError(
                f"SAE trained on layer {hook_layer}, but model has only "
                f"{self.model.n_layers} layers."
            )

        tokens = self.model.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            return_token_type_ids=False,
        )
        residual = _capture_residual(self.model, tokens, hook_layer, hook_kind)

        with no_grad():
            sae_acts = self.sae_model.encode(residual)

        results[keys.SAE_RECORD] = SAEActivationStore(
            activations=sae_acts.detach().cpu(),
            tokens=cast(Tensor, tokens["input_ids"]).detach().cpu(),
            attention_mask=cast(Tensor, tokens["attention_mask"]).detach().cpu(),
            texts=list(prompts),
            hook=Node(hook_layer, hook_kind),
            release=self.sae_model.release,
            sae_id=self.sae_model.sae_id,
            n_features=sae_acts.shape[-1],
        )
        return results


class SAETopActivations(Step):
    """Rank top-K activating contexts per SAE feature.

    For each feature, scans every ``(text, token)`` position in the
    ``SAEActivationStore`` and keeps the K positions with the largest
    activation. Padded tokens are excluded. BOS-token positions are
    excluded by default, since residual-stream SAEs tend to develop strong
    BOS-anchored features that dominate the top-K and crowd out
    content-bearing features.

    Reads from results:
        results['sae_record']: SAEActivationStore

    Writes to results:
        results['feature_examples']: SAEFeatureExamples

    Args:
        model: MuranoModel, used to decode triggering tokens.
        k: Number of top contexts per feature; must be ``>= 1``.
        feat_ids: Specific features to rank. ``None`` ranks every feature.
        skip_bos: If True, mask out positions whose token id equals the
            tokenizer's ``bos_token_id`` before ranking. Has no effect when
            the tokenizer has no BOS token.

    Raises:
        ValueError: If ``k < 1``, the SAE activations are not
            ``[N, seq, n_features]``-shaped, or any requested ``feat_id``
            is out of range.
    """

    reads = [keys.SAE_RECORD]
    writes = [keys.FEATURE_EXAMPLES]
    read_types = {keys.SAE_RECORD: SAEActivationStore}
    write_types = {keys.FEATURE_EXAMPLES: SAEFeatureExamples}

    def __init__(
        self,
        model: ModelBackend,
        k: int = 10,
        feat_ids: list[int] | None = None,
        skip_bos: bool = True,
    ):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.model = model
        self.k = k
        self.feat_ids = list(feat_ids) if feat_ids is not None else None
        self.skip_bos = skip_bos

    def __call__(self, results: Results) -> Results:
        store: SAEActivationStore = results[keys.SAE_RECORD]
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
        if self.skip_bos:
            bos_id = self.model.tokenizer.bos_token_id
            if bos_id is not None:
                flat_mask = flat_mask & (store.tokens.reshape(-1) != bos_id)
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
            "SAETopActivations: %d features, k=%d (capped at %d), hook=%s",
            len(feat_ids),
            self.k,
            k_used,
            store.hook,
        )

        results[keys.FEATURE_EXAMPLES] = SAEFeatureExamples(
            feat_ids=feat_ids,
            contexts=contexts,
            tokens=tokens,
            act_vals=act_vals,
            hook=store.hook,
            release=store.release,
            sae_id=store.sae_id,
            k=self.k,
        )
        return results
