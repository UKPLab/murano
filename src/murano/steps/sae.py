"""Sparse-Autoencoder steps for Murano.

This module lets you encode prompts through a pre-trained SAE loaded from
HuggingFace via `sae-lens` and then read the resulting features two ways.

Building blocks
---------------

- :class:`SAEModel` wraps the underlying `sae-lens` SAE with lazy loading.
  Its one user-facing method is ``feature_direction(feature_id)`` (the SAE's
  decoder row for that feature, used for steering and labeling); ``encode``
  and ``n_features`` are plumbing that :class:`SAEEncode` drives internally.
- :func:`top_sae_features_per_prompt` is a plain helper that picks each
  prompt's top firing feature(s), either at the last token (``reduce="last"``)
  or by mean over the sentence's content tokens (``reduce="mean"``), filtering
  out the BOS-anchored sink and broad high-frequency features.
- :func:`sae_steer` is a plain helper that builds an
  :class:`~murano.steps.intervene.Intervene` step which adds a single feature's
  decoder direction to the residual stream at the exact site the SAE was
  trained on, so steering works for any SAE hook point without the caller
  picking a layer or module. It reuses the existing steering machinery
  (:func:`~murano.steps.intervene.steer_direction`) rather than adding a new
  one.

Pipeline steps
--------------

- :class:`SAEEncode` runs prompts through the SAE and writes
  :class:`SAEActivationStore` (``[N, seq, n_features]`` activations).
- :class:`SAETopActivations` is the **input-side** interpretation:
  given feature ids, it returns the prompt contexts where each feature
  fires hardest as :class:`SAEFeatureExamples`.
- :class:`SAEFeatureLabel` is the **output-side** interpretation: given
  feature ids, it projects each feature's decoder direction through the
  model's final norm + unembedding via ``model.project_on_vocab`` (the same
  ``ln_final + lm_head`` projection :class:`~murano.steps.logit_lens.LogitLens`
  applies to per-layer residuals) and returns the top tokens each feature
  promotes in next-token space, as :class:`SAEFeatureLabels`.

Typical flow::

    results = Pipeline([
        LoadPrompts(prompts),
        SAEEncode(model, release=..., sae_id=...),     # -> sae_record
    ]).run()
    feats = top_sae_features_per_prompt(results["sae_record"], n=1)
    feat_ids = sorted({f for prompt in feats for f in prompt})

    results = Pipeline([
        SAEFeatureLabel(model, feat_ids=feat_ids, k_tokens=5),
    ]).run(results)                                     # -> feature_labels
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from torch import (  # pyright: ignore[reportPrivateImportUsage]
    Tensor,
    no_grad,
    stack,
    tensor,  # pyright: ignore[reportPrivateImportUsage]
)

from murano import keys
from murano._optional import require_optional
from murano._proxy import unwrap_traced
from murano.artifacts import PromptBatch
from murano.logging import logger
from murano.nodes import Node
from murano.results import Results
from murano.steps.base import Step

if TYPE_CHECKING:
    from murano.backend import ModelBackend
    from murano.steps.intervene import Intervene


_LAYER_RE = re.compile(r"(?:blocks|layer)[._](\d+)")


def _parse_layer(s: str | None) -> int | None:
    """Pull a layer index from strings like ``blocks.8.hook_resid_pre`` or ``layer_8/...``."""
    if s is None:
        return None
    m = _LAYER_RE.search(s)
    return int(m.group(1)) if m else None


def _resolve_hook(sae_model: SAEModel) -> tuple[int, str]:
    """Resolve (layer, hook_kind) from a loaded SAE's metadata.

    Falls back to parsing the layer index from the sae_id or hook_name when
    ``cfg.metadata.hook_layer`` is unset (some releases like gemma-scope do
    not populate it).
    """
    meta = sae_model.metadata
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


def _steer_site(hook_layer: int, hook_kind: str) -> tuple[int, str]:
    """Map an SAE hook point to the ``(layer, module)`` an intervention must
    target so steering modifies the exact tensor the SAE was trained on.

    ``SAEEncode`` reads the residual at a different site for each hook kind
    (see :func:`_capture_residual`), so steering with the default
    ``modules="residual"`` only lines up for ``resid_post`` SAEs. The mapping:

    - ``resid_post`` -> layer ``L`` output (``"residual"``)
    - ``mlp_out``    -> layer ``L`` MLP output (``"mlp"``)
    - ``attn_out``   -> layer ``L`` attention output (``"self_attn"``)
    - ``resid_pre``  -> layer ``L - 1`` output, since the resid_pre of ``L``
      is the resid_post of ``L - 1``

    Raises:
        ValueError: For a ``resid_pre`` SAE at layer 0 (its input is the
            embedding output, not reachable as a decoder layer).
        NotImplementedError: For ``resid_mid``, which has no unambiguous
            residual site on sandwich-norm models like Gemma-2.
    """
    if hook_kind == "resid_post":
        return hook_layer, "residual"
    if hook_kind == "mlp_out":
        return hook_layer, "mlp"
    if hook_kind == "attn_out":
        return hook_layer, "self_attn"
    if hook_kind == "resid_pre":
        if hook_layer == 0:
            raise ValueError(
                "Cannot steer a resid_pre SAE at layer 0: its input is the "
                "embedding output, which is not reachable as a decoder layer."
            )
        return hook_layer - 1, "residual"
    raise NotImplementedError(
        f"Steering is not supported for hook kind {hook_kind!r}; resid_mid "
        "has no unambiguous residual site on sandwich-norm models."
    )


def _is_sandwich_norm(model: ModelBackend, hook_layer: int) -> bool:
    """Whether ``model`` normalizes sublayer outputs before the residual add.

    Sandwich-norm layouts (e.g. Gemma-2/3) apply a norm to the attention output
    before it is added back, which breaks the plain ``resid_pre + attn_out``
    reconstruction of resid_mid. They are recognized by the extra feedforward
    norms their decoder layers carry; if the layout cannot be introspected the
    model is treated as plain pre-norm.
    """
    try:
        layer = model.hf_model.layers[hook_layer]
    except (AttributeError, IndexError, TypeError):
        return False
    return hasattr(layer, "post_feedforward_layernorm") or hasattr(
        layer, "pre_feedforward_layernorm"
    )


def _capture_residual(
    model: ModelBackend, tokens: Any, hook_layer: int, hook_kind: str
) -> Tensor:
    """Trace ``tokens`` through ``model`` and return the residual the SAE expects."""
    if hook_kind == "resid_mid" and _is_sandwich_norm(model, hook_layer):
        raise NotImplementedError(
            "resid_mid capture is not supported on sandwich-norm architectures "
            "(e.g. Gemma-2): the attention output is normalized before the "
            "residual add, so it cannot be reconstructed as resid_pre + attn_out. "
            "Use a resid_post SAE or a plain pre-norm architecture."
        )
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
        else:  # resid_mid (plain pre-norm residual)
            # resid_mid = resid_pre + attn_out (post-attention, pre-MLP). This
            # holds for pre-norm architectures (Llama, Mistral, GPT-2); sandwich-
            # norm layouts normalize attn_out before the residual add and are
            # rejected above, so this reconstruction is only reached where valid.
            saved["main"] = layer.input.save()
            saved["extra"] = model.resolve_module(hook_layer, "self_attn").output.save()

    residual = unwrap_traced(saved["main"])
    if "extra" in saved:
        residual = residual + unwrap_traced(saved["extra"])
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


def top_sae_features_per_prompt(
    record: SAEActivationStore,
    n: int = 1,
    reduce: str = "last",
    sink_position: int = 1,
    max_density: float = 0.4,
) -> list[list[int]]:
    """Top SAE features per prompt, excluding BOS sinks.

    Two ways to summarize a prompt into one feature vector before ranking,
    selected by ``reduce``:

    - ``"last"`` (default): the activations at the prompt's last real token.
      For a fact-completion prompt this is the concept the model has committed
      to at the end (often a local "what comes next" feature, e.g. a generic
      "city" feature for "...located in Paris").
    - ``"mean"``: the mean activation across the prompt's content tokens (real
      tokens, skipping the BOS-sink region). This asks what the sentence is
      *about as a whole*. Best with a few diverse prompts: a concept feature
      then fires on only its own prompt and stands out, whereas broad
      high-frequency features fire everywhere and are dropped (see
      ``max_density``).

    Residual-stream SAEs reliably learn a sink feature that fires huge at
    ``<bos>`` and the first content token. ``"last"`` removes those by checking
    each feature's global-peak position (``<= sink_position``); ``"mean"``
    removes them by excluding the first ``sink_position + 1`` positions from the
    average. Both default to treating positions 0-1 as the sink region, which
    relies on right padding (enforced by :class:`SAEEncode`).

    Args:
        record: SAEActivationStore from a prior :class:`SAEEncode` run.
        n: Features to keep per prompt. Must be in ``[1, n_features]``.
        reduce: ``"last"`` or ``"mean"`` (see above).
        sink_position: Index of the last position treated as the BOS-sink
            region. Defaults to 1 (positions 0-1).
        max_density: Only used when ``reduce="mean"``. Features active on more
            than this fraction of the batch's real tokens are dropped as broad,
            non-specific features (these are usually high-frequency, poorly
            interpretable features that otherwise dominate the average). Set to
            ``1.0`` to keep all features.

    Returns:
        ``[[feat_ids_for_prompt_0], ...]``. Each inner list holds up to ``n``
        feature ids; it is shorter (possibly empty) for a prompt whose top
        features were all removed as sinks or as broad features.

    Raises:
        ValueError: If ``n`` is outside ``[1, n_features]``, ``reduce`` is not
            ``"last"``/``"mean"``, or any prompt has no real tokens.
    """
    n_prompts, seq, n_features = record.activations.shape
    if n < 1 or n > n_features:
        raise ValueError(f"n must be in [1, {n_features}], got {n}")
    if reduce not in ("last", "mean"):
        raise ValueError(f"reduce must be 'last' or 'mean', got {reduce!r}")
    if not bool((record.attention_mask.sum(dim=1) >= 1).all()):
        raise ValueError("every prompt must have at least one real token")

    acts = record.activations
    mask = record.attention_mask.bool()

    if reduce == "last":
        # Activations at each prompt's last real token (right padding).
        last_pos = record.attention_mask.sum(dim=1) - 1
        pooled = stack([acts[i, int(last_pos[i])] for i in range(n_prompts)])
        # Sink detection: features whose global peak across the batch lands at
        # position <= sink_position are BOS sinks; mask them out.
        flat = acts.reshape(n_prompts * seq, n_features)
        sink = (flat.argmax(dim=0) % seq) <= sink_position
        pooled = pooled.masked_fill(sink.unsqueeze(0), float("-inf"))
    else:  # "mean": average over content tokens, what the sentence is about
        content = mask.clone()
        content[:, : sink_position + 1] = False  # drop <bos> + sink position(s)
        weight = content.to(acts.dtype)
        pooled = (acts * weight.unsqueeze(-1)).sum(dim=1) / weight.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        # Drop broad features: across a diverse batch a real concept feature
        # fires on only its own prompt (low density), while high-frequency,
        # poorly interpretable features fire on most tokens and otherwise
        # dominate the average. density = fraction of real tokens where active.
        real = mask.unsqueeze(-1)
        density = ((acts > 0) & real).sum(dim=(0, 1)).to(
            acts.dtype
        ) / real.sum().clamp_min(1)
        pooled = pooled.masked_fill((density > max_density).unsqueeze(0), float("-inf"))

    # Drop -inf slots so masked sinks never leak in when fewer than n survive.
    out: list[list[int]] = []
    for i in range(n_prompts):
        vals, idx = pooled[i].topk(n)
        out.append(
            [int(f) for f, v in zip(idx.tolist(), vals.tolist()) if v != float("-inf")]
        )
    return out


def top_sae_features_for_tokens(
    record: SAEActivationStore,
    model: ModelBackend,
    target_tokens: set[str],
    n: int = 5,
    skip_positions: int = 2,
) -> list[int]:
    """Top SAE features by mean activation on a set of target tokens.

    Unlike :func:`top_sae_features_per_prompt` (which reads each prompt's last
    token), this scans every position whose decoded token is in
    ``target_tokens`` and ranks features by their mean activation there. Use it
    to discover the feature for a concept that appears mid-sentence, e.g. to
    find a "Golden Gate Bridge" feature from prompts that mention it.

    The first ``skip_positions`` positions of every prompt are excluded to
    avoid the BOS sink (see :func:`top_sae_features_per_prompt`). Because it
    decodes token ids, this helper takes ``model`` where
    :func:`top_sae_features_per_prompt` does not.

    Args:
        record: SAEActivationStore from a prior :class:`SAEEncode` run.
        model: Model whose tokenizer decodes token ids.
        target_tokens: Decoded, stripped token strings to match, e.g.
            ``{"Golden", "Gate", "Bridge"}``.
        n: Number of features to return. Must be ``>= 1``.
        skip_positions: Leading positions to exclude per prompt to avoid the
            BOS sink. Defaults to 2 (``<bos>`` plus the first content token).

    Returns:
        The ``n`` feature ids with the highest mean activation across the
        matched positions, highest first.

    Raises:
        ValueError: If ``n < 1`` or no position matches ``target_tokens``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    tok = model.tokenizer
    decoded = [
        [tok.decode([int(t)]).strip() for t in row.tolist()] for row in record.tokens
    ]
    mask = tensor([[t in target_tokens for t in row] for row in decoded])
    mask[:, :skip_positions] = False
    if not bool(mask.any()):
        raise ValueError(
            f"no token in target_tokens {sorted(target_tokens)} found after "
            f"skipping the first {skip_positions} positions of each prompt"
        )
    matched = record.activations[mask]  # [n_matched, n_features]
    mean_per_feature = matched.mean(dim=0)
    return [int(f) for f in mean_per_feature.topk(n).indices]


@dataclass
class SAEFeatureLabels:
    """Top-promoted vocabulary tokens per SAE feature.

    Produced by projecting each feature's decoder direction through the
    model's final norm and unembedding (the standard logit-lens shortcut).
    Gives a fast, corpus-free semantic handle on what concept each feature
    carries: the tokens the feature pushes the model toward.

    This is an approximation: it skips the intermediate layers between the
    SAE's layer and the unembedding. For ground-truth input-side labels
    (the contexts that make the feature fire), pair with
    :class:`SAETopActivations`.

    Attributes:
        feat_ids: SAE feature indices included.
        tokens: ``{feat_id: list[str]}`` top-K promoted tokens per feature.
        logits: ``{feat_id: list[float]}`` top-K logit scores per feature.
        k_tokens: Cap on tokens kept per feature.
        layer: Layer the SAE was trained on.
        release: HuggingFace SAE release identifier.
        sae_id: SAE id within the release.
    """

    feat_ids: list[int]
    tokens: dict[int, list[str]]
    logits: dict[int, list[float]]
    k_tokens: int
    layer: int
    release: str
    sae_id: str


@dataclass
class SAEFeatureExamples:
    """Top-K activating contexts per SAE feature, sorted descending by activation.

    Attributes:
        feat_ids: Features included.
        contexts: ``{feat_id: list[str]}`` up to K context strings per feature
            (fewer if the batch has fewer non-padding positions).
        tokens: ``{feat_id: list[str]}`` up to K triggering tokens per feature.
        act_vals: ``{feat_id: list[float]}`` up to K activation values per feature.
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
            require_optional("sae")
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

    @property
    def metadata(self) -> Any:
        """SAE config metadata (``hook_name``, ``hook_layer``, and related fields)."""
        self._ensure_loaded()
        return self._sae.cfg.metadata

    @property
    def decoder(self) -> Tensor:
        """The decoder matrix ``[n_features, d_model]``: one direction per feature.

        The whole bank at once, for callers that project every feature through the
        unembedding (see
        :func:`~murano.plotting.plot_sae_feature_logit_effects`). For a single
        feature prefer :meth:`feature_direction`.

        Returns:
            Detached copy of the decoder weight, on the SAE's device.
        """
        self._ensure_loaded()
        return self._sae.W_dec.detach().clone()

    def encode(self, residual: Tensor) -> Tensor:
        """Encode residual ``[N, seq, d_model]`` to SAE codes ``[N, seq, n_features]``."""
        self._ensure_loaded()
        return self._sae.encode(residual.to(self.device))

    def feature_direction(self, feature_id: int) -> Tensor:
        """Decoder direction (``[d_model]``) for a single SAE feature.

        For steering prefer :func:`sae_steer`, which adds this direction at the
        exact site the SAE was trained on.

        Args:
            feature_id: SAE feature index in ``[0, n_features)``.

        Returns:
            Detached copy of the decoder row, on the SAE's device.
        """
        self._ensure_loaded()
        return self._sae.W_dec[feature_id].detach().clone()


def sae_steer(
    model: ModelBackend,
    sae_model: SAEModel,
    feature_id: int,
    alpha: float,
    gen_kwargs: dict | None = None,
) -> Intervene:
    """Build an :class:`~murano.steps.intervene.Intervene` step that steers a
    generation with a single SAE feature.

    Adds ``alpha * feature_direction(feature_id)`` to the residual stream at
    the exact site the SAE was trained on, so the same call works whether the
    SAE is ``resid_post``, ``mlp_out``, ``attn_out`` or ``resid_pre`` without
    the caller having to choose a layer or module (which is easy to get wrong:
    a hardcoded ``modules="residual"`` only matches ``resid_post`` SAEs). The
    direction is normalized to unit norm internally (see
    :func:`~murano.steps.intervene.steer_direction`), so ``alpha`` is the
    absolute magnitude added; positive enhances the feature, negative
    suppresses it.

    ``alpha`` is therefore scale-dependent, and it has no default worth quoting:
    the edit is applied at every decoded token, so what matters is its size
    relative to the residual stream at the steering site, which ranges from tens
    to thousands across models. Measure that norm and start near a tenth of it.
    Pushing much past it swamps the residual stream and the model repeats one
    token; see ``notebooks/applications/sae_steering.ipynb``, where no strength
    both invokes the concept and preserves the text. Adding a fixed magnitude is
    only one way to steer a feature; clamping the feature's own activation
    (*Scaling Monosemanticity*) leaves the rest of the residual stream intact and
    is not implemented here.

    The returned step generates a clean (unsteered) and a steered response for
    each prompt and writes an :class:`~murano.steps.intervene.InterveneResult`::

        results = Pipeline([
            LoadPrompts(prompts),
            sae_steer(model, encode.sae_model, feature_id, alpha=alpha),
        ]).run()
        comparison = results["intervene"]    # clean vs steered

    Args:
        model: Model to generate with (must match the SAE's base model).
        sae_model: Loaded SAE providing the feature direction and hook point.
        feature_id: SAE feature index in ``[0, n_features)``.
        alpha: Steering strength, as an absolute magnitude added to the residual
            stream; positive enhances, negative suppresses. Calibrate against the
            residual norm at the steering site.
        gen_kwargs: Generation kwargs forwarded to the model. Defaults to
            ``{"max_new_tokens": 256, "do_sample": False}``.

    Returns:
        A configured :class:`~murano.steps.intervene.Intervene` step.

    Raises:
        ValueError: If the resolved steering layer is out of bounds for
            ``model``, or for a ``resid_pre`` SAE at layer 0.
        NotImplementedError: For SAE hook kinds with no clean steering site
            (e.g. ``resid_mid``).
    """
    from murano.steps.intervene import Intervene, steer_direction

    sae_model._ensure_loaded()
    hook_layer, hook_kind = _resolve_hook(sae_model)
    layer, module = _steer_site(hook_layer, hook_kind)
    if layer < 0 or layer >= model.n_layers:
        raise ValueError(
            f"Steering site resolved to layer {layer}, out of range for a "
            f"model with {model.n_layers} layers."
        )
    direction = sae_model.feature_direction(feature_id)
    # A bare layer int coerces only to the residual module, so key by the exact
    # Node the generation hook uses; this matches every hook kind.
    fn = steer_direction({Node(layer, module): direction}, alpha)
    return Intervene(
        model, fn=fn, layers=[layer], modules=[module], gen_kwargs=gen_kwargs
    )


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

        # Force right-padding for this encode. The downstream feature helpers
        # (top_sae_features_per_prompt, top_sae_features_for_tokens) assume the
        # BOS sink sits at positions 0-1 and the last real token at sum(mask)-1,
        # which only holds for right padding. Many instruct tokenizers (Gemma
        # included) default to left padding, which would shift every position.
        tok = self.model.tokenizer
        prev_side = tok.padding_side
        tok.padding_side = "right"
        try:
            tokens = tok(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=self.max_length is not None,
                max_length=self.max_length,
                return_token_type_ids=False,
            )
        finally:
            tok.padding_side = prev_side
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
    """Input-side interpretation: rank the top-K activating contexts per SAE feature.

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


class SAEFeatureLabel(Step):
    """Output-side interpretation: label SAE features by the tokens they promote.

    For each requested feature, projects ``W_dec[feature_id]`` through the
    model's standardized final norm and unembedding via
    ``model.project_on_vocab`` (the same logit-lens projection
    :class:`~murano.steps.logit_lens.LogitLens` applies to per-layer
    residuals). The top-K tokens by resulting logit are returned as a label
    per feature.

    This is the fast, corpus-free counterpart to
    :class:`SAETopActivations`. Pair the two when possible:
    ``SAETopActivations`` tells you what makes a feature fire (input side),
    this tells you what the feature pushes the model toward (output side).

    Reads from results:
        results['sae_record']: SAEActivationStore. The SAE's ``release``
            and ``sae_id`` are taken from here so the same SAE used for
            encoding is used for labeling.

    Writes to results:
        results['feature_labels']: SAEFeatureLabels

    Args:
        model: Model providing ``project_on_vocab`` (final norm + unembedding).
        feat_ids: SAE feature indices to label.
        k_tokens: How many top-promoted tokens to keep per feature.
            Must be ``>= 1``.
        sae_model: Optional pre-built :class:`SAEModel` to source decoder
            directions from. When ``None`` (default), one is constructed
            from ``sae_record.release`` and ``sae_record.sae_id``. Pass an
            existing instance to avoid reloading when chaining after
            :class:`SAEEncode`.

    Raises:
        ValueError: If ``k_tokens < 1`` or any requested ``feat_id`` is out
            of range for the SAE width recorded in ``sae_record``.
    """

    reads = ["sae_record"]
    writes = ["feature_labels"]
    read_types = {"sae_record": SAEActivationStore}
    write_types = {"feature_labels": SAEFeatureLabels}

    def __init__(
        self,
        model: ModelBackend,
        feat_ids: list[int],
        k_tokens: int = 5,
        *,
        sae_model: SAEModel | None = None,
    ):
        if k_tokens < 1:
            raise ValueError(f"k_tokens must be >= 1, got {k_tokens}")
        self.model = model
        self.feat_ids = list(feat_ids)
        self.k_tokens = k_tokens
        self.sae_model = sae_model

    def __call__(self, results: Results) -> Results:
        store: SAEActivationStore = results["sae_record"]
        feat_ids = self.feat_ids
        invalid = [f for f in feat_ids if f < 0 or f >= store.n_features]
        if invalid:
            raise ValueError(
                f"feat_ids {invalid} out of range for SAE with "
                f"{store.n_features} features"
            )

        sae_model = self.sae_model
        if sae_model is None:
            sae_model = SAEModel(release=store.release, sae_id=store.sae_id)
        sae_model._ensure_loaded()

        tokens_per_feat: dict[int, list[str]] = {}
        logits_per_feat: dict[int, list[float]] = {}
        with no_grad():
            for fid in feat_ids:
                direction = sae_model.feature_direction(fid)
                feat_logits = self.model.project_on_vocab(direction)
                top_vals, top_ids = feat_logits.topk(self.k_tokens)
                tokens_per_feat[int(fid)] = [
                    self.model.tokenizer.decode([int(t)]) for t in top_ids.tolist()
                ]
                logits_per_feat[int(fid)] = [float(v) for v in top_vals.tolist()]

        logger.info(
            "SAEFeatureLabel: %d features, k_tokens=%d, layer=%d",
            len(feat_ids),
            self.k_tokens,
            store.hook.layer,
        )

        results["feature_labels"] = SAEFeatureLabels(
            feat_ids=[int(f) for f in feat_ids],
            tokens=tokens_per_feat,
            logits=logits_per_feat,
            k_tokens=self.k_tokens,
            layer=store.hook.layer,
            release=store.release,
            sae_id=store.sae_id,
        )
        return results
