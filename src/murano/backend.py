"""ModelBackend: the model surface that pipeline steps depend on.

Steps interact with the model only through this protocol, never by reaching
into the underlying nnterp/nnsight objects. :class:`murano.model.MuranoModel`
is the sole implementation today; the protocol records the surface a future
backend would have to provide.

Note:
    This wraps nnterp/nnsight and is not a backend-agnostic abstraction.
    ``trace``, ``layer``, ``resolve_module``, and ``attn_out_proj`` return
    nnsight-compatible proxy objects, and steps still rely on nnsight proxy
    semantics (``.output.save()``, ``.input.save()``, assigning to
    ``mod_proxy.output``); the ``raw_*`` accessors instead return the underlying
    ``torch.nn.Module`` for native hook registration and weight reads. A
    non-nnsight backend would have to reproduce those
    proxy semantics, so this protocol is a prerequisite for backend
    independence, not backend independence itself.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Callable

    from murano.nodes import Node


@runtime_checkable
class ModelBackend(Protocol):
    """Surface a model must expose for Murano's pipeline steps.

    Members that return nnsight proxies are typed ``Any`` on purpose: the
    proxy types are not worth fighting the type checker over, and steps consume
    them through nnsight's runtime proxy semantics (see the module note).

    Attributes:
        n_layers: Number of decoder layers.
        d_model: Residual-stream width.
        n_heads: Number of attention heads.
        head_dim: Per-head width (``d_model // n_heads``).
        tokenizer: The model's tokenizer.
        hf_model: Underlying HuggingFace module, for weight-level access.
    """

    n_layers: int
    d_model: int
    n_heads: int
    head_dim: int
    tokenizer: Any
    hf_model: Any

    @property
    def raw_model(self) -> Any:
        """Underlying causal-LM module, including its output embedding."""
        ...

    def layer(self, idx: int) -> Any:
        """Return the module proxy for decoder layer ``idx``."""
        ...

    def resolve_module(self, layer_idx: int, module: str) -> Any:
        """Return the submodule proxy named ``module`` at layer ``layer_idx``."""
        ...

    def attn_out_proj(self, layer_idx: int, module: str) -> Any:
        """Return the attention output-projection proxy for per-head capture."""
        ...

    def raw_layer(self, idx: int) -> Any:
        """Return the raw ``torch.nn.Module`` for decoder layer ``idx``.

        For native ``torch`` hook registration or weight access, where a step
        needs the module itself rather than an nnsight proxy.
        """
        ...

    def raw_module(self, layer_idx: int, module: str) -> Any:
        """Return the raw ``torch.nn.Module`` named ``module`` at ``layer_idx``."""
        ...

    def raw_attn_out_proj(self, layer_idx: int, module: str) -> Any:
        """Return the raw ``torch.nn.Module`` of the attention output projection."""
        ...

    def trace(self, tokens: Any) -> AbstractContextManager[Any]:
        """Open a tracing context over ``tokens``."""
        ...

    def project_on_vocab(self, hidden: Tensor) -> Tensor:
        """Project hidden states onto the vocabulary."""
        ...

    @property
    def unembed_weight(self) -> Tensor:
        """The unembedding (lm_head) weight ``[vocab, d_model]``."""
        ...

    @property
    def final_norm(self) -> Any:
        """The standardized final-normalization module (before the unembedding)."""
        ...

    def forward_logits(
        self,
        tokens: Any,
        fn: "Callable[[Tensor, Node], Tensor] | None" = None,
        layers: list[int] | str = "all",
        modules: str | list[str] = "residual",
        per_head: bool = False,
    ) -> Tensor:
        """Run a forward pass and return output logits ``[B, S, V]``.

        With ``fn`` given, rewrite each target module's activation before the
        logits are read (the forward-pass analogue of ``generate_with_hooks``);
        ``per_head`` exposes the per-head activation for attention heads.
        """
        ...

    def generate_with_hooks(
        self,
        text: str,
        fn: "Callable[[Tensor, Node], Tensor] | None" = None,
        layers: list[int] | str = "all",
        modules: str | list[str] = "residual",
        gen_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Generate from ``text``, optionally applying ``fn`` per layer/module."""
        ...

    @property
    def attn_probs_available(self) -> bool:
        """Whether per-head attention weights can be read and written."""
        ...

    @property
    def attention_probabilities(self) -> Any:
        """Accessor for per-head softmax attention weights.

        Inside a :meth:`trace`, ``[layer]`` reads the ``[batch, n_heads, query,
        key]`` pattern and ``[layer] = tensor`` overwrites it. Only usable when
        the backend was loaded with attention weights enabled.
        """
        ...

    def require_attention_probs(self) -> None:
        """Raise if per-head attention weights are unavailable on this backend."""
        ...

    def forward_logits_attention(
        self, tokens: Any, edits: "dict[int, tuple[Tensor, Tensor]]"
    ) -> Tensor:
        """Run a forward pass overwriting per-head attention weights, return logits.

        Each layer's weights become ``pattern * (1 - mask) + replacement * mask``
        for its ``{layer: (replacement, mask)}`` entry in ``edits``.
        """
        ...

    def chat_template(self, messages: list[dict[str, Any]]) -> str:
        """Render ``messages`` through the tokenizer's chat template."""
        ...
