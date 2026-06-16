"""Node: one component-address vocabulary for "where in the model".

Murano addresses activation sites three incompatible ways today: a
``(layer, module)`` tuple in the recording and steering stores, a bare layer
index in the logit-lens result, and a layer int plus a hook string in the SAE
store. :class:`Node` is the single canonical address those will converge on.

A :class:`Node` names one component: a layer, a module within that layer, and
optional sub-addressing (attention head, Q/K/V/O side, token position). Shorthand
forms coerce to it via :meth:`Node.coerce`: a bare ``int`` is that layer's block
output, a ``(layer, module)`` tuple is the historical key, and a string like
``"L5.self_attn.h3"`` round-trips through :meth:`Node.parse`. Optional fields
refine the same site; an SAE feature, an attention pattern, or a single neuron
is a genuinely different object and gets its own type composed on a
:class:`Node`, not an extra field here.

This module is deliberately free of any heavy dependency (no torch): it is the
address vocabulary, not the resolver that turns an address into a live hook. That
resolution (head and side slicing, fused vs split QKV, grouped-query attention,
position selection) is a separate concern handled by the model, much of it still
future work.

Canonical module names are the residual-stream points :data:`RESID_PRE`,
:data:`RESID_MID`, :data:`RESID_POST` and the sub-block outputs :data:`MLP` and
:data:`SELF_ATTN`; dotted submodules such as ``self_attn.o_proj`` pass through
unchanged. ``residual``, ``mlp_out``, and ``attn_out`` are accepted as aliases
and normalized to the canonical names at construction.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from functools import total_ordering
from re import fullmatch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import TypeAlias


# Canonical residual-stream addressing points: block input, post-attention
# midpoint, block output. RESID_MID is a derived sum (layer.input +
# self_attn.output), not a single module hook; the resolver computes it.
RESID_PRE = "resid_pre"
RESID_MID = "resid_mid"
RESID_POST = "resid_post"

# Canonical sub-block module outputs. These are the model's actual submodule
# names, so they resolve directly.
MLP = "mlp"
SELF_ATTN = "self_attn"

# A bare layer index addresses the block output, the common non-expert target
# ("layer 5's residual stream").
_DEFAULT_MODULE = RESID_POST

# Friendly, TransformerLens, and sae-lens spellings normalized to canonical
# names at construction, so a Node's ``module`` field is always canonical. That
# invariant is what makes ``Node.parse(str(node)) == node`` hold.
_MODULE_ALIASES = {
    "residual": RESID_POST,
    "mlp_out": MLP,
    "attn_out": SELF_ATTN,
}


def canonical_module(module: str) -> str:
    """Return the canonical spelling of a module name, normalizing aliases.

    Maps ``residual`` to :data:`RESID_POST`, ``mlp_out`` to :data:`MLP`, and
    ``attn_out`` to :data:`SELF_ATTN`; any other name (canonical names and
    dotted submodules) is returned unchanged. This is the single source the
    address type and the model's hook resolver share, so a stored ``Node``'s
    module always resolves back to a live hook.
    """
    return _MODULE_ALIASES.get(module, module)


class Side(str, Enum):
    """Attention projection side: query, key, value, or output.

    Subclasses ``str`` so a side compares and serializes as its short name
    (``Side.Q == "Q"``) and survives the ``torch.save``/JSON path without a
    custom codec.
    """

    Q = "Q"
    K = "K"
    V = "V"
    O = "O"  # noqa: E741 - canonical name for the attention output side.

    def __str__(self) -> str:
        return self.value


_SIDE_VALUES = frozenset(s.value for s in Side)


def _coerce_side(side: object) -> Side:
    """Return ``side`` as a :class:`Side`, accepting the short string form.

    Raises:
        ValueError: If ``side`` is not a :class:`Side` or one of Q/K/V/O.
    """
    if isinstance(side, Side):
        return side
    try:
        return Side(str(side).upper())
    except ValueError:
        raise ValueError(
            f"side must be one of {sorted(_SIDE_VALUES)} or a Side, got {side!r}"
        ) from None


def _validated_int(value: object, name: str, *, allow_negative: bool) -> int:
    """Return ``value`` as a plain int, validating it is a non-bool int.

    Normalizes int subclasses (e.g. ``IntEnum``) to a plain ``int`` so the
    stored field has a stable repr and string form across Python versions.

    Raises:
        ValueError: If ``value`` is a bool, not an int, or negative when
            ``allow_negative`` is False.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {value!r}")
    if not allow_negative and value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return int(value)


@total_ordering
@dataclass(frozen=True, slots=True)
class Node:
    """One component address: a layer, a module, and optional sub-addressing.

    A bare ``Node(layer, module)`` names a whole module's activation, equivalent
    to today's ``(layer, module)`` key. ``head``, ``side``, and ``position``
    refine that same site rather than selecting a different kind of object.

    ``head`` and ``side`` are attention-only and are accepted only when
    ``module`` is :data:`SELF_ATTN` (after alias normalization); combining them
    with any other module raises ``ValueError`` so a nonsensical address never
    reaches a store.

    Attributes:
        layer: Decoder layer index (``>= 0``).
        module: Canonical module name (:data:`RESID_PRE` / :data:`RESID_MID` /
            :data:`RESID_POST` / :data:`MLP` / :data:`SELF_ATTN`) or a dotted
            submodule such as ``self_attn.o_proj``. Aliases ``residual`` /
            ``mlp_out`` / ``attn_out`` are normalized at construction.
        head: Attention head index (``>= 0``), or ``None`` for the whole module.
        position: Token position, or ``None`` for every position. May be
            negative to index from the end, matching ``Record``'s convention.
        side: Attention projection side (:class:`Side`), or ``None``.
    """

    layer: int
    module: str
    head: int | None = None
    position: int | None = None
    side: Side | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "layer", _validated_int(self.layer, "layer", allow_negative=False)
        )
        if not isinstance(self.module, str) or not self.module:
            raise ValueError(f"module must be a non-empty str, got {self.module!r}")

        canonical = _MODULE_ALIASES.get(self.module, self.module)
        if canonical != self.module:
            object.__setattr__(self, "module", canonical)

        # An empty dotted segment ("mlp.", "self_attn..o_proj") cannot be a real
        # attribute path, so reject it before it reaches a store or resolver.
        if "" in self.module.split("."):
            raise ValueError(
                f"module {self.module!r} has an empty path segment; use a dotted "
                f"attribute path like 'self_attn.o_proj'."
            )

        # A module segment shaped like a head ("h<int>") or side (Q/K/V/O) token
        # is indistinguishable from a head/side suffix in str()/parse(), so the
        # round-trip invariant requires forbidding it. Real submodule names do
        # not collide (e.g. o_proj, gate_proj).
        last_segment = self.module.rsplit(".", 1)[-1]
        if last_segment in _SIDE_VALUES or fullmatch(r"h\d+", last_segment):
            raise ValueError(
                f"module {self.module!r} ends in {last_segment!r}, which collides "
                f"with the head/side address grammar; rename the segment or "
                f"address the head/side explicitly."
            )

        if self.side is not None and not isinstance(self.side, Side):
            object.__setattr__(self, "side", _coerce_side(self.side))
        if self.head is not None:
            object.__setattr__(
                self, "head", _validated_int(self.head, "head", allow_negative=False)
            )
        if self.position is not None:
            object.__setattr__(
                self,
                "position",
                _validated_int(self.position, "position", allow_negative=True),
            )

        # A head or side names one attention head/projection, which only exists
        # on the attention module; reject the combination anywhere else.
        if (
            self.head is not None or self.side is not None
        ) and self.module != SELF_ATTN:
            raise ValueError(
                f"head/side are attention-only and require module={SELF_ATTN!r}, "
                f"got module={self.module!r}. To sub-address attention, use "
                f"module={SELF_ATTN!r} with a head/side rather than a dotted "
                f"submodule."
            )

    def __str__(self) -> str:
        text = f"L{self.layer}.{self.module}"
        if self.head is not None:
            text += f".h{self.head}"
        if self.side is not None:
            text += f".{self.side.value}"
        if self.position is not None:
            text += f"@p{self.position}"
        return text

    def __reduce__(self):
        # Frozen + slots dataclasses are not picklable by default: the unpickler
        # restores slots via setattr, which the frozen __setattr__ rejects.
        # Reconstruct through the constructor so torch.save round-trips a Node.
        return (
            self.__class__,
            (self.layer, self.module, self.head, self.position, self.side),
        )

    def _order_key(self) -> tuple:
        # None head/side sort before any real value; position may be negative, so
        # a missing position sorts below every concrete index.
        return (
            self.layer,
            self.module,
            -1 if self.head is None else self.head,
            float("-inf") if self.position is None else self.position,
            "" if self.side is None else self.side.value,
        )

    def __lt__(self, other: object):
        if not isinstance(other, Node):
            return NotImplemented
        return self._order_key() < other._order_key()

    @classmethod
    def coerce(cls, address: AddressLike) -> Node:
        """Return ``address`` as a :class:`Node`, accepting shorthand forms.

        This is the seam every public API uses so callers can pass shorthand
        while stored keys and returned values stay canonical :class:`Node`
        objects.

        Args:
            address: A :class:`Node` (returned unchanged), a bare layer ``int``
                (that layer's block output), a ``(layer, module)`` tuple, or a
                canonical string parsed by :meth:`parse`.

        Returns:
            The equivalent canonical :class:`Node`.

        Raises:
            TypeError: If ``address`` is none of the accepted forms.
        """
        if isinstance(address, Node):
            return address
        if isinstance(address, bool):
            raise TypeError(f"address must not be a bool, got {address!r}")
        if isinstance(address, int):
            return cls(address, _DEFAULT_MODULE)
        if isinstance(address, str):
            return cls.parse(address)
        if isinstance(address, tuple):
            if (
                len(address) != 2
                or not isinstance(address[0], int)
                or isinstance(address[0], bool)
                or not isinstance(address[1], str)
            ):
                raise TypeError(
                    f"tuple address must be (layer: int, module: str), got {address!r}"
                )
            return cls(address[0], address[1])
        raise TypeError(
            f"cannot coerce {type(address).__name__} to Node; expected int, "
            f"(layer, module) tuple, str, or Node"
        )

    @classmethod
    def parse(cls, text: str) -> Node:
        """Parse a canonical address string into a :class:`Node`.

        The grammar is ``L<layer>.<module>[.h<head>][.<side>][@p<position>]``,
        where ``<module>`` may itself contain dots (e.g. ``self_attn.o_proj``).
        ``Node.parse(str(node))`` reproduces ``node`` exactly.

        Args:
            text: A canonical address string, e.g. ``"L5.self_attn.h3.Q@p-1"``.

        Returns:
            The parsed :class:`Node`.

        Raises:
            ValueError: If ``text`` does not match the address grammar.
        """
        raw = text
        position: int | None = None
        if "@" in text:
            text, _, pos_part = text.partition("@")
            if not pos_part.startswith("p"):
                raise ValueError(f"position suffix must look like '@p<int>': {raw!r}")
            position = _parse_int(pos_part[1:], "position", raw)

        if not text.startswith("L"):
            raise ValueError(f"address must start with 'L<layer>': {raw!r}")
        layer_str, dot, rest = text[1:].partition(".")
        if not dot:
            raise ValueError(f"address must name a module after the layer: {raw!r}")
        layer = _parse_int(layer_str, "layer", raw)

        # The head ("h<int>") and side (Q/K/V/O) suffixes are recognized only as
        # trailing dotted tokens, so the remaining tokens (which may contain
        # dots for a submodule) reassemble into the module name.
        tokens = rest.split(".")
        side: Side | None = None
        if tokens and tokens[-1] in _SIDE_VALUES:
            side = Side(tokens.pop())
        head: int | None = None
        if tokens and fullmatch(r"h\d+", tokens[-1]):
            head = int(tokens.pop()[1:])
        module = ".".join(tokens)
        if not module:
            raise ValueError(f"address has an empty module: {raw!r}")
        return cls(layer, module, head=head, position=position, side=side)


AddressLike: TypeAlias = int | tuple[int, str] | str | Node


def _parse_int(value: str, name: str, raw: str) -> int:
    """Parse ``value`` as an int, naming ``name`` and ``raw`` on failure.

    Raises:
        ValueError: If ``value`` is not an integer literal.
    """
    try:
        return int(value)
    except ValueError:
        raise ValueError(
            f"{name} must be an integer in address {raw!r}, got {value!r}"
        ) from None


@dataclass(frozen=True, slots=True)
class Edge:
    """A directed connection between two component addresses.

    Expresses path-patching / circuit edges. Construction and serialization
    land now; the engine that consumes edges is later work.

    Attributes:
        source: Upstream :class:`Node`.
        dest: Downstream :class:`Node`.
    """

    source: Node
    dest: Node

    def __post_init__(self) -> None:
        for name, value in (("source", self.source), ("dest", self.dest)):
            if not isinstance(value, Node):
                raise TypeError(f"Edge {name} must be a Node, got {value!r}")

    def __str__(self) -> str:
        return f"{self.source} -> {self.dest}"

    def __reduce__(self):
        return (self.__class__, (self.source, self.dest))

    @classmethod
    def parse(cls, text: str) -> Edge:
        """Parse ``"<source> -> <dest>"`` into an :class:`Edge`.

        Args:
            text: An edge string, e.g. ``"L5.self_attn.h3 -> L8.mlp"``.

        Returns:
            The parsed :class:`Edge`.

        Raises:
            ValueError: If ``text`` does not have exactly one ``" -> "``
                separator.
        """
        if text.count(" -> ") != 1:
            raise ValueError(
                f"edge must look like '<source> -> <dest>' with one separator: {text!r}"
            )
        src, _, dst = text.partition(" -> ")
        return cls(Node.parse(src), Node.parse(dst))


@dataclass(frozen=True, slots=True)
class NodeSet:
    """An ordered, de-duplicated collection of :class:`Node` addresses.

    Deliberately a plain container, not a query language: it preserves insertion
    order, drops duplicates, and supports iteration, ``len``, and membership.
    The classmethods build common fan-outs (a span of layers, the heads of one
    attention module, a set of positions, a cartesian product) as singleton
    :class:`Node` elements, so a range never has to live inside a single
    :class:`Node`.

    Attributes:
        nodes: The de-duplicated nodes in insertion order.
    """

    nodes: tuple[Node, ...]

    def __post_init__(self) -> None:
        # dict preserves insertion order and de-duplicates by Node equality.
        deduped: dict[Node, None] = {}
        for node in self.nodes:
            if not isinstance(node, Node):
                raise TypeError(f"NodeSet holds Node objects, got {node!r}")
            deduped.setdefault(node, None)
        object.__setattr__(self, "nodes", tuple(deduped))

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, node: object) -> bool:
        return node in self.nodes

    def __reduce__(self):
        return (self.__class__, (self.nodes,))

    @classmethod
    def expand_layers(
        cls,
        layers: Iterable[int],
        module: str = _DEFAULT_MODULE,
        *,
        head: int | None = None,
        side: Side | None = None,
        position: int | None = None,
    ) -> NodeSet:
        """Return one :class:`Node` per layer at a fixed module and refinement."""
        return cls(
            tuple(
                Node(layer, module, head=head, side=side, position=position)
                for layer in layers
            )
        )

    @classmethod
    def expand_heads(
        cls,
        layer: int,
        heads: Iterable[int],
        *,
        side: Side | None = None,
        position: int | None = None,
    ) -> NodeSet:
        """Return one attention :class:`Node` per head at ``layer``."""
        return cls(
            tuple(
                Node(layer, SELF_ATTN, head=head, side=side, position=position)
                for head in heads
            )
        )

    @classmethod
    def expand_positions(cls, node: Node, positions: Iterable[int]) -> NodeSet:
        """Return ``node`` repeated once per token position."""
        return cls(tuple(replace(node, position=position) for position in positions))

    @classmethod
    def product(
        cls,
        layers: Iterable[int],
        modules: Iterable[str],
        *,
        heads: Iterable[int | None] = (None,),
        sides: Iterable[Side | None] = (None,),
        positions: Iterable[int | None] = (None,),
    ) -> NodeSet:
        """Return the cartesian product of the given address axes.

        Each combination is constructed as a :class:`Node`, so an invalid
        combination (a head on a non-attention module) raises ``ValueError``.
        """
        return cls(
            tuple(
                Node(layer, module, head=head, side=side, position=position)
                for layer in layers
                for module in modules
                for head in heads
                for side in sides
                for position in positions
            )
        )


class NodeDict(dict):
    """A ``dict`` keyed by :class:`Node` that coerces shorthand keys on access.

    Lets non-experts look activations up with shorthand (``store[5]``,
    ``store[(5, "mlp")]``, ``store["L5.mlp"]``) while every stored key stays a
    canonical :class:`Node`. Keys are coerced through :meth:`Node.coerce` on
    insertion, lookup, membership, and deletion, so two spellings of the same
    address resolve to one entry.

    Membership and :meth:`get` of an uncoercible key return ``False`` / the
    default rather than raising, matching plain-``dict`` semantics for a missing
    key.
    """

    def __init__(self, data=None):
        super().__init__()
        if data is not None:
            items = data.items() if hasattr(data, "items") else data
            for key, value in items:
                self[key] = value

    def __setitem__(self, key, value) -> None:
        super().__setitem__(Node.coerce(key), value)

    def __getitem__(self, key):
        return super().__getitem__(Node.coerce(key))

    def __delitem__(self, key) -> None:
        super().__delitem__(Node.coerce(key))

    def __contains__(self, key) -> bool:
        try:
            return super().__contains__(Node.coerce(key))
        except (TypeError, ValueError):
            return False

    def get(self, key, default=None):
        try:
            return super().get(Node.coerce(key), default)
        except (TypeError, ValueError):
            return default

    def setdefault(self, key, default=None):
        return super().setdefault(Node.coerce(key), default)

    def update(self, other=(), /, **kwargs) -> None:
        # dict.update bypasses __setitem__, so route every key through coercion
        # to keep stored keys canonical.
        items = other.items() if hasattr(other, "items") else other
        for key, value in items:
            self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def copy(self) -> NodeDict:
        return NodeDict(self)

    def pop(self, key, *args):
        # Match get/__contains__: coerce the lookup; honor a supplied default for
        # an absent or uncoercible key, otherwise raise KeyError.
        try:
            node = Node.coerce(key)
        except (TypeError, ValueError):
            if args:
                return args[0]
            raise KeyError(key) from None
        return super().pop(node, *args)

    # dict's | and |= are C-level and bypass __setitem__, so an uncoerced key
    # could be stored (and then be unreachable through the coercing lookups).
    # Route every merge through the coercing update().
    def __ior__(self, other):
        self.update(other)
        return self

    def __or__(self, other) -> NodeDict:
        merged = NodeDict(self)
        merged.update(other)
        return merged

    def __ror__(self, other) -> NodeDict:
        merged = NodeDict(other)
        merged.update(self)
        return merged


__all__ = [
    "MLP",
    "RESID_MID",
    "RESID_PRE",
    "RESID_POST",
    "SELF_ATTN",
    "AddressLike",
    "Edge",
    "Node",
    "NodeDict",
    "NodeSet",
    "Side",
    "canonical_module",
]
