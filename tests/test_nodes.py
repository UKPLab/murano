"""Tests for the Node component-address vocabulary.

CPU-only and torch-free: these exercise the address type itself, not hook
resolution. The operations-coverage tests cover the full addressing surface
(per-head, residual/MLP/attention outputs, Q/K/V/O sides, single/range/set
positions, edges, sweeps, and string round-trips).
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

from murano.nodes import (
    MLP,
    RESID_MID,
    RESID_POST,
    RESID_PRE,
    SELF_ATTN,
    AddressLike,
    Edge,
    Node,
    NodeDict,
    NodeSet,
    Side,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


# ── Construction & Defaults ───────────────────────────────────────────
def test_bare_node_defaults():
    node = Node(5, RESID_POST)
    assert node.layer == 5
    assert node.module == RESID_POST
    assert node.head is None
    assert node.position is None
    assert node.side is None


def test_attention_node_with_head_side_position():
    node = Node(5, SELF_ATTN, head=3, position=-1, side=Side.Q)
    assert node.head == 3
    assert node.position == -1
    assert node.side is Side.Q


# ── Alias Normalization ───────────────────────────────────────────────
@pytest.mark.parametrize(
    ("alias", "canonical"),
    [("residual", RESID_POST), ("mlp_out", MLP), ("attn_out", SELF_ATTN)],
)
def test_module_aliases_normalize_to_canonical(alias, canonical):
    assert Node(5, alias).module == canonical


def test_attn_out_alias_allows_head():
    # attn_out normalizes to self_attn, so head addressing is valid through it.
    node = Node(5, "attn_out", head=2)
    assert node.module == SELF_ATTN
    assert node.head == 2


def test_canonical_names_pass_through():
    for module in (RESID_PRE, RESID_MID, RESID_POST, MLP, SELF_ATTN):
        assert Node(5, module).module == module


def test_dotted_submodule_passes_through():
    assert Node(5, "self_attn.o_proj").module == "self_attn.o_proj"


# ── Fail-Fast Validation ──────────────────────────────────────────────
def test_head_on_non_attention_raises():
    with pytest.raises(ValueError, match="attention-only"):
        Node(5, MLP, head=1)


def test_side_on_non_attention_raises():
    with pytest.raises(ValueError, match="attention-only"):
        Node(5, RESID_POST, side=Side.Q)


def test_head_on_dotted_attention_submodule_raises():
    # Per-head addressing attaches to the whole self_attn module, not a dotted
    # submodule; the resolver does the slicing.
    with pytest.raises(ValueError, match="attention-only"):
        Node(5, "self_attn.o_proj", head=0)


def test_negative_layer_raises():
    with pytest.raises(ValueError, match="layer must be >= 0"):
        Node(-1, RESID_POST)


def test_bool_layer_raises():
    with pytest.raises(ValueError, match="layer must be an int"):
        Node(True, RESID_POST)


def test_negative_head_raises():
    with pytest.raises(ValueError, match="head must be >= 0"):
        Node(5, SELF_ATTN, head=-1)


def test_empty_module_raises():
    with pytest.raises(ValueError, match="non-empty"):
        Node(5, "")


def test_negative_position_allowed():
    assert Node(5, RESID_POST, position=-1).position == -1


def test_side_string_normalizes_to_enum():
    assert Node(5, SELF_ATTN, side="Q").side is Side.Q
    assert Node(5, SELF_ATTN, side="o").side is Side.O


def test_invalid_side_raises():
    with pytest.raises(ValueError, match="side must be one of"):
        Node(5, SELF_ATTN, side="X")


def test_side_equality_independent_of_input_form():
    assert Node(5, SELF_ATTN, side="Q") == Node(5, SELF_ATTN, side=Side.Q)


def test_side_on_mlp_module_raises():
    with pytest.raises(ValueError, match="attention-only"):
        Node(5, "mlp", side=Side.Q)


def test_bool_head_raises():
    with pytest.raises(ValueError, match="head must be an int"):
        Node(5, SELF_ATTN, head=True)


def test_float_head_raises():
    with pytest.raises(ValueError, match="head must be an int"):
        Node(5, SELF_ATTN, head=1.0)


def test_bool_position_raises():
    with pytest.raises(ValueError, match="position must be an int"):
        Node(5, RESID_POST, position=True)


def test_non_int_position_raises():
    with pytest.raises(ValueError, match="position must be an int"):
        Node(5, RESID_POST, position="0")


def test_non_str_module_raises():
    with pytest.raises(ValueError, match="non-empty str"):
        Node(5, 5)


def test_int_subclass_normalized_to_plain_int():
    from enum import IntEnum

    class L(IntEnum):
        FIVE = 5

    node = Node(L.FIVE, RESID_POST)
    assert type(node.layer) is int
    assert str(node) == "L5.resid_post"


@pytest.mark.parametrize(
    "module",
    ["self_attn.O", "self_attn.Q", "self_attn.h0", "mlp.O", "resid_post.K", "h3", "O"],
)
def test_module_colliding_with_head_side_grammar_raises(module):
    # A module whose last dotted segment looks like a head/side suffix would
    # break str/parse round-trips, so construction must reject it.
    with pytest.raises(ValueError, match="collides with the head/side"):
        Node(5, module)


@pytest.mark.parametrize("module", ["mlp.", ".mlp", "self_attn..o_proj", "."])
def test_module_with_empty_segment_raises(module):
    with pytest.raises(ValueError, match="empty path segment"):
        Node(5, module)


def test_parse_rejects_empty_dotted_segment():
    with pytest.raises(ValueError, match="empty path segment"):
        Node.parse("L5.self_attn..o_proj")


# ── Hashing, Equality, Pickling ───────────────────────────────────────
def test_nodes_are_hashable_and_usable_as_dict_keys():
    store = {Node(5, RESID_POST): "a", Node(5, MLP): "b"}
    assert store[Node(5, RESID_POST)] == "a"
    assert store[Node(5, MLP)] == "b"


def test_equal_nodes_hash_equal():
    assert hash(Node(5, "residual")) == hash(Node(5, RESID_POST))


def test_pickle_round_trip():
    for node in (
        Node(5, RESID_POST),
        Node(7, SELF_ATTN, head=3, position=-1, side=Side.O),
        Node(0, "self_attn.o_proj"),
    ):
        assert pickle.loads(pickle.dumps(node)) == node


def test_pickle_round_trip_edge_and_nodeset():
    edge = Edge(Node(5, SELF_ATTN, head=3, side=Side.O, position=-1), Node(8, MLP))
    assert pickle.loads(pickle.dumps(edge)) == edge
    node_set = NodeSet((Node(5, RESID_POST), Node(6, SELF_ATTN, head=2, position=-1)))
    assert pickle.loads(pickle.dumps(node_set)) == node_set


def test_torch_save_round_trip():
    # Node must survive the torch.save path; __reduce__ exists for
    # exactly this, so exercise it (not just stdlib pickle).
    import io

    import torch

    for obj in (
        Node(7, SELF_ATTN, head=3, position=-1, side=Side.O),
        Edge(Node(5, SELF_ATTN, head=3), Node(8, MLP)),
        NodeSet((Node(5, RESID_POST), Node(6, MLP))),
    ):
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        buffer.seek(0)
        assert torch.load(buffer, weights_only=False) == obj


# ── String Round-Trips ────────────────────────────────────────────────
@pytest.mark.parametrize(
    "node",
    [
        Node(5, RESID_POST),
        Node(0, RESID_PRE),
        Node(11, RESID_MID),
        Node(3, MLP),
        Node(7, SELF_ATTN),
        Node(9, SELF_ATTN, head=9),
        Node(9, SELF_ATTN, head=9, side=Side.Q),
        Node(9, SELF_ATTN, side=Side.K),
        Node(9, SELF_ATTN, head=4, side=Side.V),
        Node(9, SELF_ATTN, head=9, side=Side.O, position=-1),
        Node(5, RESID_POST, position=4),
        Node(2, "self_attn.o_proj"),
        Node(2, "mlp.gate_proj"),
    ],
)
def test_str_parse_round_trip(node):
    assert Node.parse(str(node)) == node


def test_str_format_examples():
    assert str(Node(5, SELF_ATTN, head=3, side=Side.Q, position=-1)) == (
        "L5.self_attn.h3.Q@p-1"
    )
    assert str(Node(5, RESID_POST)) == "L5.resid_post"


def test_str_emits_canonical_not_alias():
    assert str(Node(5, "residual")) == "L5.resid_post"


@pytest.mark.parametrize(
    "bad",
    ["5.mlp", "Lx.mlp", "L5", "L5.", "L5.self_attn.h3@pX", "L5.mlp@q3"],
)
def test_parse_rejects_malformed(bad):
    with pytest.raises(ValueError):
        Node.parse(bad)


@pytest.mark.parametrize("bad", ["L5.h3", "L5.Q"])
def test_parse_rejects_suffix_only_module(bad):
    # A string that is only a head/side suffix leaves no module.
    with pytest.raises(ValueError, match="empty module"):
        Node.parse(bad)


# ── Coercion ──────────────────────────────────────────────────────────
def test_coerce_int_is_block_output():
    assert Node.coerce(5) == Node(5, RESID_POST)


def test_coerce_tuple_normalizes_alias():
    assert Node.coerce((5, "residual")) == Node(5, RESID_POST)
    assert Node.coerce((5, "mlp")) == Node(5, MLP)


def test_coerce_string():
    assert Node.coerce("L5.self_attn.h3") == Node(5, SELF_ATTN, head=3)


def test_coerce_node_is_identity():
    node = Node(5, MLP)
    assert Node.coerce(node) is node


def test_coerce_bool_raises():
    with pytest.raises(TypeError, match="must not be a bool"):
        Node.coerce(True)


@pytest.mark.parametrize("bad", [(5,), (5, "mlp", "x"), ("5", "mlp"), (True, "mlp")])
def test_coerce_bad_tuple_raises(bad):
    with pytest.raises(TypeError):
        Node.coerce(bad)


def test_coerce_unsupported_type_raises():
    with pytest.raises(TypeError, match="cannot coerce"):
        Node.coerce(5.0)


# ── Edges ─────────────────────────────────────────────────────────────
def test_edge_str_round_trip():
    edge = Edge(Node(5, SELF_ATTN, head=9), Node(8, MLP))
    assert str(edge) == "L5.self_attn.h9 -> L8.mlp"
    assert Edge.parse(str(edge)) == edge


def test_edge_parse_rejects_malformed():
    with pytest.raises(ValueError, match="<source> -> <dest>"):
        Edge.parse("L5.mlp L8.mlp")


def test_edge_parse_rejects_multiple_arrows():
    with pytest.raises(ValueError, match="one separator"):
        Edge.parse("L1.mlp -> L2.mlp -> L3.mlp")


def test_edge_rejects_non_node_fields():
    with pytest.raises(TypeError, match="must be a Node"):
        Edge("L5.mlp", "L8.mlp")
    with pytest.raises(TypeError, match="must be a Node"):
        Edge(Node(5, MLP), "L8.mlp")


def test_address_like_is_a_runtime_type():
    # Exported AddressLike must be a real union, not a forward-ref string.
    import types

    assert isinstance(AddressLike, types.UnionType)
    assert not isinstance(AddressLike, str)


# ── NodeSet ───────────────────────────────────────────────────────────
def test_nodeset_dedupes_and_preserves_order():
    a, b = Node(5, RESID_POST), Node(6, RESID_POST)
    node_set = NodeSet((a, b, a))
    assert list(node_set) == [a, b]
    assert len(node_set) == 2
    assert a in node_set


def test_nodeset_rejects_non_nodes():
    with pytest.raises(TypeError, match="Node objects"):
        NodeSet((Node(5, RESID_POST), (5, "mlp")))


def test_nodeset_equality():
    assert NodeSet((Node(5, MLP),)) == NodeSet((Node(5, MLP), Node(5, MLP)))


def test_expand_layers():
    node_set = NodeSet.expand_layers(range(3), MLP)
    assert list(node_set) == [Node(0, MLP), Node(1, MLP), Node(2, MLP)]


def test_expand_heads():
    node_set = NodeSet.expand_heads(9, [9, 6, 0])
    assert list(node_set) == [
        Node(9, SELF_ATTN, head=9),
        Node(9, SELF_ATTN, head=6),
        Node(9, SELF_ATTN, head=0),
    ]


def test_expand_positions():
    node_set = NodeSet.expand_positions(Node(5, RESID_POST), [0, 1, -1])
    assert list(node_set) == [
        Node(5, RESID_POST, position=0),
        Node(5, RESID_POST, position=1),
        Node(5, RESID_POST, position=-1),
    ]


def test_product():
    node_set = NodeSet.product([5, 6], [RESID_POST, MLP])
    assert list(node_set) == [
        Node(5, RESID_POST),
        Node(5, MLP),
        Node(6, RESID_POST),
        Node(6, MLP),
    ]


def test_product_invalid_combo_raises():
    with pytest.raises(ValueError, match="attention-only"):
        NodeSet.product([5], [MLP], heads=[0])


def test_all_sides_addressable():
    for side in Side:
        node = Node(5, SELF_ATTN, head=0, side=side)
        assert node.side is side
        assert Node.parse(str(node)) == node


def test_expand_heads_with_side():
    node_set = NodeSet.expand_heads(5, [0, 1], side=Side.V)
    assert list(node_set) == [
        Node(5, SELF_ATTN, head=0, side=Side.V),
        Node(5, SELF_ATTN, head=1, side=Side.V),
    ]


def test_product_attention_axes():
    node_set = NodeSet.product([5], [SELF_ATTN], heads=[0, 1], sides=[Side.Q, Side.K])
    assert list(node_set) == [
        Node(5, SELF_ATTN, head=0, side=Side.Q),
        Node(5, SELF_ATTN, head=0, side=Side.K),
        Node(5, SELF_ATTN, head=1, side=Side.Q),
        Node(5, SELF_ATTN, head=1, side=Side.K),
    ]


# ── Ordering ──────────────────────────────────────────────────────────
def test_nodes_sort_by_layer_then_refinement():
    nodes = [
        Node(2, RESID_POST),
        Node(0, SELF_ATTN, head=3),
        Node(0, SELF_ATTN, head=1),
        Node(0, RESID_POST),
    ]
    assert sorted(nodes) == [
        Node(0, RESID_POST),
        Node(0, SELF_ATTN, head=1),
        Node(0, SELF_ATTN, head=3),
        Node(2, RESID_POST),
    ]


def test_ordering_is_total():
    a, b = Node(0, MLP), Node(1, MLP)
    assert a < b and b > a and a <= a and a >= a


def test_none_refinements_sort_before_concrete():
    assert Node(5, SELF_ATTN) < Node(5, SELF_ATTN, head=0)
    assert Node(5, RESID_POST) < Node(5, RESID_POST, position=-1)


# ── NodeDict ──────────────────────────────────────────────────────────
def test_nodedict_coerces_keys_on_set_and_get():
    d = NodeDict()
    d[5] = "a"
    d[(5, "mlp")] = "b"
    d["L5.self_attn.h3"] = "c"
    assert d[5] == "a"
    assert d[Node(5, RESID_POST)] == "a"
    assert d[(5, "mlp")] == "b"
    assert d[Node(5, MLP)] == "b"
    assert d[Node(5, SELF_ATTN, head=3)] == "c"


def test_nodedict_unifies_spellings():
    # int 5 and "residual" both canonicalize to Node(5, resid_post).
    d = NodeDict({5: "a"})
    d[(5, "residual")] = "b"
    assert len(d) == 1
    assert d[5] == "b"


def test_nodedict_contains_and_get_are_safe():
    d = NodeDict({(5, "mlp"): 1})
    assert (5, "mlp") in d
    assert Node(5, MLP) in d
    assert 99 not in d
    assert "not-an-address" not in d
    assert d.get(5) is None
    assert d.get((5, "mlp")) == 1
    assert d.get("nope", "default") == "default"


def test_nodedict_keys_are_canonical_nodes():
    d = NodeDict({5: 1, (3, "mlp"): 2})
    assert set(d.keys()) == {Node(5, RESID_POST), Node(3, MLP)}


def test_nodedict_update_and_setdefault_coerce():
    d = NodeDict()
    d.update({5: 1, (3, "mlp"): 2})
    assert set(d.keys()) == {Node(5, RESID_POST), Node(3, MLP)}
    assert d[5] == 1
    d.setdefault((3, "mlp"), 99)
    assert d[(3, "mlp")] == 2  # existing key untouched
    d.setdefault(7, 3)
    assert d[7] == 3
    assert isinstance(list(d.keys())[-1], Node)


def test_nodedict_copy_is_nodedict():
    d = NodeDict({5: 1})
    c = d.copy()
    assert isinstance(c, NodeDict)
    assert c[5] == 1


def test_nodedict_merge_operators_coerce():
    d = NodeDict({5: 1})
    d |= {(3, "mlp"): 2}  # in-place merge must coerce
    assert set(d.keys()) == {Node(5, RESID_POST), Node(3, MLP)}
    assert d[(3, "mlp")] == 2

    merged = NodeDict({5: 1}) | {(3, "mlp"): 2}
    assert isinstance(merged, NodeDict)
    assert merged[(3, "mlp")] == 2

    rmerged = {(3, "mlp"): 9} | NodeDict({5: 1})
    assert isinstance(rmerged, NodeDict)
    assert set(rmerged.keys()) == {Node(3, MLP), Node(5, RESID_POST)}


def test_nodedict_ior_does_not_orphan_keys():
    # A merged key must be coerced, not stored raw (else it is unreachable).
    d = NodeDict({5: "original"})
    d |= {5: "merged"}
    assert len(d) == 1
    assert d[5] == "merged"
    assert all(isinstance(k, Node) for k in d)


def test_nodedict_pop_coerces():
    d = NodeDict({(3, "mlp"): 2, 5: 1})
    assert d.pop(5) == 1
    assert d.pop((3, "mlp")) == 2
    assert d.pop(99, "default") == "default"
    assert d.pop("not-an-address", "d") == "d"
    with pytest.raises(KeyError):
        d.pop(7)


def test_nodedict_pickle_and_torch_save_round_trip():
    import io

    import torch

    d = NodeDict({5: 1, (3, "mlp"): 2, Node(2, SELF_ATTN, head=1): 3})
    assert pickle.loads(pickle.dumps(d)) == d
    buffer = io.BytesIO()
    torch.save(d, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False)
    assert loaded == d
    # Coercion survives the round-trip.
    assert loaded[5] == 1
    assert loaded[(3, "mlp")] == 2


# ── Torch-Free Import ─────────────────────────────────────────────────
def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = str(SRC_ROOT)
    if env.get("PYTHONPATH"):
        pythonpath = f"{pythonpath}:{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath
    return env


def test_importing_nodes_does_not_import_torch():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import murano.nodes; print('torch' in sys.modules)",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env=_subprocess_env(),
    )
    assert completed.stdout.strip() == "False"
