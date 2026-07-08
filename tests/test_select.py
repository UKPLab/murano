"""Tests for SelectComponents and the discover-then-patch composition.

SelectComponents turns a per-component attribution readout into a
ComponentSelection, and Patch/PathPatch/Ablate can read that selection at run
time. The unit tests pin the ranking, filtering, and cutoff logic on a
hand-built LogitAttributionResult (weight-independent); the composition tests run
the tiny ``murano_model`` fixture on CPU to prove attribute-then-patch is one
pipeline and that a selection resolves to the same result as the explicit target
list it names.
"""

from __future__ import annotations

import pytest
import torch

from murano import Pipeline, keys
from murano.artifacts import ComponentSelection
from murano.dataset import CleanCorruptDataset
from murano.nodes import MLP, SELF_ATTN, Node
from murano.results import Results
from murano.steps.ablate import Ablate
from murano.steps.logit_attribution import LogitAttribution, LogitAttributionResult
from murano.steps.paired import LoadPaired
from murano.steps.patch import Patch
from murano.steps.path_patch import PathPatch
from murano.steps.select import SelectComponents

# Equal token length per pair so patch positions align (see test_patch.py).
CLEAN = ["hello world", "good world"]
CORRUPT = ["good world", "bad world"]


def _attribution() -> LogitAttributionResult:
    """A small attribution result: two heads and one MLP with known scores."""
    return LogitAttributionResult(
        contributions={
            Node(0, SELF_ATTN, head=0): 2.0,
            Node(1, SELF_ATTN, head=1): -3.0,
            Node(0, MLP): 0.5,
        },
        embed_contribution=0.0,
        other_contribution=0.0,
        target="logit_diff",
        total=0.0,
        completeness_error=0.0,
    )


def _with_attribution() -> Results:
    results = Results()
    results[keys.LOGIT_ATTRIBUTION] = _attribution()
    return results


# ── SelectComponents ranking / filtering ──────────────────────────────


class TestSelectComponents:
    def test_top_k_by_abs_ranks_by_magnitude(self):
        out = SelectComponents(top_k=2)(_with_attribution())
        selection = out[keys.SELECTION]
        assert isinstance(selection, ComponentSelection)
        # |-3| > |2| > |0.5|, so the two strongest are the layer-1 head then the
        # layer-0 head, best first.
        assert selection.nodes == [
            Node(1, SELF_ATTN, head=1),
            Node(0, SELF_ATTN, head=0),
        ]

    def test_signed_prefers_most_positive(self):
        out = SelectComponents(top_k=1, by="signed")(_with_attribution())
        assert out[keys.SELECTION].nodes == [Node(0, SELF_ATTN, head=0)]

    def test_negative_prefers_most_negative(self):
        out = SelectComponents(top_k=1, by="negative")(_with_attribution())
        assert out[keys.SELECTION].nodes == [Node(1, SELF_ATTN, head=1)]

    def test_threshold_abs_keeps_everything_past_cutoff(self):
        out = SelectComponents(threshold=1.0)(_with_attribution())
        nodes = out[keys.SELECTION].nodes
        assert set(nodes) == {Node(0, SELF_ATTN, head=0), Node(1, SELF_ATTN, head=1)}

    def test_modules_filter_keeps_heads_only(self):
        # Filtering to self_attn drops the MLP, so a downstream Patch sees a single
        # (per-head) mode. "attn_out" is an accepted alias for the canonical name.
        out = SelectComponents(top_k=5, modules="attn_out")(_with_attribution())
        assert all(node.module == SELF_ATTN for node in out[keys.SELECTION].nodes)
        assert Node(0, MLP) not in out[keys.SELECTION].nodes

    def test_scores_recorded_on_selection(self):
        selection = SelectComponents(top_k=2)(_with_attribution())[keys.SELECTION]
        assert selection.scores[Node(1, SELF_ATTN, head=1)] == -3.0

    def test_empty_selection_raises(self):
        with pytest.raises(ValueError, match="chose no component"):
            SelectComponents(threshold=100.0)(_with_attribution())

    def test_writes_selection_type(self):
        step = SelectComponents(top_k=1)
        assert step.reads == [keys.LOGIT_ATTRIBUTION]
        assert step.writes == [keys.SELECTION]
        assert step.write_types == {keys.SELECTION: ComponentSelection}

    def test_constructor_guards(self):
        with pytest.raises(ValueError, match="exactly one"):
            SelectComponents()
        with pytest.raises(ValueError, match="exactly one"):
            SelectComponents(top_k=1, threshold=0.5)
        with pytest.raises(ValueError, match="top_k must be positive"):
            SelectComponents(top_k=0)
        with pytest.raises(ValueError, match="by must be one of"):
            SelectComponents(top_k=1, by="huge")


# ── Discover-then-patch composition ────────────────────────────────────


def _loaded(clean=CLEAN, corrupt=CORRUPT, correct=5, incorrect=6):
    ds = CleanCorruptDataset(
        clean=clean, corrupt=corrupt, correct=correct, incorrect=incorrect
    )
    return LoadPaired(ds)(Results())


class TestDiscoverThenPatch:
    def test_patch_targets_key_matches_explicit_targets(self, murano_model):
        # A selection resolves at run time to the same sites the explicit list
        # names, so the patched logits are identical.
        targets = [Node(0, SELF_ATTN, head=0)]
        selection = ComponentSelection(nodes=list(targets))

        explicit = Patch(murano_model, targets)(_loaded())[keys.PATCHED_LOGITS]

        results = _loaded()
        results[keys.SELECTION] = selection
        via_key = Patch(murano_model, targets_key=keys.SELECTION)(results)[
            keys.PATCHED_LOGITS
        ]
        assert torch.equal(explicit, via_key)

    def test_patch_targets_key_declares_the_read(self, murano_model):
        step = Patch(murano_model, targets_key=keys.SELECTION)
        assert keys.SELECTION in step.reads
        assert step.read_types[keys.SELECTION] is ComponentSelection

    def test_missing_selection_fails_preflight(self, murano_model):
        pipe = Pipeline(
            [
                LoadPaired(CleanCorruptDataset(clean=CLEAN, corrupt=CORRUPT)),
                Patch(murano_model, targets_key=keys.SELECTION),
            ]
        )
        with pytest.raises(KeyError, match=keys.SELECTION):
            pipe.validate()

    def test_full_attribute_then_patch_arc_validates(self, murano_model):
        produced = Pipeline(
            [
                LoadPaired(
                    CleanCorruptDataset(
                        clean=CLEAN, corrupt=CORRUPT, correct=5, incorrect=6
                    )
                ),
                LogitAttribution(murano_model, correct=5, incorrect=6),
                SelectComponents(top_k=2, modules="self_attn"),
                Patch(murano_model, targets_key=keys.SELECTION),
            ]
        ).validate()
        assert keys.PATCHED_LOGITS in produced

    def test_full_attribute_then_patch_arc_runs(self, murano_model):
        out = Pipeline(
            [
                LoadPaired(
                    CleanCorruptDataset(
                        clean=CLEAN, corrupt=CORRUPT, correct=5, incorrect=6
                    )
                ),
                LogitAttribution(murano_model, correct=5, incorrect=6),
                SelectComponents(top_k=2, modules="self_attn"),
                Patch(murano_model, targets_key=keys.SELECTION),
            ]
        ).run()
        assert torch.isfinite(out[keys.PATCHED_LOGITS]).all()
        # The selection is per-head, so Patch inferred per-head mode from it.
        assert out[keys.SELECTION].nodes and all(
            node.head is not None for node in out[keys.SELECTION].nodes
        )


class TestDiscoverThenPathPatch:
    def test_senders_key_matches_explicit_senders(self, murano_model):
        senders = [Node(0, SELF_ATTN, head=0)]
        selection = ComponentSelection(nodes=list(senders))

        explicit = PathPatch(murano_model, senders)(_loaded())[keys.PATH_PATCHED_LOGITS]

        results = _loaded()
        results[keys.SELECTION] = selection
        via_key = PathPatch(murano_model, senders_key=keys.SELECTION)(results)[
            keys.PATH_PATCHED_LOGITS
        ]
        assert torch.equal(explicit, via_key)

    def test_senders_key_declares_the_read(self, murano_model):
        step = PathPatch(murano_model, senders_key=keys.SELECTION)
        assert keys.SELECTION in step.reads
        assert step.read_types[keys.SELECTION] is ComponentSelection

    def test_missing_selection_fails_preflight(self, murano_model):
        pipe = Pipeline(
            [
                LoadPaired(CleanCorruptDataset(clean=CLEAN, corrupt=CORRUPT)),
                PathPatch(murano_model, senders_key=keys.SELECTION),
            ]
        )
        with pytest.raises(KeyError, match=keys.SELECTION):
            pipe.validate()

    def test_empty_selection_at_runtime_raises(self, murano_model):
        results = _loaded()
        results[keys.SELECTION] = ComponentSelection(nodes=[])
        with pytest.raises(ValueError, match="at least one sender"):
            PathPatch(murano_model, senders_key=keys.SELECTION)(results)


# ── Deferred-target constructor guards ─────────────────────────────────


class TestDeferredTargetGuards:
    def test_ablate_requires_exactly_one_target_source(self, murano_model):
        with pytest.raises(ValueError, match="exactly one"):
            Ablate(murano_model)
        with pytest.raises(ValueError, match="exactly one"):
            Ablate(murano_model, 0, targets_key=keys.SELECTION)

    def test_patch_requires_exactly_one_target_source(self, murano_model):
        with pytest.raises(ValueError, match="exactly one"):
            Patch(murano_model)
        with pytest.raises(ValueError, match="exactly one"):
            Patch(murano_model, 0, targets_key=keys.SELECTION)

    def test_means_with_targets_key_rejected(self, murano_model):
        with pytest.raises(ValueError, match="means="):
            Ablate(
                murano_model,
                targets_key=keys.SELECTION,
                method="mean",
                means={Node(0, "residual"): torch.zeros(murano_model.d_model)},
            )

    def test_pathpatch_requires_exactly_one_sender_source(self, murano_model):
        with pytest.raises(ValueError, match="exactly one"):
            PathPatch(murano_model)
        with pytest.raises(ValueError, match="exactly one"):
            PathPatch(murano_model, 0, senders_key=keys.SELECTION)

    def test_pathpatch_edge_with_senders_key_rejected(self, murano_model):
        from murano.nodes import Edge

        edge = Edge(
            Node(0, SELF_ATTN, head=0), Node(murano_model.n_layers - 1, "residual")
        )
        # An Edge is a senders= form, so combining it with senders_key= trips the
        # exactly-one guard.
        with pytest.raises(ValueError, match="exactly one"):
            PathPatch(murano_model, edge, senders_key=keys.SELECTION)
