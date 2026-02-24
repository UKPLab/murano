"""
TDD test suite for MetricComputationLenses.
Tests are written before implementation to drive design.

"""

import pytest
import torch
import torch.nn.functional as F

from murano.model import MuranoModel
from murano.lenses.metric_lenses import (
    CrossEntropyLossLens,
    AccuracyLens,
    ComparisonComputationLens,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def murano_model():
    """Load a real GPT-2 model once for the whole test module."""
    return MuranoModel("gpt2")


@pytest.fixture(scope="module")
def forward_artifact(murano_model):
    """
    Perform a single traced forward pass through GPT-2 and build an artifact
    that contains 'final_logits' and 'target_ids' – the two keys expected by
    both CrossEntropyLossLens and AccuracyLens.

    Target IDs are the input IDs shifted left by one position (next-token
    prediction), matching the standard causal-LM convention.
    """
    model = murano_model.model
    prompt = "The quick brown fox jumps"

    with model.trace(prompt) as _:
        raw_input_ids = model.embed_tokens.input.save()
        final_logits = model.output.logits.save()

    # final_logits: [1, seq_len, vocab_size]
    # raw_input_ids: [1, seq_len]
    logits = final_logits
    input_ids = raw_input_ids

    shifted_logits = logits[:, :-1, :].contiguous()  # [1, seq_len-1, vocab_size]
    shifted_targets = input_ids[:, 1:].contiguous()  # [1, seq_len-1]

    return {
        "final_logits": shifted_logits,
        "target_ids": shifted_targets,
    }


# ---------------------------------------------------------------------------
# CrossEntropyLossLens tests
# ---------------------------------------------------------------------------


class TestCrossEntropyLossLens:
    def test_scalar_loss_present_in_artifact(self, forward_artifact):
        """After processing, the artifact must contain a 'loss' key."""
        lens = CrossEntropyLossLens(reduction="mean")
        artifact = dict(forward_artifact)  # this si a shallow copy
        enriched = lens.process(artifact)

        assert "loss" in enriched, "'loss' key not found in enriched artifact"

    def test_scalar_loss_is_positive_float(self, forward_artifact):
        """Cross-entropy loss must be a positive scalar tensor."""
        lens = CrossEntropyLossLens(reduction="mean")
        enriched = lens.process(dict(forward_artifact))

        loss = enriched["loss"]
        assert isinstance(loss, torch.Tensor), "loss should be a torch.Tensor"
        assert loss.ndim == 0, "mean-reduced loss should be a scalar (0-dim tensor)"
        assert loss.item() > 0.0, "Cross-entropy loss must be positive"

    def test_per_token_loss_shape(self, forward_artifact):
        """With reduction='none', loss shape must match (batch, seq_len)."""
        lens = CrossEntropyLossLens(reduction="none")
        enriched = lens.process(dict(forward_artifact))

        loss = enriched["loss"]
        batch, seq_len_minus_1 = forward_artifact["target_ids"].shape
        assert loss.shape == (batch, seq_len_minus_1), (
            f"Expected per-token loss shape {(batch, seq_len_minus_1)}, got {loss.shape}"
        )

    def test_loss_matches_pytorch_reference(self, forward_artifact):
        """Computed loss must match a direct PyTorch F.cross_entropy call."""
        lens = CrossEntropyLossLens(reduction="mean")
        enriched = lens.process(dict(forward_artifact))

        logits = forward_artifact["final_logits"]  # [B, S, V]
        targets = forward_artifact["target_ids"]  # [B, S]
        B, S, V = logits.shape
        reference_loss = F.cross_entropy(
            logits.reshape(B * S, V), targets.reshape(B * S)
        )

        assert abs(enriched["loss"].item() - reference_loss.item()) < 1e-5, (
            "Computed loss diverges from PyTorch reference"
        )

    def test_does_not_overwrite_existing_keys(self, forward_artifact):
        """The lens must not overwrite pre-existing keys in the artifact."""
        artifact = dict(forward_artifact)
        artifact["existing_key"] = "sentinel"
        lens = CrossEntropyLossLens(reduction="mean")
        enriched = lens.process(artifact)

        assert enriched["existing_key"] == "sentinel"


# ---------------------------------------------------------------------------
# AccuracyLens tests
# ---------------------------------------------------------------------------


class TestAccuracyLens:
    def test_accuracy_present_in_artifact(self, forward_artifact):
        """After processing, the artifact must contain an 'accuracy' key."""
        lens = AccuracyLens()
        enriched = lens.process(dict(forward_artifact))

        assert "accuracy" in enriched, "'accuracy' key not found in enriched artifact"

    def test_accuracy_is_float_in_unit_interval(self, forward_artifact):
        """Token-level accuracy must be in [0.0, 1.0]."""
        lens = AccuracyLens()
        enriched = lens.process(dict(forward_artifact))

        acc = enriched["accuracy"]
        assert isinstance(acc, float), "accuracy should be a plain Python float"
        assert 0.0 <= acc <= 1.0, f"accuracy {acc} is out of [0, 1]"

    def test_accuracy_matches_manual_calculation(self, forward_artifact):
        """Verify accuracy equals the fraction of argmax matches."""
        logits = forward_artifact["final_logits"]  # [B, S, V]
        targets = forward_artifact["target_ids"]  # [B, S]

        predicted = logits.argmax(dim=-1)  # [B, S]
        expected_acc = (predicted == targets).float().mean().item()

        lens = AccuracyLens()
        enriched = lens.process(dict(forward_artifact))

        assert abs(enriched["accuracy"] - expected_acc) < 1e-6, (
            f"AccuracyLens returned {enriched['accuracy']}, expected {expected_acc}"
        )

    def test_perfect_accuracy_on_synthetic_data(self):
        """On perfectly matching predictions, accuracy must be 1.0."""
        vocab_size = 10
        seq_len = 5
        # Create logits where argmax == target
        targets = torch.tensor([[2, 5, 7, 1, 9]])  # [1, 5]
        logits = torch.zeros(1, seq_len, vocab_size)
        for t_idx, t in enumerate(targets[0]):
            logits[0, t_idx, t] = 10.0  # high logit

        artifact = {"final_logits": logits, "target_ids": targets}
        lens = AccuracyLens()
        enriched = lens.process(artifact)

        assert enriched["accuracy"] == pytest.approx(1.0)

    def test_zero_accuracy_on_synthetic_data(self):
        """On completely wrong predictions, accuracy must be 0.0."""
        vocab_size = 10
        seq_len = 5
        targets = torch.tensor([[0, 0, 0, 0, 0]])
        logits = torch.zeros(1, seq_len, vocab_size)
        for t_idx in range(seq_len):
            logits[0, t_idx, 1] = 10.0  # Always predicts token 1

        artifact = {"final_logits": logits, "target_ids": targets}
        lens = AccuracyLens()
        enriched = lens.process(artifact)

        assert enriched["accuracy"] == pytest.approx(0.0)

    def test_does_not_overwrite_existing_keys(self, forward_artifact):
        """The lens must not overwrite pre-existing keys in the artifact."""
        artifact = dict(forward_artifact)
        artifact["sentinel"] = 42
        lens = AccuracyLens()
        enriched = lens.process(artifact)

        assert enriched["sentinel"] == 42


# ---------------------------------------------------------------------------
# ComparisonComputationLens tests
# ---------------------------------------------------------------------------


class TestComparisonComputationLens:
    """Tests for ComparisonComputationLens with 'difference' and 'cosine_similarity' modes."""

    @pytest.fixture
    def base_artifact(self):
        """Simple artifact with two 2-D tensors to compare."""
        torch.manual_seed(0)
        return {
            "clean_acts": torch.randn(4, 16),
            "corrupt_acts": torch.randn(4, 16),
        }

    # -- difference ----------------------------------------------------------

    def test_difference_output_present(self, base_artifact):
        lens = ComparisonComputationLens(
            key_a="clean_acts",
            key_b="corrupt_acts",
            output_key="delta",
            comparison_type="difference",
        )
        enriched = lens.process(dict(base_artifact))
        assert "delta" in enriched

    def test_difference_shape_preserved(self, base_artifact):
        lens = ComparisonComputationLens(
            key_a="clean_acts",
            key_b="corrupt_acts",
            output_key="delta",
            comparison_type="difference",
        )
        enriched = lens.process(dict(base_artifact))
        assert enriched["delta"].shape == base_artifact["clean_acts"].shape

    def test_difference_values_correct(self, base_artifact):
        lens = ComparisonComputationLens(
            key_a="clean_acts",
            key_b="corrupt_acts",
            output_key="delta",
            comparison_type="difference",
        )
        enriched = lens.process(dict(base_artifact))
        expected = base_artifact["clean_acts"] - base_artifact["corrupt_acts"]
        assert torch.allclose(enriched["delta"], expected), (
            "Difference tensor does not match element-wise subtraction"
        )

    # -- cosine_similarity ----------------------------------------------------

    def test_cosine_similarity_output_present(self, base_artifact):
        lens = ComparisonComputationLens(
            key_a="clean_acts",
            key_b="corrupt_acts",
            output_key="cos_sim",
            comparison_type="cosine_similarity",
        )
        enriched = lens.process(dict(base_artifact))
        assert "cos_sim" in enriched

    def test_cosine_similarity_values_in_range(self, base_artifact):
        """Cosine similarity row-wise must lie in [-1, 1]."""
        lens = ComparisonComputationLens(
            key_a="clean_acts",
            key_b="corrupt_acts",
            output_key="cos_sim",
            comparison_type="cosine_similarity",
        )
        enriched = lens.process(dict(base_artifact))
        cos = enriched["cos_sim"]
        assert (cos >= -1.0 - 1e-5).all() and (cos <= 1.0 + 1e-5).all(), (
            "Cosine similarity values out of [-1, 1]"
        )

    def test_cosine_similarity_shape(self, base_artifact):
        """Row-wise cosine similarity of (4, 16) tensors should yield shape (4,)."""
        lens = ComparisonComputationLens(
            key_a="clean_acts",
            key_b="corrupt_acts",
            output_key="cos_sim",
            comparison_type="cosine_similarity",
        )
        enriched = lens.process(dict(base_artifact))
        assert enriched["cos_sim"].shape == (base_artifact["clean_acts"].shape[0],)

    def test_cosine_similarity_identical_vectors_is_one(self):
        """Cosine similarity of a tensor with itself must be 1.0 everywhere."""
        acts = torch.randn(3, 8)
        artifact = {"a": acts, "b": acts}
        lens = ComparisonComputationLens(
            key_a="a", key_b="b", output_key="sim", comparison_type="cosine_similarity"
        )
        enriched = lens.process(artifact)
        assert torch.allclose(enriched["sim"], torch.ones(3), atol=1e-5)

    # -- immutability ---------------------------------------------------------

    def test_source_keys_not_overwritten(self, base_artifact):
        """The lens must not mutate key_a or key_b in the artifact."""
        clean_original = base_artifact["clean_acts"].clone()
        corrupt_original = base_artifact["corrupt_acts"].clone()
        lens = ComparisonComputationLens(
            key_a="clean_acts",
            key_b="corrupt_acts",
            output_key="delta",
            comparison_type="difference",
        )
        enriched = lens.process(dict(base_artifact))
        assert torch.equal(enriched["clean_acts"], clean_original)
        assert torch.equal(enriched["corrupt_acts"], corrupt_original)

    # -- invalid type --------------------------------------------------------

    def test_invalid_comparison_type_raises(self, base_artifact):
        lens = ComparisonComputationLens(
            key_a="clean_acts",
            key_b="corrupt_acts",
            output_key="out",
            comparison_type="invalid_mode",
        )
        with pytest.raises(ValueError, match="Unsupported comparison_type"):
            lens.process(dict(base_artifact))
