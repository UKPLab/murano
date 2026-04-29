"""Tests for metric computation steps.

Adapted from the legacy ``test_metric_lenses.py`` blueprint.
Uses ``murano.results.Results`` instead of flat artifact dicts.
"""

import pytest
import torch
import torch.nn.functional as F

from murano.results import Results
from murano.steps.metrics import (
    CrossEntropyLossStep,
    AccuracyStep,
    ComparisonComputationStep,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def logits_and_targets():
    """Synthetic logits and target IDs for a single batch."""
    torch.manual_seed(42)
    B, S, V = 2, 5, 10
    logits = torch.randn(B, S, V)
    targets = torch.randint(0, V, (B, S))
    return logits, targets


@pytest.fixture
def results_with_logits(logits_and_targets):
    """Results object pre-populated with logits and targets."""
    logits, targets = logits_and_targets
    r = Results()
    r["final_logits"] = logits
    r["target_ids"] = targets
    return r


# ---------------------------------------------------------------------------
# CrossEntropyLossStep tests
# ---------------------------------------------------------------------------


class TestCrossEntropyLossStep:
    def test_scalar_loss_present_in_results(self, results_with_logits):
        """After processing, results must contain the loss key."""
        step = CrossEntropyLossStep(reduction="mean")
        results = step(results_with_logits)

        assert "loss" in results, "'loss' key not found in results"

    def test_scalar_loss_is_positive_float(self, results_with_logits):
        """Cross-entropy loss must be a positive scalar tensor."""
        step = CrossEntropyLossStep(reduction="mean")
        results = step(results_with_logits)

        loss = results["loss"]
        assert isinstance(loss, torch.Tensor), "loss should be a torch.Tensor"
        assert loss.ndim == 0, "mean-reduced loss should be a scalar (0-dim tensor)"
        assert loss.item() > 0.0, "Cross-entropy loss must be positive"

    def test_per_token_loss_shape(self, results_with_logits, logits_and_targets):
        """With reduction='none', loss shape must match (batch, seq_len)."""
        _, targets = logits_and_targets
        step = CrossEntropyLossStep(reduction="none")
        results = step(results_with_logits)

        loss = results["loss"]
        assert loss.shape == targets.shape, (
            f"Expected per-token loss shape {targets.shape}, got {loss.shape}"
        )

    def test_loss_matches_pytorch_reference(self, results_with_logits, logits_and_targets):
        """Computed loss must match a direct PyTorch F.cross_entropy call."""
        logits, targets = logits_and_targets
        step = CrossEntropyLossStep(reduction="mean")
        results = step(results_with_logits)

        B, S, V = logits.shape
        reference_loss = F.cross_entropy(
            logits.reshape(B * S, V), targets.reshape(B * S)
        )

        assert abs(results["loss"].item() - reference_loss.item()) < 1e-5, (
            "Computed loss diverges from PyTorch reference"
        )

    def test_does_not_overwrite_existing_keys(self, results_with_logits):
        """The step must not overwrite pre-existing keys in results."""
        results_with_logits["existing_key"] = "sentinel"
        step = CrossEntropyLossStep(reduction="mean")
        results = step(results_with_logits)

        assert results["existing_key"] == "sentinel"

    def test_custom_keys(self):
        """Step must work with user-specified input/output keys."""
        logits = torch.randn(1, 3, 5)
        targets = torch.tensor([[0, 1, 2]])
        r = Results()
        r["my_logits"] = logits
        r["my_targets"] = targets

        step = CrossEntropyLossStep(
            logits_key="my_logits",
            targets_key="my_targets",
            output_key="my_loss",
            reduction="mean",
        )
        results = step(r)
        assert "my_loss" in results
        assert isinstance(results["my_loss"], torch.Tensor)

    def test_sum_reduction(self, results_with_logits):
        """Sum reduction must produce a scalar."""
        step = CrossEntropyLossStep(reduction="sum")
        results = step(results_with_logits)
        assert results["loss"].ndim == 0


# ---------------------------------------------------------------------------
# AccuracyStep tests
# ---------------------------------------------------------------------------


class TestAccuracyStep:
    def test_accuracy_present_in_results(self, results_with_logits):
        """After processing, results must contain the accuracy key."""
        step = AccuracyStep()
        results = step(results_with_logits)

        assert "accuracy" in results, "'accuracy' key not found in results"

    def test_accuracy_is_float_in_unit_interval(self, results_with_logits):
        """Token-level accuracy must be in [0.0, 1.0]."""
        step = AccuracyStep()
        results = step(results_with_logits)

        acc = results["accuracy"]
        assert isinstance(acc, float), "accuracy should be a plain Python float"
        assert 0.0 <= acc <= 1.0, f"accuracy {acc} is out of [0, 1]"

    def test_accuracy_matches_manual_calculation(self, results_with_logits, logits_and_targets):
        """Verify accuracy equals the fraction of argmax matches."""
        logits, targets = logits_and_targets

        predicted = logits.argmax(dim=-1)
        expected_acc = (predicted == targets).float().mean().item()

        step = AccuracyStep()
        results = step(results_with_logits)

        assert abs(results["accuracy"] - expected_acc) < 1e-6, (
            f"AccuracyStep returned {results['accuracy']}, expected {expected_acc}"
        )

    def test_perfect_accuracy_on_synthetic_data(self):
        """On perfectly matching predictions, accuracy must be 1.0."""
        vocab_size = 10
        seq_len = 5
        targets = torch.tensor([[2, 5, 7, 1, 9]])
        logits = torch.zeros(1, seq_len, vocab_size)
        for t_idx, t in enumerate(targets[0]):
            logits[0, t_idx, t] = 10.0

        r = Results()
        r["final_logits"] = logits
        r["target_ids"] = targets

        step = AccuracyStep()
        results = step(r)
        assert results["accuracy"] == pytest.approx(1.0)

    def test_zero_accuracy_on_synthetic_data(self):
        """On completely wrong predictions, accuracy must be 0.0."""
        vocab_size = 10
        seq_len = 5
        targets = torch.tensor([[0, 0, 0, 0, 0]])
        logits = torch.zeros(1, seq_len, vocab_size)
        for t_idx in range(seq_len):
            logits[0, t_idx, 1] = 10.0

        r = Results()
        r["final_logits"] = logits
        r["target_ids"] = targets

        step = AccuracyStep()
        results = step(r)
        assert results["accuracy"] == pytest.approx(0.0)

    def test_does_not_overwrite_existing_keys(self, results_with_logits):
        """The step must not overwrite pre-existing keys in results."""
        results_with_logits["sentinel"] = 42
        step = AccuracyStep()
        results = step(results_with_logits)
        assert results["sentinel"] == 42

    def test_custom_keys(self):
        """Step must work with user-specified input/output keys."""
        logits = torch.randn(1, 3, 5)
        targets = torch.tensor([[0, 1, 2]])
        r = Results()
        r["my_logits"] = logits
        r["my_targets"] = targets

        step = AccuracyStep(
            logits_key="my_logits",
            targets_key="my_targets",
            output_key="my_acc",
        )
        results = step(r)
        assert "my_acc" in results
        assert isinstance(results["my_acc"], float)


# ---------------------------------------------------------------------------
# ComparisonComputationStep tests
# ---------------------------------------------------------------------------


class TestComparisonComputationStep:
    """Tests for ComparisonComputationStep with 'difference' and 'cosine_similarity' modes."""

    @pytest.fixture
    def base_results(self):
        """Results with two 2-D tensors to compare."""
        torch.manual_seed(0)
        r = Results()
        r["clean_acts"] = torch.randn(4, 16)
        r["corrupt_acts"] = torch.randn(4, 16)
        return r

    # -- difference ----------------------------------------------------------

    def test_difference_output_present(self, base_results):
        step = ComparisonComputationStep(
            key_a="clean_acts",
            key_b="corrupt_acts",
            output_key="delta",
            comparison_type="difference",
        )
        results = step(base_results)
        assert "delta" in results

    def test_difference_shape_preserved(self, base_results):
        step = ComparisonComputationStep(
            key_a="clean_acts",
            key_b="corrupt_acts",
            output_key="delta",
            comparison_type="difference",
        )
        results = step(base_results)
        assert results["delta"].shape == base_results["clean_acts"].shape

    def test_difference_values_correct(self, base_results):
        step = ComparisonComputationStep(
            key_a="clean_acts",
            key_b="corrupt_acts",
            output_key="delta",
            comparison_type="difference",
        )
        results = step(base_results)
        expected = base_results["clean_acts"] - base_results["corrupt_acts"]
        assert torch.allclose(results["delta"], expected), (
            "Difference tensor does not match element-wise subtraction"
        )

    # -- cosine_similarity ----------------------------------------------------

    def test_cosine_similarity_output_present(self, base_results):
        step = ComparisonComputationStep(
            key_a="clean_acts",
            key_b="corrupt_acts",
            output_key="cos_sim",
            comparison_type="cosine_similarity",
        )
        results = step(base_results)
        assert "cos_sim" in results

    def test_cosine_similarity_values_in_range(self, base_results):
        """Cosine similarity row-wise must lie in [-1, 1]."""
        step = ComparisonComputationStep(
            key_a="clean_acts",
            key_b="corrupt_acts",
            output_key="cos_sim",
            comparison_type="cosine_similarity",
        )
        results = step(base_results)
        cos = results["cos_sim"]
        assert (cos >= -1.0 - 1e-5).all() and (cos <= 1.0 + 1e-5).all(), (
            "Cosine similarity values out of [-1, 1]"
        )

    def test_cosine_similarity_shape(self, base_results):
        """Row-wise cosine similarity of (4, 16) tensors should yield shape (4,)."""
        step = ComparisonComputationStep(
            key_a="clean_acts",
            key_b="corrupt_acts",
            output_key="cos_sim",
            comparison_type="cosine_similarity",
        )
        results = step(base_results)
        assert results["cos_sim"].shape == (base_results["clean_acts"].shape[0],)

    def test_cosine_similarity_identical_vectors_is_one(self):
        """Cosine similarity of a tensor with itself must be 1.0 everywhere."""
        acts = torch.randn(3, 8)
        r = Results()
        r["a"] = acts
        r["b"] = acts

        step = ComparisonComputationStep(
            key_a="a", key_b="b", output_key="sim", comparison_type="cosine_similarity"
        )
        results = step(r)
        assert torch.allclose(results["sim"], torch.ones(3), atol=1e-5)

    # -- immutability ---------------------------------------------------------

    def test_source_keys_not_overwritten(self, base_results):
        """The step must not mutate key_a or key_b in results."""
        clean_original = base_results["clean_acts"].clone()
        corrupt_original = base_results["corrupt_acts"].clone()

        step = ComparisonComputationStep(
            key_a="clean_acts",
            key_b="corrupt_acts",
            output_key="delta",
            comparison_type="difference",
        )
        results = step(base_results)
        assert torch.equal(results["clean_acts"], clean_original)
        assert torch.equal(results["corrupt_acts"], corrupt_original)

    # -- invalid type --------------------------------------------------------

    def test_invalid_comparison_type_raises(self, base_results):
        step = ComparisonComputationStep(
            key_a="clean_acts",
            key_b="corrupt_acts",
            output_key="out",
            comparison_type="invalid_mode",
        )
        with pytest.raises(ValueError, match="Unsupported comparison_type"):
            step(base_results)

    # -- custom keys ---------------------------------------------------------

    def test_custom_keys(self):
        """Step must work with user-specified keys."""
        a = torch.randn(2, 4)
        b = torch.randn(2, 4)
        r = Results()
        r["tensor_a"] = a
        r["tensor_b"] = b

        step = ComparisonComputationStep(
            key_a="tensor_a",
            key_b="tensor_b",
            output_key="result",
            comparison_type="difference",
        )
        results = step(r)
        assert "result" in results
        assert torch.allclose(results["result"], a - b)