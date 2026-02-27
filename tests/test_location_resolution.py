"""
Essential tests for semantic token positioning.
"""

import pytest
import sys
from pathlib import Path
import torch

# Add src directory to path
_src_path = Path(__file__).parent.parent / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from murano.utils import Location, steering_vector_to_activation_dataset


# ==============================================================================
# LOCATION CLASS - ESSENTIAL TESTS (12 tests)
# ==============================================================================


class TestLocationEssentials:
    """Essential tests for Location class with semantic keywords."""

    # --- Initialization & Validation (5 tests) ---

    def test_valid_keywords_accepted(self):
        """All valid keywords should be accepted."""
        for keyword in ["prompt", "generation", "all", "last"]:
            loc = Location(layers=[6], modules=["mlp"], token_pos=keyword)
            assert loc.token_pos == keyword

    def test_invalid_keyword_rejected(self):
        """Invalid keywords should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid token_pos keyword"):
            Location(layers=[6], modules=["mlp"], token_pos="invalid")

    def test_backward_compatibility_integer_list(self):
        """Integer lists should still work."""
        loc = Location(layers=[6], modules=["mlp"], token_pos=[0, -1])
        assert loc.token_pos == [0, -1]

    def test_backward_compatibility_none(self):
        """None should still work."""
        loc = Location(layers=[6], modules=["mlp"], token_pos=None)
        assert loc.token_pos is None

    def test_backward_compatibility_single_integer(self):
        """Single integer should be wrapped in list."""
        loc = Location(layers=[6], modules=["mlp"], token_pos=5)
        assert loc.token_pos == [5]

    # --- resolve_indices Method (4 tests) ---

    def test_resolve_prompt_keyword(self):
        """'prompt' should resolve to slice(0, prompt_len)."""
        loc = Location(layers=[6], modules=["mlp"], token_pos="prompt")
        result = loc.resolve_indices(total_seq_len=20, prompt_len=10)
        assert result == slice(0, 10)

    def test_resolve_generation_keyword(self):
        """'generation' should resolve to slice(prompt_len, total_seq_len)."""
        loc = Location(layers=[6], modules=["mlp"], token_pos="generation")
        result = loc.resolve_indices(total_seq_len=20, prompt_len=10)
        assert result == slice(10, 20)

    def test_resolve_all_keyword(self):
        """'all' should resolve to slice(0, total_seq_len)."""
        loc = Location(layers=[6], modules=["mlp"], token_pos="all")
        result = loc.resolve_indices(total_seq_len=20, prompt_len=10)
        assert result == slice(0, 20)

    def test_resolve_last_keyword(self):
        """'last' should resolve to last prompt token."""
        loc = Location(layers=[6], modules=["mlp"], token_pos="last")
        result = loc.resolve_indices(total_seq_len=20, prompt_len=10)
        assert result == slice(9, 10)

    # --- Phase Detection (3 tests) ---

    def test_prompt_phase_detection(self):
        """'prompt' should apply only to prompt phase."""
        loc = Location(layers=[6], modules=["mlp"], token_pos="prompt")
        assert loc.applies_to_prompt_phase(prompt_len=10) is True
        assert loc.applies_to_generation_phase(prompt_len=10) is False

    def test_generation_phase_detection(self):
        """'generation' should apply only to generation phase."""
        loc = Location(layers=[6], modules=["mlp"], token_pos="generation")
        assert loc.applies_to_prompt_phase(prompt_len=10) is False
        assert loc.applies_to_generation_phase(prompt_len=10) is True

    def test_all_phase_detection(self):
        """'all' should apply to both phases."""
        loc = Location(layers=[6], modules=["mlp"], token_pos="all")
        assert loc.applies_to_prompt_phase(prompt_len=10) is True
        assert loc.applies_to_generation_phase(prompt_len=10) is True


# ==============================================================================
# INTEGRATION TESTS - ESSENTIAL (13 tests)
# ==============================================================================


class TestMuranoModelEssentials:
    """Essential integration tests with MuranoModel."""

    # --- record() Method (3 tests) ---

    def test_record_with_prompt_keyword(self, murano_model):
        """Record should extract activations from prompt tokens only."""
        loc = Location(layers=[6], modules=["mlp"], token_pos="prompt")
        input_text = "Hello world, this is a test."

        artifact = murano_model.record(input_text, loc)

        assert "activations" in artifact
        acts = artifact["activations"]

        # Verify prompt tokens were extracted
        input_ids = murano_model.model.tokenizer(input_text, return_tensors="pt")[
            "input_ids"
        ]
        prompt_len = input_ids.shape[1]

        # Check that we got prompt-length activations
        assert acts[0][0].shape[0] == prompt_len or acts[0][0].shape[1] == prompt_len

    def test_record_with_last_keyword(self, murano_model):
        """Record should extract only the last token."""
        loc = Location(layers=[6], modules=["mlp"], token_pos="last")
        input_text = "Hello world, this is a test."

        artifact = murano_model.record(input_text, loc)

        assert "activations" in artifact
        # Should have extracted just 1 token
        acts = artifact["activations"]
        # The last dimension should be 1 (one token)
        assert acts[0][0].shape[0] == 1 or acts[0][0].shape[1] == 1

    def test_record_backward_compatibility(self, murano_model):
        """Record should still work with integer lists."""
        loc = Location(layers=[6], modules=["mlp"], token_pos=[0, -1])
        input_text = "Hello world, this is a test."

        artifact = murano_model.record(input_text, loc)

        assert "activations" in artifact
        # Should have 2 tokens (first and last)
        acts = artifact["activations"]
        assert acts[0][0].shape[0] == 2 or acts[0][0].shape[1] == 2

    # --- generate_intervene() Method (6 tests) ---

    def test_generate_intervene_with_prompt_phase(self, murano_model):
        """Intervention with 'prompt' should work without errors."""
        loc_intervene = Location(layers=[6], modules=["mlp"], token_pos="prompt")

        hidden_dim = murano_model.model.config.hidden_size
        steering_vector = torch.randn(1, 1, 1, 1, hidden_dim) * 0.1
        sv_dataset = steering_vector_to_activation_dataset(
            steering_vector, loc_intervene
        )

        test_prompt = "The movie was"
        result = murano_model.generate_intervene(
            input=test_prompt,
            intervene_location=loc_intervene,
            activation_dataset=sv_dataset,
            max_new_tokens=5,
            mode="addition",
        )

        assert "output_ids" in result
        assert result["output_ids"].shape[1] > len(
            murano_model.model.tokenizer(test_prompt, return_tensors="pt")["input_ids"][
                0
            ]
        )

    def test_generate_intervene_with_generation_phase(self, murano_model):
        """Intervention with 'generation' should work without errors."""
        loc_intervene = Location(layers=[6], modules=["mlp"], token_pos="generation")

        hidden_dim = murano_model.model.config.hidden_size
        steering_vector = torch.randn(1, 1, 1, 1, hidden_dim) * 0.1
        sv_dataset = steering_vector_to_activation_dataset(
            steering_vector, loc_intervene
        )

        test_prompt = "The movie was"
        result = murano_model.generate_intervene(
            input=test_prompt,
            intervene_location=loc_intervene,
            activation_dataset=sv_dataset,
            max_new_tokens=5,
            mode="addition",
        )

        assert "output_ids" in result
        assert result["output_ids"].shape[1] > len(
            murano_model.model.tokenizer(test_prompt, return_tensors="pt")["input_ids"][
                0
            ]
        )

    def test_generate_intervene_with_all_phase(self, murano_model):
        """Intervention with 'all' should work without errors."""
        loc_intervene = Location(layers=[6], modules=["mlp"], token_pos="all")

        hidden_dim = murano_model.model.config.hidden_size
        steering_vector = torch.randn(1, 1, 1, 1, hidden_dim) * 0.1
        sv_dataset = steering_vector_to_activation_dataset(
            steering_vector, loc_intervene
        )

        test_prompt = "The movie was"
        result = murano_model.generate_intervene(
            input=test_prompt,
            intervene_location=loc_intervene,
            activation_dataset=sv_dataset,
            max_new_tokens=5,
            mode="addition",
        )

        assert "output_ids" in result

    def test_generate_intervene_backward_compatibility(self, murano_model):
        """generate_intervene should work with integer lists."""
        loc_intervene = Location(layers=[6], modules=["mlp"], token_pos=[-1])

        hidden_dim = murano_model.model.config.hidden_size
        steering_vector = torch.randn(1, 1, 1, 1, hidden_dim) * 0.1
        sv_dataset = steering_vector_to_activation_dataset(
            steering_vector, loc_intervene
        )

        test_prompt = "The movie was"
        result = murano_model.generate_intervene(
            input=test_prompt,
            intervene_location=loc_intervene,
            activation_dataset=sv_dataset,
            max_new_tokens=5,
            mode="addition",
        )

        assert "output_ids" in result

    def test_generate_intervene_replacement_mode(self, murano_model):
        """Test replacement mode works."""
        loc_intervene = Location(layers=[6], modules=["mlp"], token_pos="generation")

        hidden_dim = murano_model.model.config.hidden_size
        steering_vector = torch.randn(1, 1, 1, 1, hidden_dim) * 0.1
        sv_dataset = steering_vector_to_activation_dataset(
            steering_vector, loc_intervene
        )

        test_prompt = "The movie was"
        result = murano_model.generate_intervene(
            input=test_prompt,
            intervene_location=loc_intervene,
            activation_dataset=sv_dataset,
            max_new_tokens=5,
            mode="replacement",
        )

        assert "output_ids" in result

    def test_generate_intervene_addition_mode(self, murano_model):
        """Test addition mode works."""
        loc_intervene = Location(layers=[6], modules=["mlp"], token_pos="generation")

        hidden_dim = murano_model.model.config.hidden_size
        steering_vector = torch.randn(1, 1, 1, 1, hidden_dim) * 0.1
        sv_dataset = steering_vector_to_activation_dataset(
            steering_vector, loc_intervene
        )

        test_prompt = "The movie was"
        result = murano_model.generate_intervene(
            input=test_prompt,
            intervene_location=loc_intervene,
            activation_dataset=sv_dataset,
            max_new_tokens=5,
            mode="addition",
        )

        assert "output_ids" in result

    # --- Complete Pipeline (4 tests) ---

    def test_complete_steering_pipeline(self, murano_model):
        """Test complete workflow: extract from prompt, steer generation."""
        # 1. Extract from prompt phase
        loc_extract = Location(layers=[6], modules=["mlp"], token_pos="prompt")

        pos_input = "I love this movie!"
        neg_input = "I hate this movie!"

        pos_artifact = murano_model.record(pos_input, loc_extract)
        neg_artifact = murano_model.record(neg_input, loc_extract)

        # Get activations
        pos_act = pos_artifact["activations"][0][0]
        neg_act = neg_artifact["activations"][0][0]

        # Handle LazyTensor
        if hasattr(pos_act, "value"):
            pos_act = pos_act.value
        if hasattr(neg_act, "value"):
            neg_act = neg_act.value

        # Average and compute steering vector from last token (shape -> [1, 1, 768])
        pos_vec = pos_act[:, -1:, :] if pos_act.ndim > 1 else pos_act
        neg_vec = neg_act[:, -1:, :] if neg_act.ndim > 1 else neg_act

        steering_vector = pos_vec - neg_vec

        # 2. Intervene on generation phase
        loc_intervene = Location(layers=[6], modules=["mlp"], token_pos="generation")
        sv_dataset = steering_vector_to_activation_dataset(
            steering_vector, loc_intervene
        )

        # 3. Generate
        test_prompt = "The movie was"
        result = murano_model.generate_intervene(
            input=test_prompt,
            intervene_location=loc_intervene,
            activation_dataset=sv_dataset,
            max_new_tokens=10,
            mode="addition",
        )

        assert "output_ids" in result
        generated_text = murano_model.model.tokenizer.decode(
            result["output_ids"][0], skip_special_tokens=True
        )
        assert len(generated_text) > len(test_prompt)

    def test_extract_last_intervene_all(self, murano_model):
        """Extract from last token, intervene on all."""
        # Extract from last prompt token
        loc_extract = Location(layers=[6], modules=["mlp"], token_pos="last")

        pos_input = "I love this!"
        pos_artifact = murano_model.record(pos_input, loc_extract)
        pos_act = pos_artifact["activations"][0][0]

        if hasattr(pos_act, "value"):
            pos_act = pos_act.value

        # Intervene on all tokens
        loc_intervene = Location(layers=[6], modules=["mlp"], token_pos="all")
        sv_reshaped = pos_act.view(1, 1, 1, 1, -1)
        sv_dataset = steering_vector_to_activation_dataset(sv_reshaped, loc_intervene)

        test_prompt = "Test"
        result = murano_model.generate_intervene(
            input=test_prompt,
            intervene_location=loc_intervene,
            activation_dataset=sv_dataset,
            max_new_tokens=5,
            mode="addition",
        )

        assert "output_ids" in result

    def test_multiple_keywords_in_sequence(self, murano_model):
        """Test using different keywords in sequence."""
        # First record with "all"
        loc1 = Location(layers=[6], modules=["mlp"], token_pos="all")
        artifact1 = murano_model.record("Test input", loc1)
        assert "activations" in artifact1

        # Then record with "last"
        loc2 = Location(layers=[6], modules=["mlp"], token_pos="last")
        artifact2 = murano_model.record("Test input", loc2)
        assert "activations" in artifact2

        # Then intervene with "prompt"
        loc3 = Location(layers=[6], modules=["mlp"], token_pos="prompt")
        hidden_dim = murano_model.model.config.hidden_size
        steering_vector = torch.randn(1, 1, 1, 1, hidden_dim) * 0.1
        sv_dataset = steering_vector_to_activation_dataset(steering_vector, loc3)

        result = murano_model.generate_intervene(
            input="Test",
            intervene_location=loc3,
            activation_dataset=sv_dataset,
            max_new_tokens=3,
            mode="addition",
        )

        assert "output_ids" in result
