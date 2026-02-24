import pytest
import torch


from murano.model import MuranoModel
from murano.lenses.logit_lens import LogitComputationLens


@pytest.fixture(scope="module")
def gpt2_model():
    """Loads a real GPT-2 model using the MuranoModel wrapper."""
    return MuranoModel("gpt2").model


def test_logit_computation_lens_with_murano_model(gpt2_model):
    """
    Tests that the LogitComputationLens correctly computes probabilities,
    tokens, and decodes words using our MuranoModel.
    """
    prompt = "The quick brown fox"

    with gpt2_model.trace(prompt) as runner:
        # breakpoint()

        input_ids = gpt2_model.embed_tokens.input.save()
        hidden_states_layer_0 = gpt2_model.layers_output[0].save()
        hidden_states_layer_1 = gpt2_model.layers_output[1].save()

    activations = [[hidden_states_layer_0], [hidden_states_layer_1]]

    artifact = {
        "model": gpt2_model,
        "activations": activations,
        "input_ids": input_ids[0],
    }

    # Run the Lens Computation
    lens = LogitComputationLens()
    enriched_artifact = lens.process(artifact)

    assert "max_probs" in enriched_artifact
    assert "predicted_tokens" in enriched_artifact
    assert "predicted_words" in enriched_artifact
    assert "input_words" in enriched_artifact
    assert "all_probs" in enriched_artifact

    # Check shapes
    num_layers = 2
    seq_len = len(artifact["input_ids"])
    vocab_size = gpt2_model.config.vocab_size

    assert enriched_artifact["all_probs"].shape == (num_layers, seq_len, vocab_size)
    assert enriched_artifact["max_probs"].shape == (num_layers, seq_len)
    assert enriched_artifact["predicted_tokens"].shape == (num_layers, seq_len)

    # Check that decoding worked (list of lists of strings)
    assert len(enriched_artifact["predicted_words"]) == num_layers
    assert len(enriched_artifact["predicted_words"][0]) == seq_len

    # Check that input words were decoded correctly
    assert len(enriched_artifact["input_words"]) == seq_len
