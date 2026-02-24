import torch
from typing import Any, Dict

from .base_lens import BaseComputationLens


class LogitComputationLens(BaseComputationLens):
    """
    Projects intermediate hidden states (activations) to the vocabulary
    to determine what the model was 'predicting' at each layer.
    """

    def __init__(self):
        super().__init__(name="LogitComputationLens")

    def process(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        model = artifact["model"]
        activations = artifact["activations"]
        input_ids = artifact["input_ids"]

        probs_layers = []
        for activation in activations:
            # Extract the actual tensor
            hidden_state = activation[0]

            normalized = model.transformer.ln_f(hidden_state)
            logits = model.lm_head(normalized)

            # Compute probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)

            if probs.dim() == 3 and probs.size(0) == 1:
                probs = probs.squeeze(0)

            probs_layers.append(probs)

        # (num_layers X sequence_length X vocab_size)
        all_probs = torch.stack(probs_layers)

        # Get max probabilities and their corresponding token IDs
        max_probs, tokens = all_probs.max(dim=-1)

        # Decode
        words = [
            [
                model.tokenizer.decode(t.cpu()).encode("unicode_escape").decode()
                for t in layer_tokens
            ]
            for layer_tokens in tokens
        ]

        input_words = [model.tokenizer.decode(t) for t in input_ids]

        # Enrich the artifact
        artifact["max_probs"] = max_probs
        artifact["predicted_tokens"] = tokens
        artifact["predicted_words"] = words
        artifact["input_words"] = input_words
        artifact["all_probs"] = all_probs

        return artifact
