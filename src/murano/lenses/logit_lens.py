from typing import Any

import plotly.express as px
import plotly.io as pio
import torch

from .base_lens import BaseLens

pio.renderers.default = "browser"


class LogitLens(BaseLens):
    def __init__(self):
        super().__init__(name="LogitLens")

    def process(self, artifact: dict) -> dict:
        model = artifact["model"]
        activations = artifact["activations"]

        probs_layers = []
        for activation in activations:
            activation = activation[0]
            normalized = model.transformer.ln_f(activation)
            logits = model.lm_head(normalized)

            probs = torch.nn.functional.softmax(logits, dim=-1)
            probs_layers.append(probs)

        all_probs = torch.stack(probs_layers)

        max_probs, tokens = all_probs.max(dim=-1)

        words = [
            [
                model.tokenizer.decode(t.cpu()).encode("unicode_escape").decode()
                for t in layer_tokens
            ]
            for layer_tokens in tokens
        ]

        input_words = [model.tokenizer.decode(t) for t in artifact["input_ids"]]

        artifact["max_probs"] = max_probs
        artifact["predicted_tokens"] = tokens
        artifact["predicted_words"] = words
        artifact["input_words"] = input_words
        artifact["all_probs"] = all_probs

        return artifact

    def visualize(self, artifact: dict) -> Any:
        max_probs = artifact["max_probs"].detach().cpu().numpy().squeeze(axis=1)
        words = artifact["predicted_words"]
        input_words = artifact["input_words"]
        layer_indices = artifact["layer_indices"]

        fig = px.imshow(
            max_probs,
            x=input_words,
            y=[f"Layer {i}" for i in layer_indices],
            color_continuous_scale=px.colors.diverging.RdYlBu_r,
            color_continuous_midpoint=0.50,
            text_auto=True,
            labels=dict(x="Input Tokens", y="Layers", color="Probability"),
        )

        fig.update_layout(title="Logit Lens Visualization", xaxis_tickangle=0)

        fig.update_traces(text=words, texttemplate="%{text}")
        fig.show()
