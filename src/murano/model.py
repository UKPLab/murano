from typing import List

from nnsight import LanguageModel

from .lenses.base_lens import BaseLens
from .utils import LayerLocation


class MuranoModel:
    def __init__(self, model_name: str):
        self.model = LanguageModel(model_name, device_map="auto", dispatch=True)
        self.model_name = model_name

    @classmethod
    def from_pretrained(cls, model_name: str):
        return cls(model_name)

    def run_with_lens(
        self, prompt: str, lens: BaseLens, locations: List[LayerLocation]
    ) -> dict:
        activations = []
        layer_indices = []

        tokenized = self.model.tokenizer(prompt, return_tensors="pt")
        input_ids = tokenized["input_ids"]

        with self.model.trace() as tracer:
            with tracer.invoke(prompt):
                layers_list = list(self.model.transformer.h)

                for location in locations:
                    if isinstance(location.layers, slice):
                        selected_layers = list(range(len(layers_list)))
                    else:
                        selected_layers = (
                            location.layers
                            if isinstance(location.layers, list)
                            else [location.layers]
                        )

                    for layer_idx in selected_layers:
                        layer = layers_list[layer_idx]

                        output = layer.output
                        if isinstance(output, tuple):
                            hidden_states = output[0]
                        else:
                            hidden_states = output

                        saved_output = hidden_states.save()
                        activations.append(saved_output)
                        layer_indices.append(layer_idx)

        artifact = {
            "prompt": prompt,
            "activations": [act.value for act in activations],
            "layer_indices": layer_indices,
            "input_ids": input_ids[0],  # type: ignore
            "model": self.model,
            "tokenizer": self.model.tokenizer,
        }

        return lens.process(artifact)
