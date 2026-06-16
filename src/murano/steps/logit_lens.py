"""LogitLens step: project per-layer residuals onto the vocabulary.

Tracks how next-token predictions evolve across the depth of the network by
applying the model's final norm and unembedding to each recorded layer's
output. Cross-architecture access is provided by nnterp's standardized
``ln_final`` and ``lm_head`` modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from torch import Tensor, no_grad, stack  # pyright: ignore[reportPrivateImportUsage]
from torch.nn.functional import softmax

from murano import keys
from murano.artifacts import PromptBatch
from murano.logging import logger
from murano.nodes import RESID_POST, Node
from murano.results import Results
from murano.steps.base import Step

if TYPE_CHECKING:
    from murano.backend import ModelBackend


@dataclass
class LogitLensResult:
    """Per-layer next-token probability distributions.

    Attributes:
        all_probs: Tensor [n_layers, n_inputs, seq, vocab] of softmax outputs.
        max_probs: Tensor [n_layers, n_inputs, seq] of argmax probabilities.
        predicted_tokens: Tensor [n_layers, n_inputs, seq] of argmax token IDs.
        predicted_words: Decoded tokens, shape [n_layers][n_inputs][seq].
        input_words: Decoded input tokens, shape [n_inputs][seq].
        attention_mask: Tensor [n_inputs, seq] marking real (1) vs padding (0)
            positions; useful for masking padded columns at visualization time.
        addresses: Component addresses (one :class:`Node` per layer) the rows
            of the tensors correspond to. The lens reads each layer's block
            output, so these are ``resid_post`` nodes.
    """

    all_probs: Tensor
    max_probs: Tensor
    predicted_tokens: Tensor
    predicted_words: list[list[list[str]]]
    input_words: list[list[str]]
    attention_mask: Tensor
    addresses: list[Node]

    def __post_init__(self) -> None:
        self.addresses = [Node.coerce(address) for address in self.addresses]


class LogitLens(Step):
    """Compute logit-lens probabilities across layers.

    Runs a single nnsight trace on the batched prompts, captures full-sequence
    residual outputs at each requested layer, and projects them through the
    model's standardized final norm and unembedding (via nnterp).

    Reads from results:
        results['prompts']: PromptBatch

    Writes to results:
        results['logit_lens']: LogitLensResult

    Args:
        model: MuranoModel to record from.
        layers: Layer indices to record, or ``"all"`` for every layer.

    Raises:
        ValueError: If ``layers`` is a string other than ``"all"``.
    """

    reads = [keys.PROMPTS]
    writes = [keys.LOGIT_LENS]
    read_types = {keys.PROMPTS: PromptBatch}
    write_types = {keys.LOGIT_LENS: LogitLensResult}

    def __init__(
        self,
        model: ModelBackend,
        layers: list[int] | str = "all",
    ):
        self.model = model
        if isinstance(layers, str):
            if layers != "all":
                raise ValueError(f"layers as string must be 'all', got {layers!r}")
            self.layers: list[int] = list(range(model.n_layers))
        else:
            self.layers = list(layers)

    def __call__(self, results: Results) -> Results:
        prompt_batch: PromptBatch = results[keys.PROMPTS]
        prompts = prompt_batch.prompts
        logger.info("LogitLens: %d prompts, %d layers", len(prompts), len(self.layers))

        tokens = self.model.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_token_type_ids=False,
        )
        input_ids = cast(Tensor, tokens["input_ids"])
        attention_mask = cast(Tensor, tokens["attention_mask"])

        saved = {}
        with self.model.trace(tokens):
            for layer in self.layers:
                saved[layer] = self.model.layer(layer).output.save()

        layer_probs: list[Tensor] = []
        with no_grad():
            for layer in self.layers:
                output = (
                    saved[layer].value
                    if hasattr(saved[layer], "value")
                    else saved[layer]
                )
                if isinstance(output, tuple):
                    output = output[0]
                # output: [n_inputs, seq, d_model]; project to vocab.
                logits = self.model.project_on_vocab(output)
                layer_probs.append(softmax(logits, dim=-1).detach().cpu())

        # all_probs: [n_layers, n_inputs, seq, vocab]
        all_probs = stack(layer_probs, dim=0)
        max_probs, predicted_tokens = all_probs.max(dim=-1)

        input_words = [
            [self.model.tokenizer.decode([int(t)]) for t in seq.tolist()]
            for seq in input_ids
        ]
        predicted_words: list[list[list[str]]] = [
            [[self.model.tokenizer.decode([int(t)]) for t in seq] for seq in layer]
            for layer in predicted_tokens
        ]

        results[keys.LOGIT_LENS] = LogitLensResult(
            all_probs=all_probs,
            max_probs=max_probs,
            predicted_tokens=predicted_tokens,
            predicted_words=predicted_words,
            input_words=input_words,
            attention_mask=attention_mask.detach().cpu(),
            addresses=[Node(layer, RESID_POST) for layer in self.layers],
        )
        return results
