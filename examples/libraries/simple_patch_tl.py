from argparse import Namespace
from typing import Callable, Dict, List, Optional, Tuple

import seaborn as sns
import torch
import transformer_lens.utils as utils
from matplotlib import pyplot as plt
from torch import Tensor
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)


def plot_heatmap(patch_result: Tensor, output_dir: str, cmap: str = "Purples") -> None:
    """
    Plot a heatmap of the causal tracing results.

    Args:
    patch_result (torch.Tensor): 2D tensor of shape (sequence_length - 1, n_layers) containing the causal tracing results.
    output_dir (str): Path to save the output heatmap.
    cmap (str): Colormap to use for the heatmap.

    Returns:
    None
    """
    fig, ax = plt.subplots()
    sns.heatmap(patch_result, ax=ax, cmap=cmap)
    # fig.savefig(output_dir)
    plt.show()  # Easier for development


class TraceTransformer(HookedTransformer):
    """
    A custom transformer model class for performing causal tracing analysis.
    Inherits from HookedTransformer and adds methods for causal tracing.
    """

    def get_target_id(self, token: str) -> int:
        """
        Get the token ID for a given target token.

        Args:
        token (str): The target token.

        Returns:
        int: The token ID.
        """
        encoded_tokens = self.tokenizer.encode(" " + token)
        assert len(encoded_tokens) == 1
        return encoded_tokens[0]

    def record_clean_activations(self, prompt: str) -> Dict[str, Tensor]:
        """
        Record the clean activations for a given prompt.

        Args:
        prompt (str): The input prompt.

        Returns:
        Dict[str, torch.Tensor]: A dictionary containing the clean activations for each layer.
        """
        prompt_token = self.to_tokens(prompt)
        logits, activations = self.run_with_cache(prompt_token)
        return activations

    def get_corrupted_probs(self, prompt: str, patch_embed_fn: Callable) -> Tensor:
        """
        Get the corrupted probabilities for a given prompt and patching function.

        Args:
        prompt (str): The input prompt.
        patch_embed_fn (Callable): The function to patch the embeddings.

        Returns:
        torch.Tensor: The corrupted probabilities for the last token.
        """
        logits = self.run_with_hooks(
            prompt, fwd_hooks=[(utils.get_act_name("embed"), patch_embed_fn)]
        )
        return logits[:, -1].softmax(dim=1)

    def find_sequence_span(self, prompt: str, seq: str) -> Tensor:
        """
        Find the token indices for a given sequence in the prompt.

        Args:
        prompt (str): The input prompt.
        seq (str): The sequence to find in the prompt.

        Returns:
        torch.Tensor: A tensor containing the indices of the sequence in the prompt.
        """
        assert seq in prompt
        seq_tokens = self.tokenizer.encode(seq)
        prompt_token_ids = self.to_tokens(prompt)[0].tolist()
        for i, token in enumerate(prompt_token_ids):
            if prompt_token_ids[i : i + len(seq_tokens)] == seq_tokens:
                return torch.arange(i, i + len(seq_tokens))
        raise ValueError("No subsequence found.")

    def get_patch_emb_fn(self, corrupt_span: Tensor, noise: float = 1.0) -> Callable:
        """
        Get a function to patch the embeddings with noise.

        Args:
        corrupt_span (torch.Tensor): The span of tokens to corrupt.
        noise (float): The amount of noise to add.

        Returns:
        Callable: A function that patches the embeddings with noise.
        """

        def patch_embed_fn(x: Tensor, hook: Optional[Namespace]) -> Tensor:
            x[:, corrupt_span, :] += noise * torch.randn(
                x[:, corrupt_span, :].shape, device=x.device
            )
            return x

        return patch_embed_fn

    def get_restore_fn(
        self, activation_record: Dict[str, Tensor], token_idx: int
    ) -> Callable:
        """
        Get a function to restore the activations for a specific token.

        Args:
        activation_record (Dict[str, torch.Tensor]): The recorded clean activations.
        token_idx (int): The index of the token to restore.

        Returns:
        Callable: A function that restores the activations for the specified token.
        """

        def restore(x: Tensor, hook: Optional[Namespace]) -> Tensor:
            x[:, token_idx, :] = activation_record[hook.name][:, token_idx, :]
            return x

        return restore

    def get_forward_hooks(
        self,
        layer: int,
        patch_embed_fn: Callable,
        patch_name: str,
        restore_fn: Callable,
        window: int = 10,
    ) -> List[Tuple[str, Callable]]:
        """
        Get the forward hooks for causal tracing.

        Args:
        layer (int): The current layer.
        patch_embed_fn (Callable): The function to patch the embeddings.
        patch_name (str): The name of the patch location ('resid_pre', 'mlp_post', or 'attn_out').
        restore_fn (Callable): The function to restore activations.
        window (int): The window size for tracing.

        Returns:
        List[Tuple[str, Callable]]: A list of tuples containing the hook names and functions.
        """
        if patch_name == "resid_pre":
            # Trace states
            return [
                (utils.get_act_name("embed"), patch_embed_fn),
                (utils.get_act_name(patch_name, layer), restore_fn),
            ]
        else:
            # Trace window
            window_layers = range(
                max(0, layer - window // 2),
                min(self.cfg.n_layers, layer - (-window // 2)),
            )
            return [(utils.get_act_name("embed"), patch_embed_fn)] + [
                (utils.get_act_name(patch_name, layer), restore_fn)
                for layer in window_layers
            ]

    def causal_trace_analysis(
        self,
        prompt: str,
        source: str,
        target: str,
        patch_name: str,
        noise: float = 1.0,
        window: int = 10,
    ) -> Tensor:
        """
        Perform causal tracing analysis on the model.

        Args:
        prompt (str): The input prompt.
        source (str): The source sequence to corrupt.
        target (str): The target token to predict.
        patch_name (str): The name of the patch location ('resid_pre', 'mlp_post', or 'attn_out').
        noise (float): The amount of noise to add when corrupting.
        window (int): The window size for tracing.

        Returns:
        torch.Tensor: A 2D tensor of shape (sequence_length - 1, n_layers) containing the causal tracing results.
        """
        prompt_tokens = self.to_tokens(prompt)
        target_id = self.get_target_id(target)
        corrupt_span = self.find_sequence_span(prompt, source)
        patch_embed_fn = self.get_patch_emb_fn(corrupt_span)
        activations = self.record_clean_activations(prompt)
        corrupted_probs = self.get_corrupted_probs(prompt, patch_embed_fn)

        patch_result = torch.zeros(prompt_tokens.size(1) - 1, self.cfg.n_layers)

        for layer in range(self.cfg.n_layers):
            for i, token in enumerate(prompt_tokens[0, 1:]):
                restore_fn = self.get_restore_fn(activations, i + 1)

                forward_hooks = self.get_forward_hooks(
                    layer, patch_embed_fn, patch_name, restore_fn, window
                )

                logits = self.run_with_hooks(prompt_tokens, fwd_hooks=forward_hooks)
                probs = logits[:, -1].softmax(dim=1)
                patch_result[i, layer] = (
                    probs[0, target_id] - corrupted_probs[0, target_id]
                )

        return patch_result


def q1_causal_trace(patch_name: str = "resid_pre") -> None:
    """
    Perform causal tracing analysis for a specific patch location.

    Args:
    patch_name (str): The name of the patch location ('resid_pre', 'mlp_post', or 'attn_out').

    Returns:
    None
    """
    cmap, name = {
        "resid_pre": ("Purples", "states"),
        "mlp_post": ("Greens", "mlp"),
        "attn_out": ("Reds", "attn"),
    }[patch_name]

    model = TraceTransformer.from_pretrained("gpt2-xl").to("cuda")
    result = model.causal_trace_analysis(
        "The Space Needle is located in",
        "The Space Needle",
        "Seattle",
        patch_name,
        noise=0.5,
    )

    plot_heatmap(result, name + ".pdf", cmap)


if __name__ == "__main__":
    q1_causal_trace("resid_pre")
    q1_causal_trace("mlp_post")
    q1_causal_trace("attn_out")
