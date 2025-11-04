"""
SAEMuranoModel - Extended MuranoModel for extracting hidden states from residual stream.

This module provides SAEMuranoModel which extends MuranoModel to extract activations
from residual stream hook points (resid_pre, resid_post) using nnsight.
"""

from typing import Dict, List, Optional, TYPE_CHECKING

import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from .model import MuranoModel

from .dataset_utils import collate_fn, process_dataset

if TYPE_CHECKING:
    from sae_lens import SAE


class SAEMuranoModel(MuranoModel):
    """
    Extended MuranoModel that extracts hidden states from residual stream using nnsight.

    This class allows:
    1. Extracting raw hidden state activations (for training linear probes)
    2. Optionally extracting SAE activations directly (without training)

    Uses nnsight (like base MuranoModel) to extract activations from hook_resid_pre
    and hook_resid_post at specified layers.
    """

    def __init__(
        self,
        model_name: str,
        hook_points: List[str],
        layers: List[int],
        sae: Optional["SAE"] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize SAEMuranoModel.

        Args:
            model_name: Name of the base model (e.g., "gpt2")
            hook_points: List of hook point names (e.g., ["resid_pre", "resid_post"])
                       These refer to residual stream hook points
            layers: List of layer indices to extract activations from
            sae: Optional pre-loaded SAE object from sae_lens (for direct SAE feature extraction)
            device: Device to run on (defaults to "cuda" if available, else "cpu")
        """
        # Initialize base MuranoModel (uses nnsight LanguageModel)
        super().__init__(model_name)
        self.model_name = model_name
        self.hook_points = hook_points
        self.layers = layers
        self.sae = sae
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.sae is not None:
            self.sae.to(self.device)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        hook_points: List[str],
        layers: List[int],
        sae_release: Optional[str] = None,
        sae_id: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Create SAEMuranoModel from pre-trained model and optionally SAE.

        Args:
            model_name: Name of the base model
            hook_points: List of hook point names (e.g., ["resid_pre", "resid_post"])
            layers: List of layer indices
            sae_release: Optional SAE release name (e.g., "gpt2-small-res-jb")
            sae_id: Optional SAE identifier (e.g., "blocks.7.hook_resid_pre")
            device: Device to run on

        Returns:
            SAEMuranoModel instance
        """
        sae = None
        if sae_release is not None and sae_id is not None:
            try:
                from sae_lens import SAE
            except ImportError as e:
                raise ImportError(
                    "sae-lens is required for SAE functionality. "
                    "Install it with: pip install sae-lens"
                ) from e
            device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            sae = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device)
        return cls(model_name, hook_points, layers, sae, device)

    def run_recording(
        self,
        input_ids: torch.Tensor,
        token_pos: Optional[int] = None,
        use_sae: bool = False,
        **kwargs
    ) -> Dict:
        """
        Run the model to record activations at specified residual stream locations.

        Uses nnsight (like base MuranoModel) to extract hidden states from residual stream.
        Can extract either raw activations (for training probes) or SAE activations (direct use).

        Args:
            input_ids: Tokenized input tensor of shape (batch_size, seq_len)
            token_pos: Optional token position to extract (if None, uses last token)
            use_sae: If True, extract SAE activations; if False, extract raw hidden states
            **kwargs: Additional arguments for model forward pass

        Returns:
            Dictionary containing activations and metadata
        """
        activations_list = []

        # Move input to device
        input_ids = input_ids.to(self.device)

        # Use nnsight to extract activations (like base MuranoModel)
        with self.model.trace() as tracer:
            with tracer.invoke(input_ids, **kwargs):
                layers_list = list(self.model.transformer.h)

                for layer_idx in self.layers:
                    layer = layers_list[layer_idx]
                    layer_activations = []

                    for hook_point in self.hook_points:
                        # Access residual stream hook points using nnsight
                        # resid_pre: input to the layer (residual stream before layer)
                        # resid_post: output of the layer (residual stream after layer)
                        if hook_point == "resid_pre":
                            # Get residual stream input to this layer
                            hidden_states = layer.input
                            if isinstance(hidden_states, tuple):
                                hidden_states = hidden_states[0]
                        elif hook_point == "resid_post":
                            # Get residual stream output from this layer
                            hidden_states = layer.output
                            if isinstance(hidden_states, tuple):
                                hidden_states = hidden_states[0]
                        else:
                            raise ValueError(
                                f"Unsupported hook point: {hook_point}. "
                                f"Supported: 'resid_pre', 'resid_post'"
                            )

                        # Extract specific token position if specified
                        if token_pos is not None:
                            hidden_states = hidden_states[:, token_pos : token_pos + 1, :]
                        else:
                            # Use last token by default
                            hidden_states = hidden_states[:, -1:, :]

                        # Save activation
                        saved_activation = hidden_states.save()
                        layer_activations.append(saved_activation)

                    activations_list.append(layer_activations)

        # Process activations: get values and optionally pass through SAE
        processed_activations = []
        for layer_acts in activations_list:
            processed_layer_acts = []
            for act in layer_acts:
                activation = act.value  # Get tensor from nnsight save object

                if use_sae:
                    # Pass through SAE encoder to get sparse features
                    if self.sae is None:
                        raise ValueError(
                            "SAE not provided. Set use_sae=False or provide SAE in __init__"
                        )
                    # SAE.encode expects (batch, seq_len, hidden_dim) -> (batch, seq_len, sae_dict_size)
                    if hasattr(self.sae, "encode"):
                        activation = self.sae.encode(activation)
                    elif callable(self.sae):
                        activation = self.sae(activation)
                    else:
                        raise ValueError("SAE object does not have encode method or is not callable")

                processed_layer_acts.append(activation)
            processed_activations.append(processed_layer_acts)

        artifact = {
            "activations": processed_activations,
            "input_ids": input_ids,
            "layers": self.layers,
            "hook_points": self.hook_points,
            "use_sae": use_sae,
        }

        return artifact

    def _stack_activations(self, activations_list: List) -> torch.Tensor:
        """
        Stack activations from nested list structure.

        Args:
            activations_list: Nested list of activations

        Returns:
            Stacked tensor of shape (num_examples, num_layers, num_hook_points, seq_len, hidden_dim)
        """
        # Recursively stack
        stacked = []
        for layer_acts in activations_list:
            layer_stacked = torch.stack(layer_acts, dim=0)
            stacked.append(layer_stacked)

        # Stack along layer dimension: (num_layers, num_hook_points, batch, seq_len, hidden_dim)
        result = torch.stack(stacked, dim=0)

        # Reshape to (batch, num_layers, num_hook_points, seq_len, hidden_dim)
        result = result.permute(2, 0, 1, 3, 4)

        return result

    def run_task(
        self,
        dataset: Dataset,
        batch_size: int = 4,
        token_pos: Optional[int] = None,
        use_sae: bool = False,
        **kwargs
    ) -> Dict:
        """
        Run the model on a dataset to extract activations from residual stream.

        Args:
            dataset: HuggingFace Dataset with "text" and "label" fields
            batch_size: Batch size for processing
            token_pos: Optional token position to extract
            use_sae: If True, extract SAE activations; if False, extract raw hidden states
            **kwargs: Additional arguments

        Returns:
            Dictionary containing activations, labels, and metadata
        """
        # Process dataset: tokenize
        processed_dataset = dataset.map(
            lambda x: process_dataset(x, self.model.tokenizer),
            batched=False,
        )

        # Create dataloader
        dataloader = DataLoader(
            processed_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )

        all_activations = []
        all_labels = []

        for batch in dataloader:
            input_ids = batch["input_ids"]
            labels = batch.get("label", None)

            # Extract activations (raw hidden states or SAE features)
            artifact = self.run_recording(
                input_ids, token_pos=token_pos, use_sae=use_sae, **kwargs
            )
            stacked_acts = self._stack_activations(artifact["activations"])

            # Flatten for linear probe: (batch, num_layers * num_hook_points * seq_len * hidden_dim)
            batch_size_actual = stacked_acts.shape[0]
            flattened = stacked_acts.reshape(batch_size_actual, -1)

            all_activations.append(flattened.cpu())
            if labels is not None:
                if isinstance(labels, list):
                    labels = torch.tensor(labels)
                all_labels.append(labels)

        # Concatenate all batches
        activations = torch.cat(all_activations, dim=0)
        labels = torch.cat(all_labels, dim=0) if all_labels else None

        global_metadata = {
            "model_name": self.model_name,
            "sae_id": (
                self.sae.cfg.hook_name if (self.sae and hasattr(self.sae, "cfg")) else None
            ),
            "hook_points": self.hook_points,
            "layers": self.layers,
            "batch_size": batch_size,
            "use_sae": use_sae,
            "num_examples": len(activations),
            "activation_shape": list(activations.shape),
        }

        artifact = {
            "activations": activations,
            "labels": labels,
            "global_metadata": global_metadata,
            "dataset": processed_dataset,
        }

        return artifact

