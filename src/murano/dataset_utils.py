"""
Dataset utility functions for processing datasets for SAE and linear probe workflows.
"""

import torch


def collate_fn(batch):
    """
    Collate function for batching dataset examples.

    Args:
        batch: List of dataset examples, each a dictionary with keys such as
            "input_ids", "attention_mask", and any other keys present in the dataset.

    Returns:
        Dictionary with batched tensors for "input_ids" and "attention_mask", and lists for other keys.
    """
    collated = {}
    for key in batch[0]:
        values = [example[key] for example in batch]
        if key in ["input_ids", "attention_mask"]:  # tensorize only these
            collated[key] = torch.stack(
                [v if isinstance(v, torch.Tensor) else torch.tensor(v) for v in values]
            )
        else:
            collated[key] = values  # keep as list
    return collated


def process_dataset(example, tokenizer, max_length=128):
    """
    Tokenize the text in a dataset example and add tokenized fields.

    Parameters:
        example: A dictionary representing a dataset example. Must contain the key "text".
        tokenizer: A tokenizer object (e.g., from HuggingFace Transformers) with a callable interface.
        max_length: Maximum sequence length for tokenization. Defaults to 128.

    Modifies:
        Modifies the `example` dict in place by adding or overwriting the following keys:
            - "input_ids": Token IDs for the text.
            - "attention_mask": Attention mask for the text.

    Returns:
        The modified example dictionary.
    """
    tokenized = tokenizer(
        example["text"], return_tensors="pt", max_length=max_length, truncation=True
    )
    example["input_ids"] = tokenized["input_ids"][0]
    example["attention_mask"] = tokenized["attention_mask"][0]
    return example

