"""
Dataset utility functions for processing datasets for SAE and linear probe workflows.
"""

import torch


def collate_fn(batch):
    """Collate function that stitches separate inputs from dataset."""
    collated = {}
    for key in batch[0]:
        values = [example[key] for example in batch]
        if key in ["input_ids", "attention_mask"]:  # tensorize only these
            collated[key] = torch.stack([torch.tensor(v) for v in values])
        else:
            collated[key] = values  # keep as list
    return collated


def process_dataset(example, tokenizer, max_length=128):
    """Tokenize text in dataset example."""
    tokenized = tokenizer(
        example["text"], return_tensors="pt", max_length=max_length, truncation=True
    )
    example["input_ids"] = tokenized["input_ids"][0]
    example["attention_mask"] = tokenized["attention_mask"][0]
    return example

