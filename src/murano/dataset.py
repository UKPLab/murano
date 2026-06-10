"""MuranoDataset: dataset representations for pipeline steps."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from datasets import Dataset as _HFDataset


def _load_dataset_cached(name, config, split) -> "_HFDataset":
    """Load a HF dataset, trying offline cache first to avoid API rate limits.

    Returns a single ``datasets.Dataset`` (never a DatasetDict or streaming variant).
    """
    import os
    from datasets import Dataset, load_dataset

    old = os.environ.get("HF_DATASETS_OFFLINE")
    ds = None
    try:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        ds = load_dataset(name, config, split=split)
    except Exception:
        pass
    finally:
        if old is None:
            os.environ.pop("HF_DATASETS_OFFLINE", None)
        else:
            os.environ["HF_DATASETS_OFFLINE"] = old
    if ds is None:
        ds = load_dataset(name, config, split=split)
    if not isinstance(ds, Dataset):
        raise TypeError(
            f"Expected a single Dataset for split={split!r}, got {type(ds).__name__}."
        )
    return ds


def _load_hub_column(
    source: str | tuple,
    column: str | None = None,
    split: str = "train",
    n: int | None = None,
) -> list[str]:
    """Load a column from a HuggingFace dataset.

    Args:
        source: Either a dataset name string, or a tuple of
                (dataset_name, config_name, column_name).
        column: Column to extract (required if source is a string).
        split: Dataset split to load.
        n: Number of examples to take (None = all).

    Returns:
        List of strings from the specified column.

    Raises:
        ValueError: If ``source`` is a tuple of unexpected length, if
            ``column`` cannot be inferred, or if the column is missing
            from the loaded dataset.
    """

    if isinstance(source, tuple):
        if len(source) == 3:
            name, config, column = source
        elif len(source) == 2:
            name, column = source
            config = None
        else:
            raise ValueError(
                f"Expected tuple of (name, column) or (name, config, column), got {source}"
            )
    else:
        name = source
        config = None

    if column is None:
        raise ValueError(
            "column must be specified either in the tuple or as a separate argument"
        )

    ds = _load_dataset_cached(name, config, split)
    try:
        texts = ds[column]
    except KeyError:
        raise ValueError(
            f"Column '{column}' not found. Available: {list(ds.column_names)}"
        ) from None
    if n is not None:
        texts = texts[:n]
    return list(texts)


class MuranoDataset:
    """Dataset container supporting contrastive pairs.

    For refusal direction: positive = harmful prompts, negative = harmless prompts.

    Attributes:
        positive_texts: List of strings in the positive class (possibly templated).
        negative_texts: List of strings in the negative class (possibly templated).
        raw_positive: Original positive texts before chat template (None if no template).
        raw_negative: Original negative texts before chat template (None if no template).
    """

    def __init__(
        self,
        positive_texts: list[str],
        negative_texts: list[str],
        raw_positive: list[str] | None = None,
        raw_negative: list[str] | None = None,
    ):
        self.positive_texts = positive_texts
        self.negative_texts = negative_texts
        self.raw_positive = raw_positive
        self.raw_negative = raw_negative

    @classmethod
    def from_hub(
        cls,
        positive: str | tuple,
        negative: str | tuple,
        n_train: int = 150,
        n_eval: int = 50,
        template_fn: Callable[[list[dict]], str] | None = None,
    ) -> tuple[MuranoDataset, MuranoDataset]:
        """Load contrastive datasets directly from HuggingFace Hub.

        Args:
            positive: HF source for positive class. Either:
                - A tuple of (dataset_name, config, column): ("walledai/HarmBench", "standard", "prompt")
                - A tuple of (dataset_name, column): ("tatsu-lab/alpaca", "instruction")
            negative: HF source for negative class, same format as positive.
            n_train: Number of examples for the training split.
            n_eval: Number of examples for the evaluation split.
            template_fn: If provided, wraps each text in a chat template.

        Returns:
            Tuple of (train_dataset, eval_dataset).

        Raises:
            ValueError: If ``positive`` or ``negative`` is a malformed source
                tuple, or the named column is missing from the loaded dataset.

        Example:
            train_ds, eval_ds = MuranoDataset.from_hub(
                positive=("walledai/HarmBench", "standard", "prompt"),
                negative=("tatsu-lab/alpaca", "instruction"),
                n_train=150, n_eval=50,
                template_fn=model.chat_template,
            )
        """
        n_total = n_train + n_eval
        pos_texts = _load_hub_column(positive, n=n_total)
        neg_texts = _load_hub_column(negative, n=n_total)

        raw_pos = list(pos_texts)
        raw_neg = list(neg_texts)

        if template_fn is not None:
            pos_texts = [
                template_fn([{"role": "user", "content": t}]) for t in pos_texts
            ]
            neg_texts = [
                template_fn([{"role": "user", "content": t}]) for t in neg_texts
            ]

        train_ds = cls(
            positive_texts=pos_texts[:n_train],
            negative_texts=neg_texts[:n_train],
            raw_positive=raw_pos[:n_train] if template_fn else None,
            raw_negative=raw_neg[:n_train] if template_fn else None,
        )
        eval_ds = cls(
            positive_texts=pos_texts[n_train:n_total],
            negative_texts=neg_texts[n_train:n_total],
            raw_positive=raw_pos[n_train:n_total] if template_fn else None,
            raw_negative=raw_neg[n_train:n_total] if template_fn else None,
        )
        return train_ds, eval_ds

    @classmethod
    def contrastive(
        cls,
        positive: list[str],
        negative: list[str],
        template_fn: Callable[[list[dict]], str] | None = None,
    ) -> MuranoDataset:
        """Create a contrastive dataset from paired text lists.

        Args:
            positive: Texts in the positive class (e.g., harmful instructions).
            negative: Texts in the negative class (e.g., harmless instructions).
            template_fn: If provided, wraps each text in a chat template.
                         Should accept a list of message dicts and return a string.
                         Typically model.chat_template.

        Returns:
            MuranoDataset with formatted texts.
        """
        raw_pos = list(positive)
        raw_neg = list(negative)
        if template_fn is not None:
            positive = [template_fn([{"role": "user", "content": t}]) for t in positive]
            negative = [template_fn([{"role": "user", "content": t}]) for t in negative]
            return cls(
                positive_texts=list(positive),
                negative_texts=list(negative),
                raw_positive=raw_pos,
                raw_negative=raw_neg,
            )
        return cls(positive_texts=raw_pos, negative_texts=raw_neg)

    def __len__(self) -> int:
        return len(self.positive_texts) + len(self.negative_texts)

    def __repr__(self) -> str:
        return (
            f"MuranoDataset(positive={len(self.positive_texts)}, "
            f"negative={len(self.negative_texts)})"
        )


class LabeledDataset:
    """Dataset pairing texts with per-example integer labels.

    Attributes:
        texts: List of input strings (possibly chat-templated).
        labels: List of integer labels (0-indexed).
        label_names: Optional mapping from int to human-readable name.
        raw_texts: Original texts before chat template (None if no template).

    Raises:
        ValueError: If ``texts`` and ``labels`` have different lengths.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        label_names: list[str] | None = None,
        raw_texts: list[str] | None = None,
    ):
        if len(texts) != len(labels):
            raise ValueError(
                f"texts ({len(texts)}) and labels ({len(labels)}) must have same length"
            )
        self.texts = texts
        self.labels = labels
        self.label_names = label_names
        self.raw_texts = raw_texts

    @classmethod
    def from_hub(
        cls,
        source: str | tuple,
        text_column: str = "text",
        label_column: str = "label",
        split: str = "train",
        n: int | None = None,
        label_names: list[str] | None = None,
        template_fn: Callable[[list[dict]], str] | None = None,
    ) -> LabeledDataset:
        """Load a labeled dataset from HuggingFace Hub.

        Args:
            source: Dataset name or (name, config) tuple.
            text_column: Column containing text.
            label_column: Column containing integer labels.
            split: Dataset split.
            n: Max examples (None = all).
            label_names: Optional label name list. If None, inferred from
                         dataset features if available.
            template_fn: Optional chat template function.

        Returns:
            LabeledDataset ready for use in a probing pipeline.

        Raises:
            ValueError: If ``text_column`` or ``label_column`` is missing
                from the loaded dataset.

        Example:
            ds = LabeledDataset.from_hub(
                "stanfordnlp/sst2",
                text_column="sentence",
                label_column="label",
                n=500,
                label_names=["negative", "positive"],
            )
        """
        if isinstance(source, tuple):
            name, config = source if len(source) == 2 else (source[0], None)
        else:
            name, config = source, None

        ds = _load_dataset_cached(name, config, split)

        for col in [text_column, label_column]:
            if col not in ds.column_names:
                raise ValueError(
                    f"Column '{col}' not found. Available: {list(ds.column_names)}"
                )

        texts = list(ds[text_column])
        labels = list(ds[label_column])

        if n is not None:
            texts = texts[:n]
            labels = labels[:n]

        # Infer label_names from dataset features if not provided
        if label_names is None:
            features = ds.features.get(label_column) if ds.features else None
            names = getattr(features, "names", None)
            if names is not None:
                label_names = names

        raw_texts = list(texts) if template_fn else None
        if template_fn is not None:
            texts = [template_fn([{"role": "user", "content": t}]) for t in texts]

        return cls(
            texts=texts,
            labels=labels,
            label_names=label_names,
            raw_texts=raw_texts,
        )

    @classmethod
    def from_lists(
        cls,
        texts: list[str],
        labels: list[int],
        label_names: list[str] | None = None,
        template_fn: Callable[[list[dict]], str] | None = None,
    ) -> LabeledDataset:
        """Create from Python lists.

        Args:
            texts: Input strings.
            labels: Integer labels.
            label_names: Optional label name list.
            template_fn: Optional chat template function.

        Returns:
            LabeledDataset wrapping the supplied texts and labels.
        """
        raw_texts = list(texts) if template_fn else None
        if template_fn is not None:
            texts = [template_fn([{"role": "user", "content": t}]) for t in texts]
        return cls(
            texts=texts,
            labels=labels,
            label_names=label_names,
            raw_texts=raw_texts,
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __repr__(self) -> str:
        n_classes = len(set(self.labels))
        return f"LabeledDataset(n={len(self.texts)}, classes={n_classes})"
