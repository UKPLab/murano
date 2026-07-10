"""Tests for the shared toy tasks.

These fixtures are consumed by the notebooks, so a silent change to their shape
would break every tutorial at once. The invariants below are the ones the
notebooks and the metric steps rely on.
"""

from __future__ import annotations

import pytest

from murano.dataset import CleanCorruptDataset
from murano.tasks import (
    IOI_TEMPLATE,
    POSITIVE_WORDS,
    ioi,
    positive_word_rate,
    sentiment,
)


# ── ioi ───────────────────────────────────────────────────────────────


def test_ioi_returns_a_paired_dataset():
    task = ioi(n=3)

    assert isinstance(task, CleanCorruptDataset)
    assert len(task.clean) == len(task.corrupt) == 3
    assert len(task.correct) == len(task.incorrect) == 3


def test_ioi_clean_prompt_names_the_subject_as_giver():
    """Clean: the subject gives, so the answer is the OTHER name."""
    task = ioi(n=1)

    assert task.clean[0] == IOI_TEMPLATE.format(a="Mary", b="John", giver="John")
    assert task.correct[0] == " Mary"
    assert task.incorrect[0] == " John"


def test_ioi_corrupt_prompt_flips_who_gives():
    """Corrupt: the indirect object gives, which flips the expected answer."""
    task = ioi(n=1)

    assert task.corrupt[0] == IOI_TEMPLATE.format(a="Mary", b="John", giver="Mary")
    assert task.clean[0] != task.corrupt[0]


def test_ioi_answers_carry_the_leading_space():
    """LogitDiffStep tokenizes these directly; a missing space picks a different id."""
    task = ioi()

    assert all(answer.startswith(" ") for answer in task.correct)
    assert all(answer.startswith(" ") for answer in task.incorrect)


def test_ioi_clean_and_corrupt_have_the_same_token_count():
    """Patching resamples position by position, so the pair must align."""
    task = ioi()

    for clean, corrupt in zip(task.clean, task.corrupt):
        assert len(clean.split()) == len(corrupt.split())


def test_ioi_every_pair_is_distinct():
    task = ioi()

    assert len(set(task.clean)) == len(task.clean)
    assert set(task.correct).isdisjoint(task.incorrect)


@pytest.mark.parametrize("n", [0, -1, 13])
def test_ioi_rejects_an_out_of_range_size(n):
    with pytest.raises(ValueError, match="n must be between"):
        ioi(n=n)


def test_ioi_records_its_provenance():
    assert ioi().metadata["task"] == "ioi"


# ── sentiment ─────────────────────────────────────────────────────────


def test_sentiment_returns_balanced_classes():
    positive, negative = sentiment(n_per_class=10)

    assert len(positive) == len(negative) == 10
    assert set(positive).isdisjoint(negative)


def test_sentiment_defaults_to_every_sentence():
    positive, negative = sentiment()

    assert len(positive) == len(negative) == 25


@pytest.mark.parametrize("n", [0, 26])
def test_sentiment_rejects_an_out_of_range_size(n):
    with pytest.raises(ValueError, match="n_per_class must be between"):
        sentiment(n_per_class=n)


# ── positive_word_rate ────────────────────────────────────────────────


def test_positive_word_rate_counts_generations_not_words():
    """One hit is enough; a second word in the same generation must not double it."""
    assert positive_word_rate(["good and great", "nothing here"]) == 0.5


def test_positive_word_rate_strips_trailing_punctuation():
    assert positive_word_rate(["it was good."]) == 1.0
    assert positive_word_rate(["it was GOOD!"]) == 1.0


def test_positive_word_rate_bounds():
    assert positive_word_rate(["dreadful"]) == 0.0
    assert positive_word_rate(["love"]) == 1.0


def test_positive_word_rate_rejects_an_empty_list():
    with pytest.raises(ValueError, match="empty list"):
        positive_word_rate([])


def test_positive_words_is_immutable():
    """The notebooks read this constant; it must not be mutable shared state."""
    with pytest.raises(AttributeError):
        POSITIVE_WORDS.add("splendid")  # type: ignore[attr-defined]
