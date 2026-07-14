"""Small canonical tasks shared by the notebooks, tests, and reproductions.

Interpretability work needs a task before it needs a method, and the same two
toy tasks keep reappearing: a circuit-analysis task with a matched clean and
corrupt prompt (:func:`ioi`), and a contrastive concept task for probing and
steering (:func:`sentiment`). Defining them once here keeps every tutorial from
carrying its own copy, and gives the fixtures a place to be tested.

Both are deliberately tiny and hand-written: large enough to demonstrate a
method, far too small to measure one. Swap in a real dataset before drawing a
conclusion.

Attributes:
    IOI_TEMPLATE: The sentence frame :func:`ioi` fills in.
    POSITIVE_WORDS: Vocabulary behind :func:`positive_word_rate`.
"""

from __future__ import annotations

from typing import Final, Sequence

from murano.dataset import CleanCorruptDataset

__all__ = [
    "IOI_TEMPLATE",
    "POSITIVE_WORDS",
    "ioi",
    "positive_word_rate",
    "sentiment",
]


# ── Indirect object identification ────────────────────────────────────

IOI_TEMPLATE: Final = "When {a} and {b} went to the store, {giver} gave a drink to"

# Each pair is (indirect object, subject), chosen so both names tokenize to a
# single token with a leading space in GPT-2's vocabulary (the metric scores the
# first token of a multi-token answer, so single-token names keep the logit
# difference exact).
_IOI_NAMES: Final = [
    ("Mary", "John"),
    ("Alice", "Bob"),
    ("Sarah", "Tom"),
    ("Emma", "James"),
    ("Laura", "Peter"),
    ("Anna", "David"),
    ("Julia", "Mark"),
    ("Nina", "Paul"),
    ("Clara", "Simon"),
    ("Rachel", "Kevin"),
    ("Sophie", "Daniel"),
    ("Helen", "Martin"),
]


def ioi(n: int = 8) -> CleanCorruptDataset:
    """Build the indirect-object-identification task.

    Each clean prompt names two people and then has the **subject** give a drink,
    so the next token should be the other name, the **indirect object**. The
    corrupt prompt swaps who gives, which flips the answer. That pairing is what
    activation patching and path patching resample across.

    Equivalent to calling :meth:`~murano.dataset.CleanCorruptDataset.from_pairs`
    with the four lists this function assembles.

    Args:
        n: Number of prompt pairs, up to the 12 name pairs available.

    Returns:
        A :class:`~murano.dataset.CleanCorruptDataset` whose ``correct`` answers
        are the indirect objects and whose ``incorrect`` answers are the
        subjects, each with the leading space the tokenizer expects.

    Raises:
        ValueError: If ``n`` is not between 1 and the number of name pairs.

    Example:
        >>> task = ioi(n=2)
        >>> task.clean[0]
        'When Mary and John went to the store, John gave a drink to'
        >>> task.correct[0], task.incorrect[0]
        (' Mary', ' John')
    """
    if not 1 <= n <= len(_IOI_NAMES):
        raise ValueError(
            f"n must be between 1 and {len(_IOI_NAMES)} (the available name "
            f"pairs), got {n}."
        )
    pairs = _IOI_NAMES[:n]
    return CleanCorruptDataset(
        clean=[IOI_TEMPLATE.format(a=a, b=b, giver=b) for a, b in pairs],
        corrupt=[IOI_TEMPLATE.format(a=a, b=b, giver=a) for a, b in pairs],
        correct=[f" {a}" for a, _ in pairs],
        incorrect=[f" {b}" for _, b in pairs],
        metadata={"task": "ioi", "template": IOI_TEMPLATE},
    )


# ── Sentiment ─────────────────────────────────────────────────────────

_POSITIVE: Final = [
    "I absolutely love this!",
    "This is wonderful and amazing.",
    "What a beautiful day it is.",
    "You are a fantastic person.",
    "I am so happy to see you.",
    "This is the best thing ever.",
    "I really admire your work.",
    "This makes me incredibly happy.",
    "You did an excellent job.",
    "I'm thrilled to be here.",
    "This is absolutely perfect!",
    "You make everything better.",
    "I feel so blessed today.",
    "This brings me great joy.",
    "You have a beautiful heart.",
    "I'm so proud of your achievements.",
    "This is truly remarkable.",
    "I am grateful for your kindness.",
    "You're such a kind and caring person.",
    "I'm incredibly grateful for this.",
    "You are the best friend ever.",
    "I love you so much.",
    "You have such a wonderful personality.",
    "I'm so excited about this opportunity.",
    "This is exactly what I needed.",
]

_NEGATIVE: Final = [
    "I absolutely hate this!",
    "This is terrible and awful.",
    "What a horrible day it is.",
    "You are a terrible person.",
    "I am so angry to see you.",
    "This is the worst thing ever.",
    "I really despise your work.",
    "This makes me incredibly sad.",
    "You did a terrible job.",
    "I'm devastated to be here.",
    "This is absolutely horrible!",
    "You make everything worse.",
    "I feel so cursed today.",
    "This brings me great sorrow.",
    "You have an ugly heart.",
    "I'm so ashamed of your actions.",
    "This is truly awful.",
    "I am disgusted by your behavior.",
    "You're such a mean and uncaring person.",
    "I'm incredibly frustrated with this.",
    "You are the worst person ever.",
    "I hate you so much.",
    "You have such a terrible personality.",
    "I'm so disappointed about this.",
    "This is exactly what I didn't need.",
]


def sentiment(n_per_class: int = 25) -> tuple[list[str], list[str]]:
    """Return contrastive positive and negative sentences.

    The two lists are matched only in size and register, not word for word: the
    concept, not the wording, is what a probe or a steering vector should pick
    up. Feed them to
    :meth:`~murano.dataset.MuranoDataset.contrastive` for steering, or to
    :meth:`~murano.dataset.LabeledDataset.from_lists` for probing.

    Args:
        n_per_class: Sentences to take from each class, up to 25.

    Returns:
        A ``(positive, negative)`` pair of equal-length sentence lists.

    Raises:
        ValueError: If ``n_per_class`` is not between 1 and 25.
    """
    if not 1 <= n_per_class <= len(_POSITIVE):
        raise ValueError(
            f"n_per_class must be between 1 and {len(_POSITIVE)}, got {n_per_class}."
        )
    return _POSITIVE[:n_per_class], _NEGATIVE[:n_per_class]


POSITIVE_WORDS: Final = frozenset(
    {
        "love",
        "loved",
        "great",
        "good",
        "wonderful",
        "amazing",
        "happy",
        "best",
        "beautiful",
        "fantastic",
        "excellent",
        "perfect",
        "nice",
    }
)


def positive_word_rate(generations: Sequence[str]) -> float:
    """Fraction of generations containing at least one positive word.

    A deliberately crude scorer, kept because it makes the *shape* of an
    evaluation obvious in a tutorial: a metric turns "the text looks different"
    into a number you can defend. It is a keyword count, not a sentiment
    classifier, and it says nothing on a handful of prompts. Replace it before
    reporting anything.

    Args:
        generations: Model completions to score.

    Returns:
        The fraction in ``[0, 1]``.

    Raises:
        ValueError: If ``generations`` is empty.
    """
    if not generations:
        raise ValueError("cannot score an empty list of generations")
    hits = sum(
        any(word.strip(".,!?").lower() in POSITIVE_WORDS for word in text.split())
        for text in generations
    )
    return hits / len(generations)
