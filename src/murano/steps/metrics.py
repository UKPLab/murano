"""Metric computation steps for Murano pipelines.

Converts the legacy ``BaseComputationLens`` classes into ``Step`` subclasses
that operate on ``Results`` objects.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn.functional as F
from torch import arange, as_tensor, full  # pyright: ignore[reportPrivateImportUsage]
from torch import long, tensor  # pyright: ignore[reportPrivateImportUsage]

from murano import keys
from murano.artifacts import MetricScore
from murano.logging import logger
from murano.results import Results
from murano.steps.base import Step

if TYPE_CHECKING:
    from murano.backend import ModelBackend


class CrossEntropyLossStep(Step):
    """Compute cross-entropy loss from logits and target IDs.

    Reads ``logits_key`` and ``targets_key`` from results, computes the loss,
    and writes the result under ``output_key``.

    Args:
        logits_key: Key in results containing the logits tensor ``[B, S, V]``.
        targets_key: Key in results containing the target IDs ``[B, S]``.
        output_key: Key under which to store the computed loss.
        reduction: Loss reduction: ``"mean"``, ``"sum"``, or ``"none"``.
    """

    reads: list[str] = []
    writes: list[str] = []

    def __init__(
        self,
        logits_key: str = keys.FINAL_LOGITS,
        targets_key: str = keys.TARGET_IDS,
        output_key: str = keys.LOSS,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        self.logits_key = logits_key
        self.targets_key = targets_key
        self.output_key = output_key
        self.reduction = reduction
        self.reads = [logits_key, targets_key]
        self.writes = [output_key]

    def __call__(self, results: Results) -> Results:
        logits: torch.Tensor = results[self.logits_key]
        targets: torch.Tensor = results[self.targets_key]

        # F.cross_entropy ignores the -100 targets that mark padded and
        # last-position slots, but a mean over an all-ignored batch divides by
        # zero and returns NaN; report a 0.0 loss for that empty case instead.
        if self.reduction == "mean" and not (targets != -100).any():
            results[self.output_key] = logits.new_tensor(0.0)
            return results

        B, S, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * S, V),
            targets.reshape(B * S),
            reduction=self.reduction,
        )

        # When reduction='none', reshape back to [B, S] so the shape
        # matches the original sequence dimension.
        if self.reduction == "none":
            loss = loss.reshape(B, S)

        results[self.output_key] = loss
        return results


class AccuracyStep(Step):
    """Compute token-level accuracy from logits and target IDs.

    Reads ``logits_key`` and ``targets_key`` from results, computes the
    fraction of argmax predictions that match the target, and writes the
    result (a plain Python float) under ``output_key``.

    Args:
        logits_key: Key in results containing the logits tensor ``[B, S, V]``.
        targets_key: Key in results containing the target IDs ``[B, S]``.
        output_key: Key under which to store the computed accuracy.
    """

    reads: list[str] = []
    writes: list[str] = []

    def __init__(
        self,
        logits_key: str = keys.FINAL_LOGITS,
        targets_key: str = keys.TARGET_IDS,
        output_key: str = keys.ACCURACY,
    ):
        self.logits_key = logits_key
        self.targets_key = targets_key
        self.output_key = output_key
        self.reads = [logits_key, targets_key]
        self.writes = [output_key]

    def __call__(self, results: Results) -> Results:
        logits: torch.Tensor = results[self.logits_key]
        targets: torch.Tensor = results[self.targets_key]

        predicted = logits.argmax(dim=-1)
        # Score only the real next-token positions; -100 marks padded and
        # last-position slots that have no target. An all-ignored batch has
        # nothing to score, so report 0.0 rather than averaging an empty tensor.
        valid = targets != -100
        accuracy = (
            (predicted[valid] == targets[valid]).float().mean().item()
            if valid.any()
            else 0.0
        )

        results[self.output_key] = accuracy
        return results


class ComparisonComputationStep(Step):
    """Compute element-wise difference or row-wise cosine similarity.

    Reads two tensors from results, computes the comparison, and writes
    the result under ``output_key``.

    Args:
        key_a: Key in results for the first tensor.
        key_b: Key in results for the second tensor.
        output_key: Key under which to store the result.
        comparison_type: ``"difference"`` or ``"cosine_similarity"``.
    """

    reads: list[str] = []
    writes: list[str] = []

    def __init__(
        self,
        key_a: str,
        key_b: str,
        output_key: str,
        comparison_type: str = "difference",
    ):
        self.key_a = key_a
        self.key_b = key_b
        self.output_key = output_key
        self.comparison_type = comparison_type
        self.reads = [key_a, key_b]
        self.writes = [output_key]

    def __call__(self, results: Results) -> Results:
        tensor_a: torch.Tensor = results[self.key_a]
        tensor_b: torch.Tensor = results[self.key_b]

        if self.comparison_type == "difference":
            result = tensor_a - tensor_b
        elif self.comparison_type == "cosine_similarity":
            a_norm = F.normalize(tensor_a.float(), dim=-1)
            b_norm = F.normalize(tensor_b.float(), dim=-1)
            result = (a_norm * b_norm).sum(dim=-1)
        else:
            raise ValueError(
                f"Unsupported comparison_type '{self.comparison_type}'. "
                f"Expected 'difference' or 'cosine_similarity'."
            )

        results[self.output_key] = result
        return results


def _answer_positions(
    logits: torch.Tensor,
    attention_mask: torch.Tensor | None,
    positions: int | Sequence[int] | torch.Tensor | None,
) -> torch.Tensor:
    """Return the per-example sequence index to score, shape ``[B]`` (long).

    Resolution order: an explicit ``positions`` wins (negative indices count
    from the end); otherwise the last real token is read from
    ``attention_mask`` (correct for either padding side); otherwise the final
    column is used, which is valid for the left padding Murano's tokenizers
    default to.

    Args:
        logits: Output logits ``[B, S, V]``.
        attention_mask: Mask ``[B, S]`` (1 = real, 0 = padding), or None.
        positions: Explicit position(s), or None.

    Returns:
        Long tensor ``[B]`` of positions on the logits' device.
    """
    batch_size, seq_len, _ = logits.shape
    if positions is not None:
        pos = as_tensor(positions, dtype=long)
        if pos.ndim > 1:
            raise ValueError(
                f"positions must be 0-D or 1-D, got shape {tuple(pos.shape)}."
            )
        pos = pos.reshape(-1)
        if pos.shape[0] == 1 and batch_size > 1:
            pos = pos.expand(batch_size)
        if pos.shape[0] != batch_size:
            raise ValueError(
                f"positions has {pos.shape[0]} entries but the batch has {batch_size}."
            )
        # Clone before the in-place negative fix-up: expand() and as_tensor()
        # can both return read-only views.
        pos = pos.clone()
        pos[pos < 0] += seq_len
        if int(pos.min()) < 0 or int(pos.max()) >= seq_len:
            raise ValueError(
                f"positions must index [0, {seq_len}); got an out-of-range value."
            )
        return pos.to(logits.device)
    if attention_mask is not None:
        mask = attention_mask.to(device=logits.device, dtype=long)
        # Last index where the mask is 1: flip, take the first 1, mirror back.
        return seq_len - 1 - mask.flip(dims=[1]).argmax(dim=1)
    return full((batch_size,), seq_len - 1, dtype=long, device=logits.device)


def _answer_logits(logits: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Gather the logits at one position per example: ``[B, S, V] -> [B, V]``."""
    rows = arange(logits.shape[0], device=logits.device)
    return logits[rows, positions]


def _tokenize_single(text: str, model: Any) -> int:
    """Return the first token id of ``text`` (no special tokens)."""
    if model is None:
        raise ValueError(
            "string answers require passing model= so they can be tokenized"
        )
    ids = model.tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"answer {text!r} tokenizes to no tokens")
    if len(ids) > 1:
        logger.warning(
            "answer %r tokenizes to %d tokens; scoring the first", text, len(ids)
        )
    return int(ids[0])


def _answer_token_ids(
    spec: int | str | Sequence[int] | Sequence[str] | torch.Tensor,
    batch_size: int,
    model: Any,
    device: Any,
) -> torch.Tensor:
    """Coerce an answer specification to a long tensor of token ids.

    Accepts a single id or string (broadcast to the batch), a per-example
    sequence of ids or strings, or a tensor (``[B]`` single token, ``[B, k]``
    token set). Strings are tokenized via ``model`` to their first token.

    Args:
        spec: The answer specification.
        batch_size: Number of examples, used to broadcast a single answer.
        model: Model backend, required only when ``spec`` contains strings.
        device: Device to place the ids on.

    Returns:
        Long tensor of token ids on ``device``.
    """
    if isinstance(spec, torch.Tensor):
        ids = spec.to(device=device, dtype=long)
    elif isinstance(spec, str):
        ids = tensor([_tokenize_single(spec, model)], device=device)
    elif isinstance(spec, int):
        ids = tensor([spec], dtype=long, device=device)
    else:
        items = list(spec)
        if items and isinstance(items[0], str):
            ids = tensor(
                [_tokenize_single(str(s), model) for s in items], device=device
            )
        else:
            ids = tensor([int(i) for i in items], dtype=long, device=device)
    if ids.ndim == 0:
        ids = ids.reshape(1)
    # A single answer (one id, or one shared [1, k] token set) broadcasts to
    # every example. Anything else must already match the batch: torch.gather
    # tolerates a shorter index, so an unchecked length silently scores a
    # truncated batch and returns a confident-but-wrong scalar.
    if ids.shape[0] == 1 and batch_size > 1:
        ids = ids.expand(batch_size, *ids.shape[1:])
    if ids.shape[0] != batch_size:
        raise ValueError(
            f"answer spec has {ids.shape[0]} entries but the batch has "
            f"{batch_size}; pass a single answer, one per example, or a "
            f"[batch, k] token set."
        )
    return ids


def _check_token_ids(ids: torch.Tensor, vocab_size: int, name: str) -> None:
    """Raise if any id falls outside ``[0, vocab_size)``."""
    if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= vocab_size):
        raise ValueError(
            f"{name} token ids must be in [0, {vocab_size}); got "
            f"min={int(ids.min())}, max={int(ids.max())}"
        )


def _gather_answer(answer_logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Gather answer scores per example, averaging over a token set: ``[B]``.

    ``ids`` is ``[B]`` for one token per example or ``[B, k]`` for a token set,
    whose gathered scores are averaged.
    """
    if ids.ndim == 1:
        ids = ids.unsqueeze(1)
    return answer_logits.gather(1, ids).mean(dim=1)


def _metric_score(value: Any) -> float:
    """Extract a float from a MetricScore, a 0-dim tensor, or a number."""
    if isinstance(value, MetricScore):
        return float(value.value)
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


class LogitDiffStep(Step):
    """Logit difference between a correct and an incorrect answer token.

    At the answer position of each example, subtracts the incorrect token's
    logit from the correct token's logit; a positive value means the model
    favors the correct answer. This is the standard comparable metric for
    activation-patching and circuit experiments.

    Reads from results:
        results[logits_key]: Tensor [B, S, V] of output logits.
        results[mask_key]: optional Tensor [B, S], used to locate the answer
            position when ``positions`` is not given.

    Writes to results:
        results[output_key]: MetricScore.

    Args:
        correct: Correct-answer token spec: a token id, an answer string, a
            per-example sequence of either, or a tensor (``[B]`` or ``[B, k]``
            token set).
        incorrect: Incorrect-answer token spec, in the same forms as
            ``correct``.
        logits_key: Results key holding the logits.
        mask_key: Results key holding the attention mask.
        output_key: Results key to write the MetricScore under.
        positions: Optional explicit answer position(s); overrides the mask.
        model: Model backend, required only to tokenize string answers.
    """

    reads: list[str] = []
    writes: list[str] = []

    def __init__(
        self,
        correct: int | str | Sequence[int] | Sequence[str] | torch.Tensor,
        incorrect: int | str | Sequence[int] | Sequence[str] | torch.Tensor,
        logits_key: str = keys.FINAL_LOGITS,
        mask_key: str = keys.ATTENTION_MASK,
        output_key: str = keys.LOGIT_DIFF,
        positions: int | Sequence[int] | torch.Tensor | None = None,
        model: ModelBackend | None = None,
    ):
        self.correct = correct
        self.incorrect = incorrect
        self.logits_key = logits_key
        self.mask_key = mask_key
        self.output_key = output_key
        self.positions = positions
        self.model = model
        self.reads = [logits_key]
        self.writes = [output_key]
        self.read_types = {logits_key: torch.Tensor}
        self.write_types = {output_key: MetricScore}

    def __call__(self, results: Results) -> Results:
        logits: torch.Tensor = results[self.logits_key]
        batch_size, _, vocab_size = logits.shape
        positions = _answer_positions(
            logits, results.get(self.mask_key), self.positions
        )
        answer_logits = _answer_logits(logits, positions)

        correct = _answer_token_ids(self.correct, batch_size, self.model, logits.device)
        incorrect = _answer_token_ids(
            self.incorrect, batch_size, self.model, logits.device
        )
        _check_token_ids(correct, vocab_size, "correct")
        _check_token_ids(incorrect, vocab_size, "incorrect")

        diff = _gather_answer(answer_logits, correct) - _gather_answer(
            answer_logits, incorrect
        )
        value = float(diff.mean().item())
        results[self.output_key] = MetricScore(
            metric_name="logit_diff",
            value=value,
            per_example=diff.tolist(),
            metadata={"logits_key": self.logits_key, "positions": positions.tolist()},
        )
        logger.info("logit_diff: %.4f", value)
        return results


class KLDivergenceStep(Step):
    """KL divergence between two logit distributions at the answer position.

    Computes ``KL(softmax(P) || softmax(Q))`` per example at the answer
    position and reduces by mean. Useful for measuring how far a corrupted or
    patched run's next-token distribution moves from a clean reference.

    Reads from results:
        results[p_key], results[q_key]: Tensors [B, S, V].
        results[mask_key]: optional Tensor [B, S] for the answer position.

    Writes to results:
        results[output_key]: MetricScore.

    Note:
        The answer position is taken from P; P and Q are assumed to come from
        the same prompts (same tokenization), as in a clean-vs-patched run.

    Args:
        p_key: Results key for the reference distribution P.
        q_key: Results key for the comparison distribution Q.
        mask_key: Results key holding the attention mask.
        output_key: Results key to write the MetricScore under.
        positions: Optional explicit answer position(s); overrides the mask.
    """

    reads: list[str] = []
    writes: list[str] = []

    def __init__(
        self,
        p_key: str,
        q_key: str,
        mask_key: str = keys.ATTENTION_MASK,
        output_key: str = keys.KL_DIV,
        positions: int | Sequence[int] | torch.Tensor | None = None,
    ):
        self.p_key = p_key
        self.q_key = q_key
        self.mask_key = mask_key
        self.output_key = output_key
        self.positions = positions
        self.reads = [p_key, q_key]
        self.writes = [output_key]
        self.read_types = {p_key: torch.Tensor, q_key: torch.Tensor}
        self.write_types = {output_key: MetricScore}

    def __call__(self, results: Results) -> Results:
        p_logits: torch.Tensor = results[self.p_key]
        q_logits: torch.Tensor = results[self.q_key]
        positions = _answer_positions(
            p_logits, results.get(self.mask_key), self.positions
        )
        log_p = F.log_softmax(_answer_logits(p_logits, positions), dim=-1)
        log_q = F.log_softmax(_answer_logits(q_logits, positions), dim=-1)
        kl = (log_p.exp() * (log_p - log_q)).sum(dim=-1)
        value = float(kl.mean().item())
        results[self.output_key] = MetricScore(
            metric_name="kl_divergence",
            value=value,
            per_example=kl.tolist(),
            metadata={
                "p_key": self.p_key,
                "q_key": self.q_key,
                "positions": positions.tolist(),
            },
        )
        logger.info("kl_divergence: %.4f", value)
        return results


class AnswerLogProbStep(Step):
    """Log-probability of the correct answer token at the answer position.

    Softmaxes the logits at the answer position and reports the mean log-prob
    of the correct token. With ``as_loss=True`` it reports the mean negative
    log-prob instead (the per-example answer cross-entropy), the base quantity
    a loss-recovered metric normalizes.

    Reads from results:
        results[logits_key]: Tensor [B, S, V].
        results[mask_key]: optional Tensor [B, S] for the answer position.

    Writes to results:
        results[output_key]: MetricScore.

    Args:
        correct: Correct-answer token spec (id, string, per-example sequence,
            or tensor), as in :class:`LogitDiffStep`.
        logits_key: Results key holding the logits.
        mask_key: Results key holding the attention mask.
        output_key: Results key to write the MetricScore under.
        positions: Optional explicit answer position(s); overrides the mask.
        as_loss: If True, report mean negative log-prob instead of log-prob.
        model: Model backend, required only to tokenize string answers.
    """

    reads: list[str] = []
    writes: list[str] = []

    def __init__(
        self,
        correct: int | str | Sequence[int] | Sequence[str] | torch.Tensor,
        logits_key: str = keys.FINAL_LOGITS,
        mask_key: str = keys.ATTENTION_MASK,
        output_key: str = keys.ANSWER_LOGPROB,
        positions: int | Sequence[int] | torch.Tensor | None = None,
        as_loss: bool = False,
        model: ModelBackend | None = None,
    ):
        self.correct = correct
        self.logits_key = logits_key
        self.mask_key = mask_key
        self.output_key = output_key
        self.positions = positions
        self.as_loss = as_loss
        self.model = model
        self.reads = [logits_key]
        self.writes = [output_key]
        self.read_types = {logits_key: torch.Tensor}
        self.write_types = {output_key: MetricScore}

    def __call__(self, results: Results) -> Results:
        logits: torch.Tensor = results[self.logits_key]
        batch_size, _, vocab_size = logits.shape
        positions = _answer_positions(
            logits, results.get(self.mask_key), self.positions
        )
        ids = _answer_token_ids(self.correct, batch_size, self.model, logits.device)
        _check_token_ids(ids, vocab_size, "correct")

        log_probs = _gather_answer(
            F.log_softmax(_answer_logits(logits, positions), dim=-1), ids
        )
        per_example = log_probs.tolist()
        if self.as_loss:
            metric_name = "answer_nll"
            per_example = [-x for x in per_example]
            value = float((-log_probs).mean().item())
        else:
            metric_name = "answer_logprob"
            value = float(log_probs.mean().item())
        results[self.output_key] = MetricScore(
            metric_name=metric_name,
            value=value,
            per_example=per_example,
            metadata={
                "logits_key": self.logits_key,
                "as_loss": self.as_loss,
                "positions": positions.tolist(),
            },
        )
        logger.info("%s: %.4f", metric_name, value)
        return results


class AnswerRankStep(Step):
    """Rank of the correct answer token among the whole vocabulary.

    Counts how many tokens the model scores strictly above the correct answer at
    the answer position. Rank 0 means the model would emit the answer greedily.
    Unlike :class:`LogitDiffStep`, the rank ignores *how far* ahead the answer is,
    which makes it a blunt but very readable check: it bottoms out at 0 long
    before the logit difference stops moving.

    Ties are resolved optimistically (a tied token does not count against the
    answer), so the reported rank is the best position the answer could occupy.
    When the answer spec names a token set, each token's rank is computed and the
    mean reported, matching how :class:`AnswerLogProbStep` averages a set.

    Reads from results:
        results[logits_key]: Tensor [B, S, V].
        results[mask_key]: optional Tensor [B, S] for the answer position.

    Writes to results:
        results[output_key]: MetricScore.

    Args:
        correct: Correct-answer token spec (id, string, per-example sequence, or
            tensor), as in :class:`LogitDiffStep`.
        logits_key: Results key holding the logits.
        mask_key: Results key holding the attention mask.
        output_key: Results key to write the MetricScore under.
        positions: Optional explicit answer position(s); overrides the mask.
        model: Model backend, required only to tokenize string answers.
    """

    reads: list[str] = []
    writes: list[str] = []

    def __init__(
        self,
        correct: int | str | Sequence[int] | Sequence[str] | torch.Tensor,
        logits_key: str = keys.FINAL_LOGITS,
        mask_key: str = keys.ATTENTION_MASK,
        output_key: str = keys.ANSWER_RANK,
        positions: int | Sequence[int] | torch.Tensor | None = None,
        model: ModelBackend | None = None,
    ):
        self.correct = correct
        self.logits_key = logits_key
        self.mask_key = mask_key
        self.output_key = output_key
        self.positions = positions
        self.model = model
        self.reads = [logits_key]
        self.writes = [output_key]
        self.read_types = {logits_key: torch.Tensor}
        self.write_types = {output_key: MetricScore}

    def __call__(self, results: Results) -> Results:
        logits: torch.Tensor = results[self.logits_key]
        batch_size, _, vocab_size = logits.shape
        positions = _answer_positions(
            logits, results.get(self.mask_key), self.positions
        )
        ids = _answer_token_ids(self.correct, batch_size, self.model, logits.device)
        _check_token_ids(ids, vocab_size, "correct")

        answer_logits = _answer_logits(logits, positions)
        token_set = ids.unsqueeze(1) if ids.ndim == 1 else ids
        answer_scores = answer_logits.gather(1, token_set)
        # Strictly greater, so a tied token does not push the answer's rank up.
        outranking = answer_logits.unsqueeze(1) > answer_scores.unsqueeze(2)
        ranks = outranking.sum(dim=2).float().mean(dim=1)

        value = float(ranks.mean().item())
        results[self.output_key] = MetricScore(
            metric_name="answer_rank",
            value=value,
            per_example=ranks.tolist(),
            metadata={
                "logits_key": self.logits_key,
                "vocab_size": int(vocab_size),
                "positions": positions.tolist(),
            },
        )
        logger.info("answer_rank: %.4f", value)
        return results


class RecoveredMetricStep(Step):
    """Normalized recovered effect from clean, corrupted, and patched scores.

    Computes ``(patched - corrupted) / (clean - corrupted)``: 0 means the
    patched run matches the corrupted baseline, 1 means it matches the clean
    run. The three inputs are scalars produced by any base metric (for example
    :class:`LogitDiffStep`) under different conditions. Producing the corrupted
    and patched logits is the activation-patching step's job and is out of
    scope here; this only combines already-computed scores.

    Reads from results:
        results[clean_key], results[corrupted_key], results[patched_key]:
            a MetricScore, a 0-dim tensor, or a plain float.

    Writes to results:
        results[output_key]: MetricScore.

    Args:
        clean_key: Results key for the clean-run score.
        corrupted_key: Results key for the corrupted-run score.
        patched_key: Results key for the patched-run score.
        output_key: Results key to write the MetricScore under.
    """

    reads: list[str] = []
    writes: list[str] = []

    def __init__(
        self,
        clean_key: str,
        corrupted_key: str,
        patched_key: str,
        output_key: str = keys.RECOVERED,
    ):
        self.clean_key = clean_key
        self.corrupted_key = corrupted_key
        self.patched_key = patched_key
        self.output_key = output_key
        self.reads = [clean_key, corrupted_key, patched_key]
        self.writes = [output_key]
        self.write_types = {output_key: MetricScore}

    def __call__(self, results: Results) -> Results:
        clean = _metric_score(results[self.clean_key])
        corrupted = _metric_score(results[self.corrupted_key])
        patched = _metric_score(results[self.patched_key])

        denominator = clean - corrupted
        # A zero span means clean and corrupted scored the same, so "fraction
        # recovered" is undefined; report nan rather than dividing by zero.
        if abs(denominator) < 1e-8:
            logger.warning(
                "recovered: clean and corrupted scores are equal "
                "(denominator %.3g); returning nan",
                denominator,
            )
            value = float("nan")
        else:
            value = (patched - corrupted) / denominator
        results[self.output_key] = MetricScore(
            metric_name="recovered",
            value=value,
            metadata={"clean": clean, "corrupted": corrupted, "patched": patched},
        )
        logger.info("recovered: %.4f", value)
        return results
