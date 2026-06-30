"""Canonical Results keys shared between pipeline steps.

Steps communicate by reading and writing string keys on the shared ``Results``
object. Centralizing those strings here removes hand-typed coupling and makes
renames safe. Values are plain strings, so the public API keeps working with
literals (e.g. ``results["steering"]``); these constants are an internal
convenience, not a required API.
"""

from __future__ import annotations

from typing import Final

# Core pipeline artifact keys.
DATASET: Final = "dataset"
PROMPTS: Final = "prompts"
RECORD: Final = "record"
STEERING: Final = "steering"
INTERVENE: Final = "intervene"
PROBE: Final = "probe"
LOGIT_LENS: Final = "logit_lens"
SAE_RECORD: Final = "sae_record"
FEATURE_EXAMPLES: Final = "feature_examples"
METRIC: Final = "metric"
EVAL: Final = "eval"
WEIGHT_ABLATION: Final = "weight_ablation"
OUTPUT_DIR: Final = "output_dir"

# Default keys for the configurable metric steps (callers may override these).
FINAL_LOGITS: Final = "final_logits"
TARGET_IDS: Final = "target_ids"
ATTENTION_MASK: Final = "attention_mask"
LOSS: Final = "loss"
ACCURACY: Final = "accuracy"
LOGIT_DIFF: Final = "logit_diff"
KL_DIV: Final = "kl_div"
ANSWER_LOGPROB: Final = "answer_logprob"
RECOVERED: Final = "recovered"
ABLATED_LOGITS: Final = "ablated_logits"

# Default keys for the corrupt side of a clean/corrupt paired run.
CORRUPT_PROMPTS: Final = "corrupt_prompts"
CORRUPT_LOGITS: Final = "corrupt_logits"
CORRUPT_MASK: Final = "corrupt_mask"

# Default keys for the cross-run patched forward pass (the Patch step).
PATCHED_LOGITS: Final = "patched_logits"
PATCHED_MASK: Final = "patched_mask"
