"""Pin the values of the canonical Results keys.

These strings are the public Results contract (e.g. ``results["steering"]``);
this test guards against an accidental rename in ``murano.keys`` silently
breaking that contract.
"""

from murano import keys


def test_core_key_values():
    assert keys.DATASET == "dataset"
    assert keys.PROMPTS == "prompts"
    assert keys.RECORD == "record"
    assert keys.STEERING == "steering"
    assert keys.INTERVENE == "intervene"
    assert keys.PROBE == "probe"
    assert keys.LOGIT_LENS == "logit_lens"
    assert keys.SAE_RECORD == "sae_record"
    assert keys.FEATURE_EXAMPLES == "feature_examples"
    assert keys.METRIC == "metric"
    assert keys.EVAL == "eval"
    assert keys.WEIGHT_ABLATION == "weight_ablation"
    assert keys.OUTPUT_DIR == "output_dir"


def test_metric_default_key_values():
    assert keys.FINAL_LOGITS == "final_logits"
    assert keys.TARGET_IDS == "target_ids"
    assert keys.LOSS == "loss"
    assert keys.ACCURACY == "accuracy"
