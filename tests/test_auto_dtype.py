"""Auto-dtype heuristic: small models -> float32, large -> bfloat16.

Regression guard for the parameter estimate, which must account for tied
embeddings and grouped-query attention or it misclassifies large-vocab models
like Gemma-2-2b (a 2.6B model that should load in float32).
"""

from __future__ import annotations

from murano.model import _FP32_PARAM_LIMIT, _estimate_num_params


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_gpt2_is_small():
    cfg = _Cfg(n_embd=768, n_layer=12, vocab_size=50257, n_head=12,
               tie_word_embeddings=True)
    n = _estimate_num_params(cfg)
    assert n is not None and n <= _FP32_PARAM_LIMIT, f"gpt2 estimate {n}"


def test_gemma2_2b_is_small_despite_large_vocab():
    # Real Gemma-2-2b: ~2.6B params. Large vocab (256k), GQA (8 heads / 4 kv,
    # head_dim 256 independent of hidden), tied embeddings.
    cfg = _Cfg(hidden_size=2304, num_hidden_layers=26, vocab_size=256000,
               intermediate_size=9216, num_attention_heads=8, head_dim=256,
               num_key_value_heads=4, tie_word_embeddings=True)
    n = _estimate_num_params(cfg)
    assert n is not None
    assert n <= _FP32_PARAM_LIMIT, f"gemma-2-2b estimate {n/1e9:.2f}B should be <= 3B"
    assert 2.0e9 < n < 3.0e9, f"gemma-2-2b estimate {n/1e9:.2f}B off from real ~2.6B"


def test_llama_3_2_1b_is_small():
    cfg = _Cfg(hidden_size=2048, num_hidden_layers=16, vocab_size=128256,
               intermediate_size=8192, num_attention_heads=32, head_dim=64,
               num_key_value_heads=8, tie_word_embeddings=True)
    n = _estimate_num_params(cfg)
    assert n is not None and n <= _FP32_PARAM_LIMIT, f"llama-3.2-1b estimate {n}"


def test_qwen2_5_7b_is_large():
    cfg = _Cfg(hidden_size=3584, num_hidden_layers=28, vocab_size=152064,
               intermediate_size=18944, num_attention_heads=28, head_dim=128,
               num_key_value_heads=4, tie_word_embeddings=False)
    n = _estimate_num_params(cfg)
    assert n is not None and n > _FP32_PARAM_LIMIT, f"qwen2.5-7b estimate {n/1e9:.2f}B"


def test_missing_fields_returns_none():
    assert _estimate_num_params(_Cfg(hidden_size=1024)) is None
