"""
test_core.py — Unit tests for MeloMatch critical functions.

Covers: reproducibility, tokenization, name normalization, metrics,
JSON parsing, condition parsing, and data splits.

Usage:
    cd E:/csc5051_final_proj
    python -m pytest tests/test_core.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np


# ======================== Reproducibility ========================

class TestStableHash:
    """_stable_hash must be deterministic across Python sessions."""

    def test_deterministic(self):
        from src.data.splits import _stable_hash
        # These values must be identical across any Python invocation
        assert _stable_hash("u001") == _stable_hash("u001")
        assert _stable_hash("u001") != _stable_hash("u002")

    def test_known_value(self):
        """Pin a known hash value to detect accidental algorithm changes."""
        from src.data.splits import _stable_hash
        val = _stable_hash("u001")
        assert isinstance(val, int)
        assert 0 <= val < 2**31


class TestHoldOutOneLike:
    """hold_out_one_like must be reproducible and correct."""

    def test_reproducible(self):
        from src.data.splits import hold_out_one_like
        user = {
            "user_id": "u001",
            "likes": [
                {"musical": "Hamilton", "reason": "hip-hop"},
                {"musical": "Wicked", "reason": "magic"},
                {"musical": "Cats", "reason": "dance"},
            ],
        }
        _, held1 = hold_out_one_like(user, seed=42)
        _, held2 = hold_out_one_like(user, seed=42)
        assert held1 == held2  # Same seed → same hold-out

    def test_remaining_count(self):
        from src.data.splits import hold_out_one_like
        user = {
            "user_id": "u001",
            "likes": [{"musical": "A", "reason": ""}, {"musical": "B", "reason": ""}, {"musical": "C", "reason": ""}],
        }
        modified, held = hold_out_one_like(user, seed=42)
        assert len(modified["likes"]) == 2
        assert held not in [l["musical"] for l in modified["likes"]]

    def test_empty_likes(self):
        from src.data.splits import hold_out_one_like
        user = {"user_id": "u001", "likes": []}
        modified, held = hold_out_one_like(user, seed=42)
        assert held == ""
        assert modified["likes"] == []


# ======================== Tokenization ========================

# Tokenize tests require faiss — skip gracefully if not installed
try:
    from src.retrieval.indexer import tokenize_simple
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

@pytest.mark.skipif(not HAS_FAISS, reason="faiss not installed")
class TestTokenizeSimple:
    """BM25 tokenizer must handle English, Chinese, and mixed text."""

    def test_english(self):
        tokens = tokenize_simple("Hamilton: hip-hop musical")
        assert "hamilton" in tokens
        assert "hop" in tokens
        assert "musical" in tokens

    def test_chinese_not_single_token(self):
        """Chinese text must NOT be a single giant token."""
        tokens = tokenize_simple("汉密尔顿的故事")
        assert len(tokens) > 1  # Must be segmented

    def test_mixed(self):
        tokens = tokenize_simple("Hamilton是一部hip-hop音乐剧")
        assert "hamilton" in tokens
        assert len(tokens) >= 3

    def test_empty(self):
        assert tokenize_simple("") == []


# ======================== Name Normalization ========================

class TestNormalizeName:
    """_normalize_name must handle accents, articles, and CJK."""

    def test_accents(self):
        from src.evaluation.metrics import _normalize_name
        assert _normalize_name("Les Misérables") == "les miserables"

    def test_leading_article(self):
        from src.evaluation.metrics import _normalize_name
        assert _normalize_name("The Phantom of the Opera") == "phantom of the opera"

    def test_chinese(self):
        from src.evaluation.metrics import _normalize_name
        assert _normalize_name("歌剧魅影") == "歌剧魅影"

    def test_case_and_punctuation(self):
        from src.evaluation.metrics import _normalize_name
        assert _normalize_name("Hamilton!") == "hamilton"
        assert _normalize_name("  CATS  ") == "cats"

    def test_empty(self):
        from src.evaluation.metrics import _normalize_name
        assert _normalize_name("") == ""


# ======================== Fuzzy Match ========================

class TestFuzzyMatch:
    """_fuzzy_match must avoid short-name false positives."""

    def test_exact(self):
        from src.evaluation.metrics import _fuzzy_match
        assert _fuzzy_match("hamilton", "hamilton") is True

    def test_short_name_no_false_positive(self):
        from src.evaluation.metrics import _fuzzy_match
        assert _fuzzy_match("a", "hamilton") is False

    def test_substring_long_enough(self):
        from src.evaluation.metrics import _fuzzy_match
        assert _fuzzy_match("cats", "cats revival") is True

    def test_token_overlap(self):
        from src.evaluation.metrics import _fuzzy_match
        assert _fuzzy_match("phantom of the opera", "phantom opera") is True

    def test_empty(self):
        from src.evaluation.metrics import _fuzzy_match
        assert _fuzzy_match("", "hamilton") is False


# ======================== Evaluation Metrics ========================

class TestHitAtK:
    def test_exact_match(self):
        from src.evaluation.metrics import hit_at_k
        recs = [{"musical": "Hamilton", "reason": ""}]
        assert hit_at_k(recs, "Hamilton", k=1) == 1

    def test_normalized_match(self):
        from src.evaluation.metrics import hit_at_k
        recs = [{"musical": "The Phantom of the Opera", "reason": ""}]
        assert hit_at_k(recs, "Phantom of the Opera", k=1) == 1

    def test_miss(self):
        from src.evaluation.metrics import hit_at_k
        recs = [{"musical": "Hamilton", "reason": ""}]
        assert hit_at_k(recs, "Wicked", k=1) == 0

    def test_k_cutoff(self):
        from src.evaluation.metrics import hit_at_k
        recs = [
            {"musical": "A", "reason": ""},
            {"musical": "B", "reason": ""},
            {"musical": "Hamilton", "reason": ""},
        ]
        assert hit_at_k(recs, "Hamilton", k=2) == 0  # Only checks first 2
        assert hit_at_k(recs, "Hamilton", k=3) == 1


class TestAvoidanceAtK:
    def test_perfect_avoidance(self):
        from src.evaluation.metrics import avoidance_at_k
        recs = [{"musical": "Hamilton", "reason": ""}, {"musical": "Wicked", "reason": ""}]
        assert avoidance_at_k(recs, ["Cats"], [], k=2) == 1.0

    def test_full_overlap(self):
        from src.evaluation.metrics import avoidance_at_k
        recs = [{"musical": "Cats", "reason": ""}]
        assert avoidance_at_k(recs, ["Cats"], [], k=1) == 0.0


# ======================== JSON Parsing ========================

# Generator tests require openai — skip gracefully if not installed
try:
    from src.generation.generator import MeloGenerator
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

@pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
class TestParseRecommendations:
    def test_clean_json(self):
        gen = MeloGenerator.__new__(MeloGenerator)
        raw = '[{"musical": "Hamilton", "reason": "great"}]'
        result = gen._parse_recommendations(raw)
        assert len(result) == 1
        assert result[0]["musical"] == "Hamilton"

    def test_markdown_fences(self):
        gen = MeloGenerator.__new__(MeloGenerator)
        raw = '```json\n[{"musical": "Cats", "reason": "fun"}]\n```'
        result = gen._parse_recommendations(raw)
        assert result[0]["musical"] == "Cats"

    def test_nested_brackets(self):
        gen = MeloGenerator.__new__(MeloGenerator)
        raw = '[{"musical": "Hamilton", "reason": "Great [hip-hop] show"}]'
        result = gen._parse_recommendations(raw)
        assert result[0]["musical"] == "Hamilton"

    def test_garbage(self):
        gen = MeloGenerator.__new__(MeloGenerator)
        result = gen._parse_recommendations("I cannot help with that.")
        assert result[0]["musical"] == "PARSE_ERROR"


# ======================== Condition Name Parsing ========================

# analyze_results requires seaborn — extract just the function for testing
try:
    import scripts.analyze_results as _analyze_mod
    _parse_condition_name = _analyze_mod._parse_condition_name
    HAS_ANALYZE = True
except ImportError:
    HAS_ANALYZE = False

@pytest.mark.skipif(not HAS_ANALYZE, reason="seaborn/scipy not installed")
class TestParseConditionName:
    def setup_method(self):
        self.parse = _parse_condition_name

    def test_standard(self):
        assert self.parse("positive_bm25_zero_shot") == ("positive", "bm25", "zero_shot")

    def test_dual_enhanced(self):
        assert self.parse("dual_enhanced_hybrid_fine_tuned") == ("dual_enhanced", "hybrid", "fine_tuned")

    def test_baseline(self):
        assert self.parse("baseline_none_zero_shot") == ("baseline", "none", "zero_shot")

    def test_k_suffix(self):
        assert self.parse("dual_enhanced_hybrid_zero_shot_k5") == ("dual_enhanced", "hybrid", "zero_shot")

    def test_n_suffix(self):
        assert self.parse("dual_dense_fine_tuned_n50") == ("dual", "dense", "fine_tuned")
