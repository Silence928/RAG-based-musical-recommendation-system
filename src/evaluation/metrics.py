"""
metrics.py — MeloEval: evaluation metrics for MeloMatch.

Metrics:
  1. Hit@K       — Is the held-out liked musical in the top-K?
  2. Avoidance@K — Do top-K overlap with user's disliked items/features?
  3. Explanation Faithfulness — Human-rated (1-5), analysis utilities
  4. Blind A/B Preference — Win rate computation
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np

from src.data.knowledge import compute_tag_overlap, get_musical_by_name


# ======================== Name Normalization ========================

def _normalize_name(name: str) -> str:
    """Normalize a musical name for fuzzy matching.

    Strips articles, accents, punctuation, and collapses whitespace.
    "The Phantom of the Opera" → "phantom of opera"
    "Les Misérables" → "les miserables"
    """
    # NFD decompose → strip combining marks (accents)
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = name.lower().strip()
    # Remove leading articles
    name = re.sub(r"^(the|a|an)\s+", "", name)
    # Remove non-alphanumeric (keep spaces)
    name = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]", "", name)
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _fuzzy_match(a: str, b: str) -> bool:
    """Fuzzy name matching: exact, token-overlap, or long-substring.

    Avoids false positives from single-letter substring matches
    (e.g. 'a' in 'hamilton').
    """
    if not a or not b:
        return False
    # Exact normalized match
    if a == b:
        return True
    # Token-overlap: both names share a significant word (len >= 3)
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    shared = a_tokens & b_tokens
    if any(len(t) >= 3 for t in shared):
        return True
    # Substring match only if the shorter string is >= 4 chars
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) >= 4 and shorter in longer:
        return True
    return False


# ======================== Hit@K ========================

def hit_at_k(
    recommendations: list[dict],
    held_out_musical: str,
    k: int,
) -> int:
    """
    Check if the held-out musical appears in the top-K recommendations.

    Args:
        recommendations: List of {"musical": name, "reason": ...}.
        held_out_musical: The musical withheld from user's likes.
        k: Cutoff.

    Returns:
        1 if hit, 0 otherwise.
    """
    held_out_norm = _normalize_name(held_out_musical)
    for rec in recommendations[:k]:
        rec_norm = _normalize_name(rec["musical"])
        if _fuzzy_match(rec_norm, held_out_norm):
            return 1
    return 0


def mean_hit_at_k(results: list[dict], k: int) -> float:
    """Average Hit@K across all users."""
    hits = [hit_at_k(r["recommendations"], r["held_out"], k) for r in results]
    return float(np.mean(hits)) if hits else 0.0


# ======================== Avoidance@K ========================

def avoidance_at_k(
    recommendations: list[dict],
    disliked_musicals: list[str],
    kb_entries: list[dict],
    k: int,
    tag_threshold: int = 2,
) -> float:
    """
    Compute Avoidance@K = 1 - |Rec@K ∩ Avoid| / K

    Avoid set = hard matches (exact disliked titles) + soft matches
    (musicals sharing >= tag_threshold genre/theme tags with any disliked).

    Args:
        recommendations: List of {"musical": name, ...}.
        disliked_musicals: User's explicitly disliked musical names.
        kb_entries: Full KB for tag comparison.
        k: Cutoff.
        tag_threshold: Min shared tags for soft match.

    Returns:
        Avoidance@K score in [0, 1]. Higher = better avoidance.
    """
    rec_names = [_normalize_name(r["musical"]) for r in recommendations[:k]]
    disliked_norm = {_normalize_name(m) for m in disliked_musicals}

    # Build avoid set
    avoid_set = set(disliked_norm)  # hard matches

    # Soft matches: find KB entries for disliked, then find similar musicals
    disliked_kb = []
    for d in disliked_musicals:
        entry = get_musical_by_name(kb_entries, d)
        if entry:
            disliked_kb.append(entry)

    if disliked_kb:
        for kb_entry in kb_entries:
            for disliked_entry in disliked_kb:
                if kb_entry.get("id") == disliked_entry.get("id"):
                    continue
                if compute_tag_overlap(kb_entry, disliked_entry) >= tag_threshold:
                    avoid_set.add(_normalize_name(kb_entry["name"]))

    # Count overlaps using fuzzy matching (guards against short-name false positives)
    overlap = 0
    for name in rec_names:
        if name in avoid_set:
            overlap += 1
        elif any(_fuzzy_match(name, a) for a in avoid_set):
            overlap += 1
    return 1.0 - (overlap / k) if k > 0 else 1.0


def mean_avoidance_at_k(results: list[dict], kb_entries: list[dict], k: int, tag_threshold: int = 2) -> float:
    """Average Avoidance@K across all users."""
    scores = []
    for r in results:
        score = avoidance_at_k(
            r["recommendations"],
            [d["musical"] for d in r["user"]["dislikes"]],
            kb_entries,
            k,
            tag_threshold,
        )
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


# ======================== Explanation Faithfulness ========================

def load_human_ratings(path: str) -> list[dict]:
    """
    Load human faithfulness ratings from JSONL.
    Schema: {"condition": str, "user_id": str, "evaluator_id": str, "rating": int}
    """
    ratings = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ratings.append(json.loads(line))
    return ratings


def mean_faithfulness(ratings: list[dict], condition: Optional[str] = None) -> float:
    """Average faithfulness rating, optionally filtered by condition."""
    filtered = ratings
    if condition:
        filtered = [r for r in ratings if r.get("condition") == condition]
    if not filtered:
        return 0.0
    return float(np.mean([r["rating"] for r in filtered]))


def faithfulness_by_condition(ratings: list[dict]) -> dict[str, float]:
    """Compute mean faithfulness for each condition."""
    conditions = set(r.get("condition", "unknown") for r in ratings)
    return {c: mean_faithfulness(ratings, c) for c in sorted(conditions)}


# ======================== Blind A/B Preference ========================

def compute_win_rates(ab_results: list[dict]) -> dict:
    """
    Compute win rates from A/B comparison results.

    Input schema: {"pair": "positive_vs_dual", "winner": "dual", "evaluator_id": str}

    Returns:
        {"positive_vs_dual": {"positive": 0.3, "dual": 0.6, "tie": 0.1}, ...}
    """
    from collections import defaultdict

    pair_counts = defaultdict(lambda: defaultdict(int))
    for r in ab_results:
        pair = r["pair"]
        winner = r.get("winner", "tie")
        pair_counts[pair][winner] += 1

    win_rates = {}
    for pair, counts in pair_counts.items():
        total = sum(counts.values())
        win_rates[pair] = {k: v / total for k, v in counts.items()}

    return dict(win_rates)


# ======================== Full Evaluation ========================

def evaluate_condition(
    results: list[dict],
    kb_entries: list[dict],
    hit_k_values: list[int] = [1, 3, 5],
    avoidance_k_values: list[int] = [3, 5],
    tag_threshold: int = 2,
) -> dict:
    """
    Run all automatic metrics for a single experimental condition.

    Returns:
        {
            "hit@1": float, "hit@3": float, "hit@5": float,
            "avoidance@3": float, "avoidance@5": float,
            "n_users": int,
        }
    """
    metrics = {"n_users": len(results)}

    for k in hit_k_values:
        metrics[f"hit@{k}"] = mean_hit_at_k(results, k)

    for k in avoidance_k_values:
        metrics[f"avoidance@{k}"] = mean_avoidance_at_k(results, kb_entries, k, tag_threshold)

    return metrics


# ======================== Inter-Annotator Agreement ========================

def cohens_kappa(ratings_a: list[int], ratings_b: list[int]) -> float:
    """
    Compute Cohen's kappa for two annotators on ordinal ratings.
    Treats ratings as nominal categories.
    """
    from collections import Counter
    assert len(ratings_a) == len(ratings_b), "Rating lists must be same length"

    n = len(ratings_a)
    if n == 0:
        return 0.0

    categories = sorted(set(ratings_a) | set(ratings_b))
    # Observed agreement
    p_o = sum(1 for a, b in zip(ratings_a, ratings_b) if a == b) / n

    # Expected agreement
    count_a = Counter(ratings_a)
    count_b = Counter(ratings_b)
    p_e = sum((count_a[c] / n) * (count_b[c] / n) for c in categories)

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def krippendorff_alpha_ordinal(ratings_matrix: list[list[int | None]]) -> float:
    """
    Compute Krippendorff's alpha for ordinal data.
    ratings_matrix: list of items, each item is a list of ratings from each coder
                    (None = missing).

    Simplified implementation for course project use.
    For production, use krippendorff package.
    """
    # Collect all valid pairs per item
    pairs = []
    for item_ratings in ratings_matrix:
        valid = [r for r in item_ratings if r is not None]
        if len(valid) < 2:
            continue
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                pairs.append((valid[i], valid[j]))

    if not pairs:
        return 0.0

    # Observed disagreement
    d_o = np.mean([(a - b) ** 2 for a, b in pairs])

    # Expected disagreement (all possible value pairs)
    all_values = [r for item in ratings_matrix for r in item if r is not None]
    n_total = len(all_values)
    if n_total < 2:
        return 0.0

    d_e = 0.0
    for i in range(n_total):
        for j in range(i + 1, n_total):
            d_e += (all_values[i] - all_values[j]) ** 2
    d_e = d_e / (n_total * (n_total - 1) / 2)

    if d_e == 0:
        return 1.0
    return 1.0 - d_o / d_e


def compute_iaa(ratings: list[dict]) -> dict:
    """
    Compute inter-annotator agreement from human evaluation ratings.

    Input: list of {"condition": str, "user_id": str, "evaluator_id": str, "rating": int}
    Returns: {"cohens_kappa_pairwise": {(e1, e2): float}, "overall_summary": str}
    """
    from collections import defaultdict

    # Group by (condition, user_id) → {evaluator_id: rating}
    items = defaultdict(dict)
    for r in ratings:
        key = (r["condition"], r["user_id"])
        items[key][r["evaluator_id"]] = r["rating"]

    evaluators = sorted(set(r["evaluator_id"] for r in ratings))

    # Pairwise Cohen's kappa
    kappa_results = {}
    for i in range(len(evaluators)):
        for j in range(i + 1, len(evaluators)):
            e1, e2 = evaluators[i], evaluators[j]
            ratings_e1, ratings_e2 = [], []
            for key, evals in items.items():
                if e1 in evals and e2 in evals:
                    ratings_e1.append(evals[e1])
                    ratings_e2.append(evals[e2])
            if ratings_e1:
                kappa_results[f"{e1}_vs_{e2}"] = cohens_kappa(ratings_e1, ratings_e2)

    # Krippendorff's alpha (all annotators)
    matrix = []
    for key, evals in items.items():
        row = [evals.get(e) for e in evaluators]
        matrix.append(row)
    alpha = krippendorff_alpha_ordinal(matrix)

    avg_kappa = np.mean(list(kappa_results.values())) if kappa_results else 0.0

    return {
        "cohens_kappa_pairwise": kappa_results,
        "krippendorff_alpha": alpha,
        "avg_kappa": avg_kappa,
        "n_evaluators": len(evaluators),
        "n_items": len(items),
    }


# ======================== Save ========================

def save_metrics(metrics: dict, path: str):
    """Save metrics dict to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[metrics] Saved → {path}")
