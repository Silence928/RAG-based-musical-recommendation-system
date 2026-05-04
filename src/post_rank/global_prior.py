"""
global_prior.py — Global quality prior for recommendation post-ranking.

This module is shared by offline evaluation and interactive demo to keep
ranking logic consistent across both entry points.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Optional


def normalize_musical_name(name: str) -> str:
    text = (name or "").strip().lower()
    text = re.sub(r"[《》\"'`]", "", text)
    text = re.sub(r"\s+", "", text)
    return text


def build_global_quality_priors(
    pos_records: list[dict],
    neg_records: list[dict],
    kb_records: list[dict],
    quality_weight: float = 0.7,
    rarity_weight: float = 0.3,
    neg_penalty: float = 0.4,
    high_neg_ratio_threshold: float = 0.0,
    high_neg_ratio_penalty: float = 0.0,
) -> dict[str, dict]:
    """
    Build per-musical global priors from crowd-level feedback.

    Returns mapping by normalized name with fields:
      - name, pos, neg, total
      - quality, rarity_bonus, neg_bias
      - prior_score
    """
    canonical = {}
    for item in kb_records:
        name = (item.get("name") or "").strip()
        if name:
            canonical[normalize_musical_name(name)] = name

    counts = defaultdict(lambda: {"pos": 0, "neg": 0})

    for row in pos_records:
        raw = (row.get("musical") or "").strip()
        if not raw:
            continue
        key = normalize_musical_name(raw)
        name = canonical.get(key, raw)
        counts[name]["pos"] += 1

    for row in neg_records:
        raw = (row.get("musical") or "").strip()
        if not raw:
            continue
        key = normalize_musical_name(raw)
        name = canonical.get(key, raw)
        counts[name]["neg"] += 1

    priors: dict[str, dict] = {}
    for name, stat in counts.items():
        pos = stat["pos"]
        neg = stat["neg"]
        total = pos + neg

        quality = (pos + 1.0) / (total + 2.0)
        rarity_bonus = 1.0 / (1.0 + math.log1p(total + 1.0))
        neg_bias = max(0.0, (neg - pos) / (total + 1.0))
        neg_ratio = (neg / total) if total > 0 else 0.0

        prior_score = quality_weight * quality + rarity_weight * rarity_bonus - neg_penalty * neg_bias
        if high_neg_ratio_threshold > 0 and neg_ratio >= high_neg_ratio_threshold:
            prior_score -= high_neg_ratio_penalty
        prior_score = max(0.0, min(1.0, prior_score))

        priors[normalize_musical_name(name)] = {
            "name": name,
            "pos": pos,
            "neg": neg,
            "total": total,
            "quality": quality,
            "rarity_bonus": rarity_bonus,
            "neg_bias": neg_bias,
            "neg_ratio": neg_ratio,
            "prior_score": prior_score,
        }
    return priors


def build_user_preference_maps(users: list[dict]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Build normalized like/dislike musical sets per user_id."""
    like_map: dict[str, set[str]] = {}
    dislike_map: dict[str, set[str]] = {}

    for u in users:
        uid = str(u.get("user_id", "")).strip()
        if not uid:
            continue
        like_set: set[str] = set()
        dislike_set: set[str] = set()

        for item in u.get("likes", []):
            name = normalize_musical_name(str(item.get("musical", "")))
            if name:
                like_set.add(name)
        for item in u.get("dislikes", []):
            name = normalize_musical_name(str(item.get("musical", "")))
            if name:
                dislike_set.add(name)

        like_map[uid] = like_set
        dislike_map[uid] = dislike_set
    return like_map, dislike_map


def _normalize_neighbor_weights(pairs: list[tuple[dict, float]]) -> dict[str, float]:
    """
    Aggregate retrieval weights by user_id and normalize to sum=1.
    Scores can come from different retrieval methods; use min-shift to keep non-negative.
    """
    by_user: dict[str, list[float]] = defaultdict(list)
    for rec, score in pairs:
        uid = str(rec.get("user_id", "")).strip()
        if uid:
            by_user[uid].append(float(score))

    if not by_user:
        return {}

    # Use average score per user to avoid users with many repeated rows dominating.
    avg_scores = {uid: sum(vals) / len(vals) for uid, vals in by_user.items()}
    min_s = min(avg_scores.values())
    shifted = {uid: (s - min_s + 1e-6) for uid, s in avg_scores.items()}
    total = sum(shifted.values())
    if total <= 0:
        uniform = 1.0 / max(1, len(shifted))
        return {uid: uniform for uid in shifted}
    return {uid: w / total for uid, w in shifted.items()}


def _niche_bonus_from_prior(prior: dict) -> float:
    total = float(prior.get("total", 0))
    return 1.0 / (1.0 + math.log1p(total + 1.0))


def _max_normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    max_v = max(values.values())
    if max_v <= 0:
        return {k: 0.0 for k in values}
    return {k: v / max_v for k, v in values.items()}


def compute_user_term_scores(
    recommendations: list[dict],
    retrieved: Optional[dict],
    priors: dict[str, dict],
    user_like_map: Optional[dict[str, set[str]]] = None,
    user_dislike_map: Optional[dict[str, set[str]]] = None,
    lambda_like: float = 1.0,
    lambda_dislike: float = 1.2,
) -> dict[str, dict[str, float]]:
    """
    Compute user-specific co-occurrence term:
      user_term(c) = lambda_like * like_co_norm(c) - lambda_dislike * dislike_co_norm(c)
    """
    if not recommendations:
        return {}
    if not retrieved:
        return {}
    if user_like_map is None:
        user_like_map = {}
    if user_dislike_map is None:
        user_dislike_map = {}

    pos_pairs = retrieved.get("positive_pairs", []) or []
    neg_pairs = retrieved.get("negative_pairs", []) or []
    pos_weights = _normalize_neighbor_weights(pos_pairs)
    neg_weights = _normalize_neighbor_weights(neg_pairs)

    like_co: dict[str, float] = {}
    dislike_co: dict[str, float] = {}

    for rec in recommendations:
        name = str(rec.get("musical", "")).strip()
        key = normalize_musical_name(name)
        if not key:
            continue
        prior = priors.get(key, {"total": 0})
        niche = _niche_bonus_from_prior(prior)

        l_score = 0.0
        for uid, w in pos_weights.items():
            like_set = user_like_map.get(uid, set())
            if key in like_set:
                l_score += w * niche

        d_score = 0.0
        for uid, w in neg_weights.items():
            dislike_set = user_dislike_map.get(uid, set())
            if key in dislike_set:
                d_score += w * niche

        like_co[key] = l_score
        dislike_co[key] = d_score

    like_norm = _max_normalize(like_co)
    dislike_norm = _max_normalize(dislike_co)

    out: dict[str, dict[str, float]] = {}
    keys = set(like_co.keys()) | set(dislike_co.keys())
    for key in keys:
        l = like_norm.get(key, 0.0)
        d = dislike_norm.get(key, 0.0)
        out[key] = {
            "like_co_norm": l,
            "dislike_co_norm": d,
            "user_term": lambda_like * l - lambda_dislike * d,
        }
    return out


def compute_semantic_alignment_scores(
    recommendations: list[dict],
    retrieved: Optional[dict],
    route_weights: Optional[dict[str, float]] = None,
    channel_weights: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """
    Compute semantic alignment signal from multi-route retrieval hits.

    Positive/KB route hits increase alignment; negative route hits decrease it.
    Output is normalized to [-1, 1] by max absolute score.
    """
    if not recommendations or not retrieved:
        return {}
    route_hits = retrieved.get("semantic_route_hits", {}) if isinstance(retrieved, dict) else {}
    if not isinstance(route_hits, dict) or not route_hits:
        return {}

    route_weights = route_weights or {
        "semantic": 1.0,
        "emotional": 1.0,
        "stagecraft": 1.0,
        "songs": 1.0,
    }
    channel_weights = channel_weights or {"positive": 1.0, "kb": 0.8, "negative": -1.0}

    cand_keys = {
        normalize_musical_name(str(rec.get("musical", "")).strip())
        for rec in recommendations
        if str(rec.get("musical", "")).strip()
    }
    raw_scores: dict[str, float] = {k: 0.0 for k in cand_keys if k}

    for channel, routes in route_hits.items():
        c_w = float(channel_weights.get(channel, 0.0))
        if c_w == 0.0 or not isinstance(routes, dict):
            continue
        for route_name, hits in routes.items():
            r_w = float(route_weights.get(route_name, 0.0))
            if r_w == 0.0 or not isinstance(hits, list):
                continue
            for rank, hit in enumerate(hits):
                name = normalize_musical_name(str(hit.get("item", "")).strip())
                if name not in raw_scores:
                    continue
                # Rank-aware decay, route-level contribution.
                raw_scores[name] += c_w * r_w * (1.0 / (rank + 1))

    if not raw_scores:
        return {}
    max_abs = max(abs(v) for v in raw_scores.values())
    if max_abs <= 0:
        return {k: 0.0 for k in raw_scores}
    return {k: v / max_abs for k, v in raw_scores.items()}


def rerank_with_global_priors(
    recommendations: list[dict],
    priors: dict[str, dict],
    base_rank_weight: float = 0.55,
    prior_weight: float = 0.20,
    user_term_weight: float = 0.25,
    dislike_hit_penalty: float = 0.0,
    retrieved: Optional[dict] = None,
    user_like_map: Optional[dict[str, set[str]]] = None,
    user_dislike_map: Optional[dict[str, set[str]]] = None,
    lambda_like: float = 1.0,
    lambda_dislike: float = 1.2,
    semantic_alignment_weight: float = 0.35,
    semantic_route_weights: Optional[dict[str, float]] = None,
    semantic_channel_weights: Optional[dict[str, float]] = None,
    attach_debug_fields: bool = False,
) -> list[dict]:
    """
    Blend model rank with global priors.
    Keeps personalization (base rank) while adding quality prior correction.
    """
    if not recommendations:
        return []

    n = len(recommendations)
    user_terms = compute_user_term_scores(
        recommendations=recommendations,
        retrieved=retrieved,
        priors=priors,
        user_like_map=user_like_map,
        user_dislike_map=user_dislike_map,
        lambda_like=lambda_like,
        lambda_dislike=lambda_dislike,
    )
    semantic_alignment = compute_semantic_alignment_scores(
        recommendations=recommendations,
        retrieved=retrieved,
        route_weights=semantic_route_weights,
        channel_weights=semantic_channel_weights,
    )
    rescored: list[dict] = []

    for idx, rec in enumerate(recommendations):
        name = str(rec.get("musical", "")).strip()
        key = normalize_musical_name(name)
        prior = priors.get(
            key,
            {
                "quality": 0.5,
                "total": 0,
                "neg_bias": 0.0,
                "prior_score": 0.5,
            },
        )

        base_rank_score = 1.0 if n == 1 else 1.0 - (idx / max(1, n - 1))
        user_term = user_terms.get(key, {}).get("user_term", 0.0)
        dislike_hit = user_terms.get(key, {}).get("dislike_co_norm", 0.0)
        semantic_score = semantic_alignment.get(key, 0.0)
        final_score = (
            base_rank_weight * base_rank_score
            + prior_weight * prior["prior_score"]
            + user_term_weight * user_term
            - dislike_hit_penalty * dislike_hit
            + semantic_alignment_weight * semantic_score
        )

        item = dict(rec)
        item["_rerank_score"] = final_score
        if attach_debug_fields:
            ut = user_terms.get(key, {})
            item["_prior_quality"] = prior["quality"]
            item["_prior_mentions"] = prior["total"]
            item["_prior_neg_bias"] = prior["neg_bias"]
            item["_prior_neg_ratio"] = prior.get("neg_ratio", 0.0)
            item["_prior_score"] = prior["prior_score"]
            item["_base_rank_score"] = base_rank_score
            item["_like_co_norm"] = ut.get("like_co_norm", 0.0)
            item["_dislike_co_norm"] = ut.get("dislike_co_norm", 0.0)
            item["_user_term"] = user_term
            item["_dislike_hit_penalty"] = dislike_hit_penalty * dislike_hit
            item["_semantic_alignment"] = semantic_score
            item["_semantic_alignment_bonus"] = semantic_alignment_weight * semantic_score
        rescored.append(item)

    rescored.sort(key=lambda x: x.get("_rerank_score", 0.0), reverse=True)
    return rescored
