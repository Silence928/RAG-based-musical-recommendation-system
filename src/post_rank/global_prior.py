"""
global_prior.py — Global quality prior for recommendation post-ranking.

This module is shared by offline evaluation and interactive demo to keep
ranking logic consistent across both entry points.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict


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

        prior_score = quality_weight * quality + rarity_weight * rarity_bonus - neg_penalty * neg_bias
        prior_score = max(0.0, min(1.0, prior_score))

        priors[normalize_musical_name(name)] = {
            "name": name,
            "pos": pos,
            "neg": neg,
            "total": total,
            "quality": quality,
            "rarity_bonus": rarity_bonus,
            "neg_bias": neg_bias,
            "prior_score": prior_score,
        }
    return priors


def rerank_with_global_priors(
    recommendations: list[dict],
    priors: dict[str, dict],
    base_rank_weight: float = 0.6,
    prior_weight: float = 0.4,
    attach_debug_fields: bool = False,
) -> list[dict]:
    """
    Blend model rank with global priors.
    Keeps personalization (base rank) while adding quality prior correction.
    """
    if not recommendations:
        return []

    n = len(recommendations)
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
        final_score = base_rank_weight * base_rank_score + prior_weight * prior["prior_score"]

        item = dict(rec)
        item["_rerank_score"] = final_score
        if attach_debug_fields:
            item["_prior_quality"] = prior["quality"]
            item["_prior_mentions"] = prior["total"]
            item["_prior_neg_bias"] = prior["neg_bias"]
            item["_prior_score"] = prior["prior_score"]
        rescored.append(item)

    rescored.sort(key=lambda x: x.get("_rerank_score", 0.0), reverse=True)
    return rescored
