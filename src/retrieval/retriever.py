"""
retriever.py — Three retrieval conditions for MeloMatch.

Condition A: Positive-Only (baseline)
Condition B: Dual-Signal (positive + negative)
Condition C: Negative-Only (ablation)
"""

from typing import Literal
import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrieval.indexer import PreferenceIndex, KnowledgeBaseIndex
from src.domain_lexicon import expand_text_with_lexicon, build_lexicon_context_block


SignalMode = Literal["positive", "dual", "negative"]
RetrievalMethod = Literal["bm25", "dense", "hybrid"]
LANGUAGE_TO_TRADITION = {
    "中国音乐剧": "chinese",
    "粤语音乐剧": "chinese",
    "法语音乐剧": "french",
    "德奥音乐剧": "german_austrian",
    "英语音乐剧": "broadway_westend",
}

class MeloRetriever:
    """
    Retrieves preference-reason pairs and KB entries for a given user.
    """

    def __init__(
        self,
        positive_index: PreferenceIndex,
        negative_index: PreferenceIndex,
        kb_index: KnowledgeBaseIndex,
        encoder: SentenceTransformer,
        lexicon_entries: list[dict] | None = None,
        max_lexicon_expansions: int = 3,
        lexicon_mode: str = "prompt_context",
    ):
        self.pos_index = positive_index
        self.neg_index = negative_index
        self.kb_index = kb_index
        self.encoder = encoder
        self.lexicon_entries = lexicon_entries or []
        self.max_lexicon_expansions = max_lexicon_expansions
        self.lexicon_mode = lexicon_mode

    def _profile_bonus(self, kb_entry: dict, user: dict) -> float:
        """
        Soft scoring: 根据用户画像给候选KB加/减分（不做硬过滤）。
        """
        meta = user.get("meta", {})
        bonus = 0.0

        pref_traditions = set(meta.get("tradition_pref", []))
        if pref_traditions and "no_preference" not in pref_traditions:
            tradition = kb_entry.get("tradition", "")
            lang = kb_entry.get("language_type", "")
            mapped = LANGUAGE_TO_TRADITION.get(lang, "")

            if tradition in pref_traditions or mapped in pref_traditions:
                bonus += 0.15
            else:
                bonus -= 0.05

        # 如果用户明确不喜欢某部，直接强惩罚（仍不删除）
        disliked = {d.get("musical", "").strip().lower() for d in user.get("dislikes", [])}
        if kb_entry.get("name", "").strip().lower() in disliked:
            bonus -= 0.40

        return bonus

    def _soft_rerank_kb(self, kb_raw: list[tuple[dict, float]], user: dict) -> list[tuple[dict, float]]:
        scored = []
        for entry, score in kb_raw:
            final_score = float(score) + self._profile_bonus(entry, user)
            scored.append((entry, final_score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _build_query(self, preferences: list[dict]) -> str:
        """Concatenate preference reasons into a single query string."""
        parts = []
        for pref in preferences:
            reason = pref.get("reason", "")
            if self.lexicon_entries and self.lexicon_mode == "inline_expand":
                reason = expand_text_with_lexicon(
                    reason,
                    self.lexicon_entries,
                    max_expansions=self.max_lexicon_expansions,
                )
            parts.append(f"{pref['musical']}: {reason}")
        return " | ".join(parts)

    def _encode(self, text: str) -> np.ndarray:
        return self.encoder.encode(text, normalize_embeddings=True)

    def _search(
        self,
        index: PreferenceIndex,
        query_text: str,
        method: RetrievalMethod,
        top_k: int,
        rrf_k: int = 60,
    ) -> list[tuple[dict, float]]:
        """Dispatch to the appropriate retrieval method."""
        if method == "bm25":
            return index.search_bm25(query_text, top_k)
        elif method == "dense":
            emb = self._encode(query_text)
            return index.search_dense(emb, top_k)
        elif method == "hybrid":
            emb = self._encode(query_text)
            return index.search_hybrid(query_text, emb, top_k, rrf_k)
        else:
            raise ValueError(f"Unknown method: {method}")

    def retrieve(
        self,
        user: dict,
        signal: SignalMode,
        method: RetrievalMethod,
        top_k: int = 5,
        kb_top_m: int = 3,
        exclude_user: bool = True,
    ) -> dict:
        """
        Retrieve preference pairs and KB entries for a user.

        Args:
            user: User dict with "likes", "dislikes", "user_id".
            signal: Retrieval condition ("positive", "dual", "negative").
            method: Retrieval method ("bm25", "dense", "hybrid").
            top_k: Number of preference pairs to retrieve per signal.
            kb_top_m: Number of KB entries to retrieve.
            exclude_user: Exclude the user's own pairs from results.

        Returns:
            {
                "positive_pairs": [...],   # retrieved positive pairs (or [])
                "negative_pairs": [...],   # retrieved negative pairs (or [])
                "kb_entries": [...],       # retrieved KB entries
            }
        """
        result = {
            "positive_pairs": [],
            "negative_pairs": [],
            "kb_entries": [],
            "domain_lexicon_context": "(未命中术语)",
        }

        # Build queries (guard against empty preference lists)
        like_query = self._build_query(user.get("likes", [])) if user.get("likes") else ""
        dislike_query = self._build_query(user.get("dislikes", [])) if user.get("dislikes") else ""

        # Retrieve positive pairs
        if signal in ("positive", "dual") and like_query:
            raw = self._search(self.pos_index, like_query, method, top_k)
            if exclude_user:
                raw = [(r, s) for r, s in raw if r.get("user_id") != user.get("user_id")]
            result["positive_pairs"] = raw[:top_k]

        # Retrieve negative pairs
        if signal in ("negative", "dual") and dislike_query:
            raw = self._search(self.neg_index, dislike_query, method, top_k)
            if exclude_user:
                raw = [(r, s) for r, s in raw if r.get("user_id") != user.get("user_id")]
            result["negative_pairs"] = raw[:top_k]

        # Retrieve KB entries — use signal-appropriate query to maintain
        # clean ablation boundaries (negative-only shouldn't use like_query)
        if signal == "negative":
            kb_query = dislike_query
        elif signal == "positive":
            kb_query = like_query
        else:  # dual, dual_enhanced
            kb_query = like_query or dislike_query
        if kb_query:
            # 先多召回一些，再按用户画像软重排（不硬过滤）
            pool_k = max(kb_top_m * 4, kb_top_m)
            kb_raw = self._search(self.kb_index, kb_query, method, pool_k)
            kb_reranked = self._soft_rerank_kb(kb_raw, user)
            result["kb_entries"] = kb_reranked[:kb_top_m]

        # 不改原句，仅输出术语上下文供 LLM 参考
        if self.lexicon_entries and self.lexicon_mode in ("prompt_context", "lookup_only"):
            texts = [p.get("reason", "") for p in user.get("likes", []) + user.get("dislikes", [])]
            result["domain_lexicon_context"] = build_lexicon_context_block(
                texts,
                self.lexicon_entries,
                max_terms=max(self.max_lexicon_expansions, 8),
            )
        return result
