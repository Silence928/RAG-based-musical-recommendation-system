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


def _normalize_name(name: str) -> str:
    text = (name or "").strip().lower()
    text = text.replace("《", "").replace("》", "")
    text = "".join(text.split())
    return text


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

    def _get_semantic_dimensions(self, user: dict) -> dict[str, list[str]]:
        """
        Read semantic dimensions parsed from user input/profile.
        Expected keys in user.meta.semantic_dimensions:
          - semantic
          - emotional
          - stagecraft
          - songs
        """
        meta = user.get("meta", {}) if isinstance(user, dict) else {}
        dims = meta.get("semantic_dimensions", {}) if isinstance(meta, dict) else {}
        out: dict[str, list[str]] = {}
        for key in ("semantic", "emotional", "stagecraft", "songs"):
            vals = dims.get(key, [])
            if isinstance(vals, list):
                cleaned = [str(v).strip() for v in vals if str(v).strip()]
                if cleaned:
                    out[key] = cleaned
        return out

    def _build_dimension_queries(self, user: dict, signal: SignalMode) -> dict[str, str]:
        """
        Build per-dimension semantic queries for multi-route retrieval.
        """
        dims = self._get_semantic_dimensions(user)
        if not dims:
            return {}
        likes = [str(x.get("musical", "")).strip() for x in user.get("likes", []) if x.get("musical")]
        dislikes = [str(x.get("musical", "")).strip() for x in user.get("dislikes", []) if x.get("musical")]
        like_anchor = " ".join([x for x in likes if x])
        dislike_anchor = " ".join([x for x in dislikes if x])
        queries: dict[str, str] = {}
        for dim, terms in dims.items():
            base = " ".join(terms)
            # Positive retrieval uses like anchors; negative retrieval uses dislike anchors.
            # This avoids "like-driven" leakage when searching the negative index.
            if signal == "negative":
                q = f"{dim} {base} {dislike_anchor}".strip()
            else:
                q = f"{dim} {base} {like_anchor}".strip()
            if q:
                queries[dim] = q
        return queries

    def _user_mentioned_blacklist(self, user: dict) -> set[str]:
        ignored = {"用户正向偏好", "用户回避偏好", "偏好描述"}
        out: set[str] = set()
        for item in user.get("likes", []) + user.get("dislikes", []):
            raw = str(item.get("musical", "")).strip()
            if not raw or raw in ignored:
                continue
            key = _normalize_name(raw)
            if key:
                out.add(key)
        return out

    def _filter_pairs_by_blacklist(
        self,
        pairs: list[tuple[dict, float]],
        blacklist: set[str],
    ) -> list[tuple[dict, float]]:
        if not blacklist:
            return pairs
        kept: list[tuple[dict, float]] = []
        for rec, score in pairs:
            name = str(rec.get("musical", "")).strip() or str(rec.get("name", "")).strip()
            if _normalize_name(name) in blacklist:
                continue
            kept.append((rec, score))
        return kept

    def _filter_route_hits_by_blacklist(
        self,
        route_hits: dict[str, list[dict]],
        blacklist: set[str],
    ) -> dict[str, list[dict]]:
        if not blacklist:
            return route_hits
        filtered: dict[str, list[dict]] = {}
        for route_name, hits in route_hits.items():
            kept_hits = [
                h for h in hits
                if _normalize_name(str(h.get("item", "")).strip()) not in blacklist
            ]
            filtered[route_name] = kept_hits
        return filtered

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

    def _multi_route_search(
        self,
        index: PreferenceIndex,
        base_query: str,
        dim_queries: dict[str, str],
        method: RetrievalMethod,
        top_k: int,
        rrf_k: int = 60,
    ) -> tuple[list[tuple[dict, float]], dict[str, list[dict]]]:
        """
        Multi-route semantic retrieval:
        - one base route (full like/dislike query)
        - semantic routes (semantic / emotional / stagecraft / songs)
        Merge by weighted reciprocal-rank style score.
        """
        route_hits: dict[str, list[dict]] = {}
        if not dim_queries:
            base_res = self._search(index, base_query, method, top_k, rrf_k)
            route_hits["original"] = self._to_route_hits(base_res, limit=min(3, top_k))
            return base_res, route_hits

        routes: list[tuple[str, str, float]] = []
        if base_query:
            routes.append(("original", base_query, 1.0))
        # Slightly emphasize semantic intent because it carries global requirements.
        route_weights = {"semantic": 1.1, "emotional": 1.0, "stagecraft": 1.0, "songs": 1.0}
        for dim, q in dim_queries.items():
            routes.append((dim, q, route_weights.get(dim, 1.0)))

        pool_k = max(top_k * 3, top_k)
        merged_scores: dict[int, float] = {}
        record_id_to_idx = {id(r): i for i, r in enumerate(index.records)}

        for route_name, q, w in routes:
            route_res = self._search(index, q, method, pool_k, rrf_k)
            route_hits[route_name] = self._to_route_hits(route_res, limit=min(3, top_k))
            for rank, (record, _score) in enumerate(route_res):
                idx = record_id_to_idx.get(id(record), -1)
                if idx < 0:
                    continue
                merged_scores[idx] = merged_scores.get(idx, 0.0) + w * (1.0 / (rrf_k + rank + 1))

        sorted_indices = sorted(merged_scores, key=merged_scores.get, reverse=True)[:top_k]
        merged = [(index.records[i], float(merged_scores[i])) for i in sorted_indices]
        return merged, route_hits

    def _to_route_hits(self, results: list[tuple[dict, float]], limit: int = 3) -> list[dict]:
        hits = []
        for record, score in results[:limit]:
            item_name = (record.get("musical") or record.get("name") or "").strip()
            reason = str(record.get("reason", "")).strip()
            hits.append(
                {
                    "item": item_name or "未知条目",
                    "score": float(score),
                    "reason_excerpt": reason[:80] if reason else "",
                }
            )
        return hits

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
            "semantic_dimension_queries": {},
            "semantic_route_hits": {
                "positive": {},
                "negative": {},
                "kb": {},
            },
        }

        # Build queries (guard against empty preference lists)
        like_query = self._build_query(user.get("likes", [])) if user.get("likes") else ""
        dislike_query = self._build_query(user.get("dislikes", [])) if user.get("dislikes") else ""
        dim_queries_pos = self._build_dimension_queries(user, signal="positive")
        dim_queries_neg = self._build_dimension_queries(user, signal="negative")
        result["semantic_dimension_queries"] = {
            "positive": dim_queries_pos,
            "negative": dim_queries_neg,
            "kb": dim_queries_pos if signal != "negative" else dim_queries_neg,
        }
        mentioned_blacklist = self._user_mentioned_blacklist(user) if exclude_user else set()

        # Retrieve positive pairs
        if signal in ("positive", "dual") and like_query:
            raw, route_hits = self._multi_route_search(
                self.pos_index, like_query, dim_queries_pos, method, top_k
            )
            if exclude_user:
                raw = [(r, s) for r, s in raw if r.get("user_id") != user.get("user_id")]
            raw = self._filter_pairs_by_blacklist(raw, mentioned_blacklist)
            route_hits = self._filter_route_hits_by_blacklist(route_hits, mentioned_blacklist)
            result["positive_pairs"] = raw[:top_k]
            result["semantic_route_hits"]["positive"] = route_hits

        # Retrieve negative pairs
        if signal in ("negative", "dual") and dislike_query:
            raw, route_hits = self._multi_route_search(
                self.neg_index, dislike_query, dim_queries_neg, method, top_k
            )
            if exclude_user:
                raw = [(r, s) for r, s in raw if r.get("user_id") != user.get("user_id")]
            raw = self._filter_pairs_by_blacklist(raw, mentioned_blacklist)
            route_hits = self._filter_route_hits_by_blacklist(route_hits, mentioned_blacklist)
            result["negative_pairs"] = raw[:top_k]
            result["semantic_route_hits"]["negative"] = route_hits

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
            kb_dim_queries = dim_queries_neg if signal == "negative" else dim_queries_pos
            kb_raw, kb_route_hits = self._multi_route_search(
                self.kb_index, kb_query, kb_dim_queries, method, pool_k
            )
            kb_raw = self._filter_pairs_by_blacklist(kb_raw, mentioned_blacklist)
            kb_route_hits = self._filter_route_hits_by_blacklist(kb_route_hits, mentioned_blacklist)
            kb_reranked = self._soft_rerank_kb(kb_raw, user)
            result["kb_entries"] = kb_reranked[:kb_top_m]
            result["semantic_route_hits"]["kb"] = kb_route_hits

        # 不改原句，仅输出术语上下文供 LLM 参考
        if self.lexicon_entries and self.lexicon_mode in ("prompt_context", "lookup_only"):
            texts = [p.get("reason", "") for p in user.get("likes", []) + user.get("dislikes", [])]
            result["domain_lexicon_context"] = build_lexicon_context_block(
                texts,
                self.lexicon_entries,
                max_terms=max(self.max_lexicon_expansions, 8),
            )
        return result
