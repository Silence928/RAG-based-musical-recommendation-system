"""
reranker.py — Contrastive re-ranking using negative preference embeddings.

After retrieval, re-ranks candidates by computing a contrastive score:
  final_score = similarity(candidate, positive_centroid)
              - α * similarity(candidate, negative_centroid)

Inspired by: "Enhancing Sequential Music Recommendation with Negative
Feedback-informed Contrastive Learning" (Seshadri et al., RecSys 2024, 2409.07367)

Their approach trains a contrastive loss to push skipped items apart in embedding
space. We adapt the core idea to inference-time re-ranking without retraining:
  - Compute centroids of positive and negative preference embeddings
  - Score candidates by proximity to positive centroid minus proximity to negative centroid
"""

import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("melomatch.reranker")


class ContrastiveReranker:
    """
    Re-ranks retrieved candidates using contrastive scoring against
    positive and negative preference centroids.
    """

    def __init__(
        self,
        encoder: SentenceTransformer,
        alpha: float = 0.3,
        normalize: bool = True,
    ):
        """
        Args:
            encoder: Sentence transformer for encoding text.
            alpha: Weight for negative centroid penalty. Higher = stronger avoidance.
                   Seshadri uses β=0.2-0.5 for their contrastive loss weight;
                   we use a similar range for re-ranking.
            normalize: Whether to L2-normalize embeddings (recommended for cosine sim).
        """
        self.encoder = encoder
        self.alpha = alpha
        self.normalize = normalize

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode and optionally normalize texts."""
        if not texts:
            return np.array([])
        embs = self.encoder.encode(texts, normalize_embeddings=self.normalize, batch_size=64)
        return embs

    def _compute_centroid(self, embeddings: np.ndarray) -> Optional[np.ndarray]:
        """Compute mean centroid of embeddings."""
        if embeddings.size == 0:
            return None
        centroid = np.mean(embeddings, axis=0)
        if self.normalize:
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
        return centroid

    def rerank(
        self,
        candidates: list[dict],
        candidate_texts: list[str],
        positive_reasons: list[str],
        negative_reasons: list[str],
    ) -> list[tuple[dict, float]]:
        """
        Re-rank candidates using contrastive scoring.

        Args:
            candidates: List of candidate dicts (KB entries or preference pairs).
            candidate_texts: Text representations of candidates for embedding.
            positive_reasons: User's positive preference reason texts.
            negative_reasons: User's negative preference reason texts.

        Returns:
            List of (candidate, contrastive_score) sorted descending (best first).
        """
        if not candidates:
            return []

        # Encode all texts
        cand_embs = self._encode(candidate_texts)
        pos_embs = self._encode(positive_reasons) if positive_reasons else np.array([])
        neg_embs = self._encode(negative_reasons) if negative_reasons else np.array([])

        pos_centroid = self._compute_centroid(pos_embs)
        neg_centroid = self._compute_centroid(neg_embs)

        scores = []
        for i, cand_emb in enumerate(cand_embs):
            score = 0.0

            # Positive affinity
            if pos_centroid is not None:
                score += float(np.dot(cand_emb, pos_centroid))

            # Negative penalty
            if neg_centroid is not None:
                score -= self.alpha * float(np.dot(cand_emb, neg_centroid))

            scores.append(score)

        # Sort by score descending
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked

    def rerank_retrieved(
        self,
        retrieved: dict,
        user: dict,
    ) -> dict:
        """
        Re-rank the full retrieval output in-place.

        Applies contrastive re-ranking to positive_pairs and kb_entries
        using the user's like/dislike reasons.

        Args:
            retrieved: Output from MeloRetriever.retrieve().
            user: User dict with "likes" and "dislikes".

        Returns:
            Modified retrieved dict with re-ranked entries.
        """
        pos_reasons = [like["reason"] for like in user.get("likes", []) if like.get("reason")]
        neg_reasons = [dis["reason"] for dis in user.get("dislikes", []) if dis.get("reason")]

        if not neg_reasons:
            # No negatives → no contrastive signal, return as-is
            return retrieved

        # Re-rank positive pairs
        if retrieved.get("positive_pairs"):
            pairs = [r for r, _ in retrieved["positive_pairs"]]
            texts = [f"{r['musical']}: {r['reason']}" for r in pairs]
            reranked = self.rerank(pairs, texts, pos_reasons, neg_reasons)
            retrieved["positive_pairs"] = reranked

        # Re-rank KB entries
        if retrieved.get("kb_entries"):
            from src.data.knowledge import build_text_for_embedding
            entries = [r for r, _ in retrieved["kb_entries"]]
            texts = [build_text_for_embedding(e) for e in entries]
            reranked = self.rerank(entries, texts, pos_reasons, neg_reasons)
            retrieved["kb_entries"] = reranked

        return retrieved
