"""
indexer.py — Build BM25, Dense (FAISS), and Hybrid indices over preference pairs and KB.

Indices built:
  - I+ : positive preference-reason pairs
  - I- : negative preference-reason pairs
  - K  : musical knowledge base entries
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.utils import load_config, load_jsonl


def tokenize_simple(text: str) -> list[str]:
    """Tokenizer for BM25 that handles both English and Chinese.

    Uses jieba for Chinese segmentation when available; falls back to
    character-level splitting for CJK ranges so BM25 can still match
    individual characters rather than entire unsegmented strings.
    """
    import re

    text = text.lower()

    # Split into CJK and non-CJK segments
    # CJK Unified Ideographs: U+4E00-U+9FFF
    segments = re.findall(r'[\u4e00-\u9fff]+|[a-z0-9]+', text)

    tokens: list[str] = []
    try:
        import jieba
        for seg in segments:
            if re.match(r'[\u4e00-\u9fff]', seg):
                tokens.extend(jieba.lcut(seg))
            else:
                tokens.append(seg)
    except ImportError:
        # Fallback: character-level for CJK, word-level for Latin
        for seg in segments:
            if re.match(r'[\u4e00-\u9fff]', seg):
                tokens.extend(list(seg))  # character-level > one giant token
            else:
                tokens.append(seg)

    return [t for t in tokens if t.strip()]


class PreferenceIndex:
    """
    Manages BM25 + Dense indices for a set of preference-reason pairs.
    """

    def __init__(self, name: str):
        self.name = name
        self.records: list[dict] = []
        self.texts: list[str] = []

        # BM25
        self.bm25: Optional[BM25Okapi] = None

        # Dense (FAISS)
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.embeddings: Optional[np.ndarray] = None

    def build(
        self,
        records: list[dict],
        encoder: SentenceTransformer,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ):
        """Build both BM25 and dense indices from records."""
        self.records = records
        self.texts = [self._build_text(r) for r in records]

        # BM25
        tokenized = [tokenize_simple(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized, k1=bm25_k1, b=bm25_b)

        # Dense
        print(f"  [{self.name}] Encoding {len(self.texts)} texts...")
        self.embeddings = encoder.encode(
            self.texts,
            show_progress_bar=True,
            normalize_embeddings=True,
            batch_size=64,
        )
        dim = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)  # inner product = cosine (normalized)
        self.faiss_index.add(self.embeddings.astype(np.float32))

        print(f"  [{self.name}] Built index: {len(records)} records, dim={dim}")

    def search_bm25(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        """BM25 retrieval."""
        tokens = tokenize_simple(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.records[i], float(scores[i])) for i in top_indices]

    def search_dense(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> list[tuple[dict, float]]:
        """Dense retrieval via FAISS."""
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        return [
            (self.records[idx], float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx >= 0
        ]

    def search_hybrid(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> list[tuple[dict, float]]:
        """
        Hybrid retrieval: Reciprocal Rank Fusion of BM25 + Dense.
        Uses id(record) for O(1) lookup instead of .index() which is O(n).
        """
        pool_size = top_k * 3
        bm25_results = self.search_bm25(query, pool_size)
        dense_results = self.search_dense(query_embedding, pool_size)

        # Build identity map for O(1) lookup
        record_id_to_idx = {id(r): i for i, r in enumerate(self.records)}

        # RRF scoring
        rrf_scores: dict[int, float] = {}

        for rank, (record, _) in enumerate(bm25_results):
            idx = record_id_to_idx.get(id(record), -1)
            if idx >= 0:
                rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rrf_k + rank + 1)

        for rank, (record, _) in enumerate(dense_results):
            idx = record_id_to_idx.get(id(record), -1)
            if idx >= 0:
                rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rrf_k + rank + 1)

        sorted_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        return [(self.records[i], rrf_scores[i]) for i in sorted_indices]

    def save(self, dir_path: str):
        """Persist index to disk."""
        out = Path(dir_path) / self.name
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "records.json", "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)
        with open(out / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)
        np.save(str(out / "embeddings.npy"), self.embeddings)
        faiss.write_index(self.faiss_index, str(out / "faiss.index"))
        print(f"  [{self.name}] Saved to {out}")

    def _build_text(self, record: dict) -> str:
        """Build text representation from a record. Override in subclasses."""
        return f"{record['musical']}: {record['reason']}"

    def load(self, dir_path: str, encoder: Optional[SentenceTransformer] = None):
        """Load index from disk."""
        src = Path(dir_path) / self.name
        with open(src / "records.json", "r", encoding="utf-8") as f:
            self.records = json.load(f)
        self.texts = [self._build_text(r) for r in self.records]
        with open(src / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        self.embeddings = np.load(str(src / "embeddings.npy"))
        self.faiss_index = faiss.read_index(str(src / "faiss.index"))
        print(f"  [{self.name}] Loaded: {len(self.records)} records")


class KnowledgeBaseIndex(PreferenceIndex):
    """Index over musical KB entries (same retrieval mechanics, different text format)."""

    def _build_text(self, record: dict) -> str:
        """Override: KB entries use name/synopsis, not musical/reason."""
        from src.data.knowledge import build_text_for_embedding
        return build_text_for_embedding(record)

    def build_from_kb(
        self,
        entries: list[dict],
        encoder: SentenceTransformer,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ):
        """Build index from KB entries. Delegates to parent build() via _build_text override."""
        # Parent build() will call self._build_text() which is overridden
        # to use build_text_for_embedding for KB entries.
        # We just need to give records a 'musical'/'reason' duck-type — but
        # since _build_text is overridden, parent.build() works directly.
        self.records = entries
        self.texts = [self._build_text(e) for e in entries]

        tokenized = [tokenize_simple(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized, k1=bm25_k1, b=bm25_b)

        print(f"  [{self.name}] Encoding {len(self.texts)} KB entries...")
        self.embeddings = encoder.encode(
            self.texts, show_progress_bar=True, normalize_embeddings=True, batch_size=64
        )
        dim = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.embeddings.astype(np.float32))
        print(f"  [{self.name}] Built KB index: {len(entries)} entries, dim={dim}")


def build_all_indices(config_path: str = "configs/config.yaml"):
    """Build all indices from processed data."""
    config = load_config(config_path)
    processed = config["paths"]["processed_data"]
    kb_path = Path(config["paths"]["knowledge_base"]) / "musicals.jsonl"
    index_dir = config["paths"].get("index_dir", "data/indices")
    embed_model = config["retrieval"]["embedding_model"]

    print(f"[indexer] Loading encoder: {embed_model}")
    encoder = SentenceTransformer(embed_model)

    # Positive pairs
    pos_records = load_jsonl(f"{processed}/positive_pairs.jsonl")
    pos_index = PreferenceIndex("positive")
    pos_index.build(pos_records, encoder)
    pos_index.save(index_dir)

    # Negative pairs
    neg_records = load_jsonl(f"{processed}/negative_pairs.jsonl")
    neg_index = PreferenceIndex("negative")
    neg_index.build(neg_records, encoder)
    neg_index.save(index_dir)

    # Knowledge base
    if kb_path.exists():
        kb_entries = load_jsonl(str(kb_path))
        kb_index = KnowledgeBaseIndex("knowledge_base")
        kb_index.build_from_kb(kb_entries, encoder)
        kb_index.save(index_dir)
    else:
        print(f"[indexer] KB not found at {kb_path}, skipping.")

    print("[indexer] All indices built.")


if __name__ == "__main__":
    build_all_indices()
