# %% [markdown]
# # MeloMatch — Colab A100 (Zero-Shot: Qwen3-14B via vLLM)
# **CSC5051 NLP Final Project — "Don't Recommend Me Cats"**
#
# This notebook runs the **zero_shot backbone** conditions on Colab A100 (80GB).
# Uses vLLM for ~5× faster inference vs raw HuggingFace.
#
# Runs: 4 signals × 3 methods × 1 backbone (zero_shot) = 12 conditions + baseline + ablations
#
# For fine_tuned backbone, use `melomatch_kaggle.py` on Kaggle T4×2.

# %% [markdown]
# ## 1. Setup

# %%
import os, sys, subprocess

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U',
    'vllm>=0.8.0',
    'sentence-transformers>=3.3.1', 'faiss-cpu>=1.9.0', 'rank-bm25>=0.2.2',
    'jieba>=0.42.1',
    'pyyaml', 'tqdm', 'gradio', 'matplotlib', 'seaborn',
], check=True)
print('✓ Packages installed.')

# %% [markdown]
# ## 2. Configuration

# %%
import gc, hashlib, json, logging, math, pickle, re, time, random, unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# ---------- Hyperparameters ----------
SEED = 42
EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"
ZERO_SHOT_MODEL = "Qwen/Qwen3-14B"

TEMPERATURE = 0.0
MAX_TOKENS = 1024
NUM_RECOMMENDATIONS = 5
DEFAULT_TOP_K = 5
KB_TOP_M = 3
RRF_K = 60
RERANK_ALPHA = 0.3

SIGNALS = ["positive", "dual", "negative", "dual_enhanced"]
METHODS = ["bm25", "dense", "hybrid"]
BACKBONES = ["zero_shot"]  # fine_tuned runs on Kaggle
TOP_K_OPTIONS = [1, 3, 5, 10]
USER_SUBSETS = [25, 50, 100]
HIT_K_VALUES = [1, 3, 5]
AVOIDANCE_K_VALUES = [3, 5]
SOFT_MATCH_TAG_THRESHOLD = 2

NLI_THRESHOLD = 0.6

# Paths — adjust for your Colab Drive mount or dataset upload
DATA_DIR = Path("/content/drive/MyDrive/melomatch/data")  # or /content/melopreference
OUTPUT_DIR = Path("/content/drive/MyDrive/melomatch/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    subprocess.run(['nvidia-smi'], check=False)

# %% [markdown]
# ## 3. Source Code — Utilities & Data

# %%
# ======================== utils ========================
def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def save_jsonl(records: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def save_metrics(metrics: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[metrics] Saved → {path}")

# ======================== splits ========================
def _stable_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2**31)

def hold_out_one_like(user: dict, seed: int = 42) -> tuple[dict, str]:
    likes = user.get("likes", [])
    if not likes:
        return user, ""
    rng = random.Random(seed + _stable_hash(user["user_id"]))
    idx = rng.randrange(len(likes))
    held_out = likes[idx]["musical"]
    modified = {**user, "likes": [l for i, l in enumerate(likes) if i != idx]}
    return modified, held_out

def stratified_user_subset(users: list[dict], n: int, seed: int = 42) -> list[dict]:
    if n >= len(users):
        return users
    rng = random.Random(seed)
    return rng.sample(users, n)

# ======================== knowledge ========================
def load_knowledge_base(path: str) -> list[dict]:
    return load_jsonl(path)

def get_musical_by_name(kb: list[dict], name: str) -> Optional[dict]:
    name_lower = name.lower().strip()
    for entry in kb:
        if entry.get("name", "").lower().strip() == name_lower:
            return entry
    return None

def compute_tag_overlap(entry_a: dict, entry_b: dict) -> int:
    tags_a = set(entry_a.get("genres", [])) | set(entry_a.get("themes", []))
    tags_b = set(entry_b.get("genres", [])) | set(entry_b.get("themes", []))
    return len(tags_a & tags_b)

# %% [markdown]
# ## 4. Source Code — Tokenizer, Indexer, Retriever

# %%
# ======================== tokenizer (BM25) ========================
try:
    import jieba
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False

def tokenize_simple(text: str) -> list[str]:
    if not text:
        return []
    text = text.lower()
    tokens = []
    if HAS_JIEBA:
        for seg in jieba.cut(text):
            seg = seg.strip()
            if seg:
                tokens.append(seg)
    else:
        for tok in re.findall(r'[\u4e00-\u9fff]|[a-z0-9]+', text):
            tokens.append(tok)
    return [t for t in tokens if t.strip()]

# ======================== indexer ========================
class PreferenceIndex:
    def __init__(self, name: str):
        self.name = name
        self.records: list[dict] = []
        self.texts: list[str] = []
        self.bm25: Optional[BM25Okapi] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.embeddings: Optional[np.ndarray] = None

    def build(self, records: list[dict], encoder: SentenceTransformer):
        self.records = records
        self.texts = [f"{r['musical']}: {r['reason']}" for r in records]
        tokenized = [tokenize_simple(t) for t in self.texts]
        if tokenized:
            self.bm25 = BM25Okapi(tokenized)
        embs = encoder.encode(self.texts, normalize_embeddings=True, show_progress_bar=False)
        self.embeddings = embs.astype(np.float32)
        self.faiss_index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.faiss_index.add(self.embeddings)
        print(f"[index] {self.name}: {len(records)} records indexed")

    def search_bm25(self, query: str, top_k: int) -> list[tuple[dict, float]]:
        if not self.bm25:
            return []
        tokens = tokenize_simple(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.records[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    def search_dense(self, query_emb: np.ndarray, top_k: int) -> list[tuple[dict, float]]:
        if self.faiss_index is None:
            return []
        q = query_emb.reshape(1, -1).astype(np.float32)
        scores, indices = self.faiss_index.search(q, min(top_k, len(self.records)))
        return [(self.records[int(i)], float(s)) for s, i in zip(scores[0], indices[0]) if i >= 0]

class KnowledgeBaseIndex(PreferenceIndex):
    def build(self, records: list[dict], encoder: SentenceTransformer):
        self.records = records
        self.texts = [
            f"{r['name']}: {r.get('synopsis', '')} "
            f"[{', '.join(r.get('genres', []))}] [{', '.join(r.get('themes', []))}]"
            for r in records
        ]
        tokenized = [tokenize_simple(t) for t in self.texts]
        if tokenized:
            self.bm25 = BM25Okapi(tokenized)
        embs = encoder.encode(self.texts, normalize_embeddings=True, show_progress_bar=False)
        self.embeddings = embs.astype(np.float32)
        self.faiss_index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.faiss_index.add(self.embeddings)
        print(f"[index] knowledge_base: {len(records)} entries indexed")

# ======================== retriever ========================
class MeloRetriever:
    def __init__(self, pos_index, neg_index, kb_index, encoder):
        self.pos_index = pos_index
        self.neg_index = neg_index
        self.kb_index = kb_index
        self.encoder = encoder

    def _build_query(self, preferences: list[dict]) -> str:
        parts = [f"{p['musical']}: {p['reason']}" for p in preferences]
        return " | ".join(parts)

    def _encode(self, text: str) -> np.ndarray:
        return self.encoder.encode(text, normalize_embeddings=True)

    def _search(self, index, query_text, method, top_k, rrf_k=60):
        if method == "bm25":
            return index.search_bm25(query_text, top_k)
        elif method == "dense":
            q_emb = self._encode(query_text)
            return index.search_dense(q_emb, top_k)
        elif method == "hybrid":
            bm25 = index.search_bm25(query_text, top_k * 2)
            q_emb = self._encode(query_text)
            dense = index.search_dense(q_emb, top_k * 2)
            return self._rrf_fuse(bm25, dense, top_k, rrf_k)
        return []

    def _rrf_fuse(self, list_a, list_b, top_k, rrf_k=60):
        scores = {}
        for rank, (rec, _) in enumerate(list_a):
            key = id(rec)
            scores[key] = scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)
        for rank, (rec, _) in enumerate(list_b):
            key = id(rec)
            scores[key] = scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)
        all_recs = {id(r): r for r, _ in list_a + list_b}
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [(all_recs[k], s) for k, s in ranked]

    def retrieve(self, user, signal, method, top_k=5, kb_top_m=3, exclude_user=True):
        result = {"positive_pairs": [], "negative_pairs": [], "kb_entries": []}
        like_query = self._build_query(user.get("likes", [])) if user.get("likes") else ""
        dislike_query = self._build_query(user.get("dislikes", [])) if user.get("dislikes") else ""

        if signal in ("positive", "dual") and like_query:
            raw = self._search(self.pos_index, like_query, method, top_k)
            if exclude_user:
                raw = [(r, s) for r, s in raw if r.get("user_id") != user.get("user_id")]
            result["positive_pairs"] = raw[:top_k]

        if signal in ("negative", "dual") and dislike_query:
            raw = self._search(self.neg_index, dislike_query, method, top_k)
            if exclude_user:
                raw = [(r, s) for r, s in raw if r.get("user_id") != user.get("user_id")]
            result["negative_pairs"] = raw[:top_k]

        if signal == "negative":
            kb_query = dislike_query
        elif signal == "positive":
            kb_query = like_query
        else:
            kb_query = like_query or dislike_query

        if kb_query:
            kb_raw = self._search(self.kb_index, kb_query, method, kb_top_m)
            result["kb_entries"] = kb_raw[:kb_top_m]

        return result

# %% [markdown]
# ## 5. Source Code — Reranker, NLI Filter, Profiler

# %%
# ======================== contrastive reranker ========================
class ContrastiveReranker:
    def __init__(self, encoder, alpha=0.3):
        self.encoder = encoder
        self.alpha = alpha

    def _encode(self, texts):
        if not texts:
            return np.array([])
        return self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def _compute_centroid(self, embeddings):
        if len(embeddings) == 0:
            return None
        return np.mean(embeddings, axis=0)

    def rerank_retrieved(self, retrieved, user):
        pos_reasons = [l["reason"] for l in user.get("likes", []) if l.get("reason")]
        neg_reasons = [d["reason"] for d in user.get("dislikes", []) if d.get("reason")]
        if retrieved.get("positive_pairs"):
            texts = [r['musical'] + ': ' + r['reason'] for r, _ in retrieved["positive_pairs"]]
            retrieved["positive_pairs"] = self._rerank(retrieved["positive_pairs"], texts, pos_reasons, neg_reasons)
        if retrieved.get("kb_entries"):
            texts = [r.get('name','') + ': ' + r.get('synopsis','') for r, _ in retrieved["kb_entries"]]
            retrieved["kb_entries"] = self._rerank(retrieved["kb_entries"], texts, pos_reasons, neg_reasons)
        return retrieved

    def _rerank(self, candidates, candidate_texts, positive_reasons, negative_reasons):
        if not candidates:
            return []
        cand_embs = self._encode(candidate_texts)
        pos_embs = self._encode(positive_reasons) if positive_reasons else np.array([])
        neg_embs = self._encode(negative_reasons) if negative_reasons else np.array([])
        pos_centroid = self._compute_centroid(pos_embs)
        neg_centroid = self._compute_centroid(neg_embs)
        scores = []
        for i, cand_emb in enumerate(cand_embs):
            score = 0.0
            if pos_centroid is not None:
                score += float(np.dot(cand_emb, pos_centroid))
            if neg_centroid is not None:
                score -= self.alpha * float(np.dot(cand_emb, neg_centroid))
            scores.append((candidates[i], score))
        scores.sort(key=lambda x: -x[1])
        return scores

# ======================== NLI filter ========================
NLI_PROMPT = """You are evaluating whether a musical theatre show conflicts with a user's stated dislikes.

User's dislike reasons:
{dislike_reasons}

Candidate musical:
{candidate_description}

Score from 0.0 (no conflict — safe to recommend) to 1.0 (strong conflict — should avoid).
Respond with JSON: {{"score": <float>, "reasoning": "<brief explanation>"}}"""

class NLIFilter:
    def __init__(self, llm, threshold=0.6, max_retries=2):
        self.llm = llm
        self.threshold = threshold
        self.max_retries = max_retries

    def score_candidate(self, candidate, dislike_reasons):
        if not dislike_reasons:
            return {"score": 0.0, "reasoning": "No dislikes.", "filtered": False}
        genres = ", ".join(candidate.get("genres", []))
        themes = ", ".join(candidate.get("themes", []))
        style = ", ".join(candidate.get("style", []))
        candidate_desc = (
            f"{candidate.get('name', '?')} ({candidate.get('era', '?')}): "
            f"{candidate.get('synopsis', 'N/A')} [Genres: {genres}] [Themes: {themes}] [Style: {style}]"
        )
        reasons_text = "\n".join(f"- {r}" for r in dislike_reasons)
        prompt = NLI_PROMPT.format(dislike_reasons=reasons_text, candidate_description=candidate_desc)

        for attempt in range(self.max_retries):
            try:
                raw = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=256)
                return self._parse(raw)
            except Exception:
                pass
        return {"score": 0.0, "reasoning": "NLI failed.", "filtered": False}

    def filter_candidates(self, candidates, dislike_reasons):
        if not dislike_reasons:
            return [(c, 0.0) for c in candidates]
        results = []
        for c in candidates:
            r = self.score_candidate(c, dislike_reasons)
            if not r["filtered"] and r["score"] < self.threshold:
                results.append((c, r["score"]))
        results.sort(key=lambda x: x[1])
        return results

    def _parse(self, raw):
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                p = json.loads(m.group())
                s = max(0.0, min(1.0, float(p.get("score", 0.0))))
                return {"score": s, "reasoning": str(p.get("reasoning", "")), "filtered": s >= self.threshold}
            except (json.JSONDecodeError, ValueError):
                pass
        m2 = re.search(r"(\d+\.?\d*)", raw)
        if m2:
            s = max(0.0, min(1.0, float(m2.group(1))))
            return {"score": s, "reasoning": raw[:200], "filtered": s >= self.threshold}
        return {"score": 0.0, "reasoning": "Parse failed.", "filtered": False}

# ======================== preference profiler ========================
PROFILE_SCHEMA_KEYS = ["avoid_genres", "avoid_themes", "avoid_styles", "avoid_elements", "tolerance_notes"]

PROFILE_PROMPT = """Extract structured dislike dimensions from these musical theatre dislike reasons:

{dislike_entries}

Respond with JSON matching this schema:
{{
  "avoid_genres": ["..."],
  "avoid_themes": ["..."],
  "avoid_styles": ["..."],
  "avoid_elements": ["..."],
  "tolerance_notes": ["..."]
}}"""

KEYWORD_RULES = {
    "no plot": ("avoid_elements", "no coherent plot"),
    "no story": ("avoid_elements", "no coherent narrative"),
    "jukebox": ("avoid_genres", "jukebox"),
    "boring": ("avoid_styles", "slow-paced"),
    "pretentious": ("avoid_themes", "pretentiousness"),
    "spectacle": ("avoid_styles", "spectacle-over-substance"),
    "cheesy": ("avoid_styles", "overly-sentimental"),
    "depressing": ("avoid_themes", "unrelenting-darkness"),
    "predictable": ("avoid_elements", "predictable plot"),
}

def extract_profile_rule_based(reasons):
    profile = {k: [] for k in PROFILE_SCHEMA_KEYS}
    combined = " ".join(reasons).lower()
    for kw, (dim, val) in KEYWORD_RULES.items():
        if kw in combined and val not in profile[dim]:
            profile[dim].append(val)
    return profile

class PreferenceProfiler:
    def __init__(self, llm=None, use_llm=True, max_retries=2):
        self.llm = llm
        self.use_llm = use_llm and llm is not None
        self.max_retries = max_retries

    def extract_profile(self, user):
        dislikes = user.get("dislikes", [])
        if not dislikes:
            return {k: [] for k in PROFILE_SCHEMA_KEYS}
        reasons = [f"{d['musical']}: {d['reason']}" for d in dislikes if d.get("reason")]
        if not reasons:
            return {k: [] for k in PROFILE_SCHEMA_KEYS}
        if not self.use_llm:
            return extract_profile_rule_based([d.get("reason", "") for d in dislikes])
        return self._extract_with_llm(reasons)

    def _extract_with_llm(self, entries):
        text = "\n".join(f"- {e}" for e in entries)
        prompt = PROFILE_PROMPT.format(dislike_entries=text)
        for _ in range(self.max_retries):
            try:
                raw = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=512)
                return self._parse(raw)
            except Exception:
                pass
        return extract_profile_rule_based(entries)

    def _parse(self, raw):
        cleaned = re.sub(r"```(?:json)?\s*", "", raw)
        cleaned = re.sub(r"```\s*$", "", cleaned)
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                p = json.loads(m.group())
                return {k: [str(v) for v in p.get(k, [])] if isinstance(p.get(k), list) else [] for k in PROFILE_SCHEMA_KEYS}
            except (json.JSONDecodeError, ValueError):
                pass
        return {k: [] for k in PROFILE_SCHEMA_KEYS}

    def profile_to_prompt_section(self, profile):
        lines = []
        for key, label in [("avoid_genres","🚫 Genres"),("avoid_themes","🚫 Themes"),("avoid_styles","🚫 Styles"),("avoid_elements","🚫 Elements"),("tolerance_notes","✅ Acceptable")]:
            vals = profile.get(key, [])
            if vals:
                lines.append(f"**{label}:** {', '.join(vals)}")
        return "\n".join(lines) if lines else ""

# %% [markdown]
# ## 6. Source Code — vLLM Backend, Generator & Metrics

# %%
# ======================== vLLM LLM backend ========================
from vllm import LLM, SamplingParams

class VllmLLM:
    """vLLM offline inference — much faster than raw HuggingFace generate()."""

    def __init__(self, model_name, dtype="auto", tensor_parallel_size=1,
                 max_model_len=4096, gpu_memory_utilization=0.90):
        print(f"Loading vLLM model: {model_name} (tp={tensor_parallel_size}, dtype={dtype})...")
        self.llm = LLM(
            model=model_name,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        self.model_name = model_name
        print(f"  ✓ {model_name} loaded via vLLM")

    def chat(self, messages, temperature=None, max_tokens=None):
        if temperature is None: temperature = 0.0
        if max_tokens is None: max_tokens = MAX_TOKENS
        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9 if temperature > 0 else 1.0,
        )
        outputs = self.llm.chat([messages], sampling_params=params)
        raw = outputs[0].outputs[0].text.strip()
        return re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()

    def batch_chat(self, messages_list, temperature=None, max_tokens=None):
        """Batch inference — feed all prompts at once for maximum throughput."""
        if temperature is None: temperature = 0.0
        if max_tokens is None: max_tokens = MAX_TOKENS
        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9 if temperature > 0 else 1.0,
        )
        outputs = self.llm.chat(messages_list, sampling_params=params)
        results = []
        for out in outputs:
            raw = out.outputs[0].text.strip()
            raw = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()
            results.append(raw)
        return results

# ======================== prompt templates (inlined) ========================
SYSTEM_PROMPT = """You are a musical theatre recommendation expert. Given a user's preferences and retrieved context, recommend exactly {n} musicals they would enjoy. Respond with a JSON array: [{{"musical": "Name", "reason": "explanation"}}, ...]"""

PROMPT_TEMPLATES = {
    "positive": """Based on the user's liked musicals and retrieved positive preferences from similar users, recommend {n} musicals.

**User's liked musicals:**
{user_likes}

**User's disliked musicals:**
{user_dislikes}

**Retrieved positive preferences (from similar users):**
{retrieved_positive}

**Musical knowledge base:**
{kb_context}

Respond with a JSON array of {n} recommendations.""",

    "dual": """Based on the user's preferences and retrieved positive AND negative preferences from similar users, recommend {n} musicals. Avoid recommending musicals similar to what the user dislikes.

**User's liked musicals:**
{user_likes}

**User's disliked musicals:**
{user_dislikes}

**Retrieved positive preferences (users with similar likes):**
{retrieved_positive}

**Retrieved negative preferences (users with similar dislikes):**
{retrieved_negative}

**Musical knowledge base:**
{kb_context}

Respond with a JSON array of {n} recommendations.""",

    "negative": """Based on the user's disliked musicals and retrieved negative preferences, recommend {n} musicals that AVOID the characteristics the user dislikes.

**User's liked musicals:**
{user_likes}

**User's disliked musicals:**
{user_dislikes}

**Retrieved negative preferences (users with similar dislikes):**
{retrieved_negative}

**Musical knowledge base:**
{kb_context}

Respond with a JSON array of {n} recommendations.""",

    "dual_enhanced": """Based on the user's preferences, retrieved context, and structured avoidance profile, recommend {n} musicals.

**User's liked musicals:**
{user_likes}

**User's disliked musicals:**
{user_dislikes}

**Retrieved positive preferences:**
{retrieved_positive}

**Retrieved negative preferences:**
{retrieved_negative}

**Structured avoidance profile:**
{avoidance_profile}

**Musical knowledge base:**
{kb_context}

Respond with a JSON array of {n} recommendations. Pay special attention to the avoidance profile.""",

    "baseline": """Recommend {n} musicals for this user based ONLY on their stated preferences (no external context).

**User's liked musicals:**
{user_likes}

**User's disliked musicals:**
{user_dislikes}

Respond with a JSON array of {n} recommendations.""",
}

def format_preferences(prefs):
    return "\n".join(f"- **{p['musical']}**: {p['reason']}" for p in prefs) if prefs else "(none)"

def format_retrieved(pairs):
    return "\n".join(f"- {r['musical']}: {r['reason']}" for r, _ in pairs) if pairs else "(none)"

def format_kb(entries):
    lines = []
    for e, _ in entries:
        g = ", ".join(e.get("genres", []))
        t = ", ".join(e.get("themes", []))
        lines.append(f"- **{e['name']}** ({e.get('era','?')}): {e.get('synopsis','N/A')} [Genres: {g}] [Themes: {t}]")
    return "\n".join(lines) if lines else "(none)"

# ======================== generator ========================
class MeloGenerator:
    def __init__(self, llm, temperature=0.0, max_tokens=1024, n=5, max_retries=3):
        self.llm = llm
        self.model_name = getattr(llm, 'model_name', 'unknown')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n = n
        self.max_retries = max_retries

    def generate(self, user, retrieved, signal):
        template = PROMPT_TEMPLATES[signal]
        kw = {"n": self.n,
              "user_likes": format_preferences(user.get("likes", [])),
              "user_dislikes": format_preferences(user.get("dislikes", [])),
              "retrieved_positive": format_retrieved(retrieved.get("positive_pairs", [])),
              "retrieved_negative": format_retrieved(retrieved.get("negative_pairs", [])),
              "kb_context": format_kb(retrieved.get("kb_entries", [])),
              "avoidance_profile": retrieved.get("avoidance_profile", "(none)")}
        user_prompt = template.format(**kw)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(n=self.n)},
            {"role": "user", "content": user_prompt},
        ]
        raw = self._call(messages)
        recs = self._parse(raw)
        if recs and recs[0].get("musical") == "PARSE_ERROR" and self.temperature == 0.0:
            raw = self._call(messages, temperature=0.1)
            recs = self._parse(raw)
        valid = [r for r in recs if isinstance(r, dict) and "musical" in r and "reason" in r]
        if not valid:
            valid = [{"musical": "PARSE_ERROR", "reason": raw[:500]}]
        return {"recommendations": valid, "raw_response": raw, "model": self.model_name, "signal": signal}

    def _call(self, messages, temperature=None):
        t = temperature if temperature is not None else self.temperature
        for attempt in range(self.max_retries):
            try:
                return self.llm.chat(messages, temperature=t, max_tokens=self.max_tokens)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

    def _parse(self, raw):
        cleaned = re.sub(r"```(?:json)?\s*", "", raw)
        cleaned = re.sub(r"```\s*$", "", cleaned)
        for pattern in [r"\[.*?\]", r"\[.*\]"]:
            m = re.search(pattern, cleaned, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group())
                    if isinstance(parsed, list):
                        return parsed
                except json.JSONDecodeError:
                    pass
        return [{"musical": "PARSE_ERROR", "reason": raw}]

# ======================== metrics ========================
def _normalize_name(name):
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = name.lower().strip()
    name = re.sub(r"^(the|a|an)\s+", "", name)
    name = re.sub(r"[^\w\s]", "", name)
    return re.sub(r"\s+", " ", name).strip()

def _fuzzy_match(a, b):
    if not a or not b: return False
    if a == b: return True
    a_tok, b_tok = set(a.split()), set(b.split())
    if any(len(t) >= 3 for t in a_tok & b_tok): return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) >= 4 and shorter in longer: return True
    return False

def hit_at_k(recs, held_out, k):
    held_norm = _normalize_name(held_out)
    for r in recs[:k]:
        if _fuzzy_match(_normalize_name(r["musical"]), held_norm):
            return 1
    return 0

def mean_hit_at_k(results, k):
    hits = [hit_at_k(r["recommendations"], r["held_out"], k) for r in results]
    return float(np.mean(hits)) if hits else 0.0

def avoidance_at_k(recs, disliked_musicals, kb_entries, k, tag_threshold=2):
    rec_names = [_normalize_name(r["musical"]) for r in recs[:k]]
    disliked_norm = {_normalize_name(m) for m in disliked_musicals}
    avoid_set = set(disliked_norm)
    for d in disliked_musicals:
        disliked_kb = [e for e in kb_entries if e.get("name","").lower().strip() == d.lower().strip()]
        for kb_e in kb_entries:
            for de in disliked_kb:
                if kb_e["name"] != de.get("name") and compute_tag_overlap(kb_e, de) >= tag_threshold:
                    avoid_set.add(_normalize_name(kb_e["name"]))
    overlap = sum(1 for n in rec_names if n in avoid_set or any(_fuzzy_match(n, a) for a in avoid_set))
    return 1.0 - (overlap / k) if k > 0 else 1.0

def mean_avoidance_at_k(results, kb_entries, k, tag_threshold=2):
    scores = [avoidance_at_k(r["recommendations"], [d["musical"] for d in r["user"]["dislikes"]], kb_entries, k, tag_threshold) for r in results]
    return float(np.mean(scores)) if scores else 1.0

def evaluate_condition(results, kb_entries, hit_k_values=HIT_K_VALUES, avoidance_k_values=AVOIDANCE_K_VALUES, tag_threshold=SOFT_MATCH_TAG_THRESHOLD):
    metrics = {"n_users": len(results)}
    for k in hit_k_values:
        metrics[f"hit@{k}"] = mean_hit_at_k(results, k)
    for k in avoidance_k_values:
        metrics[f"avoidance@{k}"] = mean_avoidance_at_k(results, kb_entries, k, tag_threshold)
    return metrics

# %% [markdown]
# ## 7. Load Model & Data

# %%
# Load embedding model
print("Loading embedding model...")
encoder = SentenceTransformer(EMBEDDING_MODEL)
print(f"  ✓ {EMBEDDING_MODEL} loaded")

# Load zero-shot model via vLLM (bf16 on A100, ~28GB VRAM)
zs_llm = VllmLLM(
    ZERO_SHOT_MODEL,
    dtype="bfloat16",
    tensor_parallel_size=1,
    max_model_len=4096,
    gpu_memory_utilization=0.90,
)

backbone_llms = {"zero_shot": zs_llm}

print("\n✓ All models loaded.")
if torch.cuda.is_available():
    subprocess.run(['nvidia-smi'], check=False)

# %%
# Load data
users = load_jsonl(str(DATA_DIR / "processed" / "users.jsonl"))
kb_entries = load_knowledge_base(str(DATA_DIR / "knowledge_base" / "musicals.jsonl"))
pos_records = load_jsonl(str(DATA_DIR / "processed" / "positive_pairs.jsonl"))
neg_records = load_jsonl(str(DATA_DIR / "processed" / "negative_pairs.jsonl"))

print(f"Users: {len(users)}, KB: {len(kb_entries)}, Pos pairs: {len(pos_records)}, Neg pairs: {len(neg_records)}")

# Build indices
pos_index = PreferenceIndex("positive")
pos_index.build(pos_records, encoder)

neg_index = PreferenceIndex("negative")
neg_index.build(neg_records, encoder)

kb_index = KnowledgeBaseIndex("knowledge_base")
kb_index.build(kb_entries, encoder)

retriever = MeloRetriever(pos_index, neg_index, kb_index, encoder)
reranker = ContrastiveReranker(encoder, alpha=RERANK_ALPHA)
nli_filter = NLIFilter(zs_llm, threshold=NLI_THRESHOLD)
profiler = PreferenceProfiler(llm=zs_llm, use_llm=True)

print("✓ Indices built, retriever + modules ready.")

# %% [markdown]
# ## 8. Run Main Experiment (4 signals × 3 methods × zero_shot)

# %%
all_metrics = {}
all_results = {}

conditions = list(product(SIGNALS, METHODS, BACKBONES))
print(f"Running {len(conditions)} conditions + baseline...")

for signal, method, backbone in tqdm(conditions, desc="Conditions"):
    cond_name = f"{signal}_{method}_{backbone}"
    llm = backbone_llms[backbone]
    generator = MeloGenerator(llm, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, n=NUM_RECOMMENDATIONS)

    cond_results = []
    for user in tqdm(users, desc=cond_name, leave=False):
        modified_user, held_out = hold_out_one_like(user, SEED)
        if not held_out:
            continue
        try:
            retrieval_signal = "dual" if signal == "dual_enhanced" else signal
            retrieved = retriever.retrieve(modified_user, retrieval_signal, method, DEFAULT_TOP_K, KB_TOP_M)

            if signal in ("dual", "dual_enhanced", "negative"):
                retrieved = reranker.rerank_retrieved(retrieved, modified_user)

            if signal == "dual_enhanced":
                dislike_reasons = [d["reason"] for d in modified_user.get("dislikes", []) if d.get("reason")]
                if dislike_reasons and retrieved.get("kb_entries"):
                    kb_cands = [e for e, _ in retrieved["kb_entries"]]
                    filtered = nli_filter.filter_candidates(kb_cands, dislike_reasons)
                    retrieved["kb_entries"] = filtered
                profile = profiler.extract_profile(modified_user)
                retrieved["avoidance_profile"] = profiler.profile_to_prompt_section(profile)

            output = generator.generate(modified_user, retrieved, signal)
            cond_results.append({
                "user": user, "held_out": held_out,
                "recommendations": output["recommendations"],
                "raw_response": output["raw_response"],
                "signal": signal, "method": method, "backbone": backbone,
            })
        except Exception as e:
            print(f"  ✗ {user['user_id']}: {e}")
            continue

    metrics = evaluate_condition(cond_results, kb_entries)
    all_metrics[cond_name] = metrics
    all_results[cond_name] = cond_results
    print(f"  {cond_name}: Hit@5={metrics.get('hit@5',0):.3f} Avoid@5={metrics.get('avoidance@5',0):.3f}")

# Pure LLM baseline
for backbone in BACKBONES:
    llm = backbone_llms[backbone]
    generator = MeloGenerator(llm, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, n=NUM_RECOMMENDATIONS)
    bl_results = []
    for user in tqdm(users, desc=f"baseline_{backbone}", leave=False):
        modified_user, held_out = hold_out_one_like(user, SEED)
        if not held_out:
            continue
        try:
            empty = {"positive_pairs": [], "negative_pairs": [], "kb_entries": []}
            output = generator.generate(modified_user, empty, "baseline")
            bl_results.append({"user": user, "held_out": held_out, "recommendations": output["recommendations"],
                               "raw_response": output["raw_response"], "signal": "baseline", "method": "none", "backbone": backbone})
        except Exception as e:
            print(f"  ✗ baseline {user['user_id']}: {e}")
    metrics = evaluate_condition(bl_results, kb_entries)
    cond_name = f"baseline_none_{backbone}"
    all_metrics[cond_name] = metrics
    print(f"  {cond_name}: Hit@5={metrics.get('hit@5',0):.3f} Avoid@5={metrics.get('avoidance@5',0):.3f}")

save_metrics(all_metrics, str(OUTPUT_DIR / "metrics_zero_shot.json"))
print(f"\n✓ Zero-shot experiment complete. {len(all_metrics)} conditions evaluated.")

# %% [markdown]
# ## 9. Ablations

# %%
# ======================== Retrieval Depth Ablation ========================
print("Running retrieval depth ablation (k)...")
k_ablation_metrics = {}
abl_gen = MeloGenerator(zs_llm, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, n=NUM_RECOMMENDATIONS)

for k_val in TOP_K_OPTIONS:
    for signal in SIGNALS:
        cond_name = f"k_abl_{signal}_hybrid_zero_shot_k{k_val}"
        cond_results = []
        for user in tqdm(users, desc=cond_name, leave=False):
            modified_user, held_out = hold_out_one_like(user, SEED)
            if not held_out:
                continue
            try:
                rsig = "dual" if signal == "dual_enhanced" else signal
                retrieved = retriever.retrieve(modified_user, rsig, "hybrid", k_val, KB_TOP_M)
                output = abl_gen.generate(modified_user, retrieved, signal)
                cond_results.append({"user": user, "held_out": held_out, "recommendations": output["recommendations"],
                                     "raw_response": output["raw_response"], "signal": signal, "method": "hybrid", "backbone": "zero_shot"})
            except Exception as e:
                continue
        metrics = evaluate_condition(cond_results, kb_entries)
        k_ablation_metrics[cond_name] = metrics
save_metrics(k_ablation_metrics, str(OUTPUT_DIR / "k_ablation_metrics.json"))
print(f"✓ k-ablation: {len(k_ablation_metrics)} conditions")

# ======================== Data Scaling Ablation ========================
print("\nRunning data scaling ablation...")
scale_metrics = {}
for n_users in USER_SUBSETS:
    subset = stratified_user_subset(users, n_users, SEED)
    cond_name = f"scale_dual_hybrid_zero_shot_n{n_users}"
    cond_results = []
    for user in tqdm(subset, desc=cond_name, leave=False):
        modified_user, held_out = hold_out_one_like(user, SEED)
        if not held_out:
            continue
        try:
            retrieved = retriever.retrieve(modified_user, "dual", "hybrid", DEFAULT_TOP_K, KB_TOP_M)
            output = abl_gen.generate(modified_user, retrieved, "dual")
            cond_results.append({"user": user, "held_out": held_out, "recommendations": output["recommendations"],
                                 "raw_response": output["raw_response"], "signal": "dual", "method": "hybrid", "backbone": "zero_shot"})
        except Exception as e:
            continue
    metrics = evaluate_condition(cond_results, kb_entries)
    scale_metrics[cond_name] = metrics
save_metrics(scale_metrics, str(OUTPUT_DIR / "scale_ablation_metrics.json"))
print(f"✓ Scale ablation: {len(scale_metrics)} conditions")

# %% [markdown]
# ## 10. Visualization & Summary

# %%
import matplotlib.pyplot as plt

def plot_results(all_metrics):
    main = {k: v for k, v in all_metrics.items()
            if not k.startswith(("k_abl", "scale_", "baseline"))}

    signal_scores = defaultdict(lambda: defaultdict(list))
    for cond, m in main.items():
        for sig in SIGNALS:
            if cond.startswith(sig + "_"):
                for metric_key in ["hit@5", "avoidance@5"]:
                    if metric_key in m:
                        signal_scores[metric_key][sig].append(m[metric_key])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"positive": "#4CAF50", "dual": "#2196F3", "negative": "#FF5722", "dual_enhanced": "#9C27B0"}

    for ax, metric in zip(axes, ["hit@5", "avoidance@5"]):
        sigs = [s for s in SIGNALS if s in signal_scores[metric]]
        means = [np.mean(signal_scores[metric][s]) for s in sigs]
        bars = ax.bar(sigs, means, color=[colors.get(s, "#999") for s in sigs])
        ax.set_title(metric.upper(), fontsize=14)
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{val:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "signal_comparison_zero_shot.png"), dpi=150)
    plt.show()

plot_results(all_metrics)

print("\n" + "=" * 80)
print("RESULTS SUMMARY (zero_shot backbone)")
print("=" * 80)
for cond in sorted(all_metrics.keys()):
    m = all_metrics[cond]
    parts = [f"{k}={v:.3f}" for k, v in m.items() if k != "n_users"]
    print(f"  {cond}: {', '.join(parts)}")
