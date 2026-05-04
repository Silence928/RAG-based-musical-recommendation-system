"""
generator.py — LLM-based recommendation generation with retrieved context.

Supports two backbones via unified LLM backend (local transformers or API):
  1. Qwen3-30B-A3B (zero-shot MoE)
  2. Qwen3-8B fine-tuned (QLoRA)

Prompt templates loaded from configs/prompts/ (externalized for easy iteration).
"""

import json
import re
import time
import logging
from pathlib import Path
from typing import Optional

from src.utils import load_config

logger = logging.getLogger("melomatch.generator")

JSON_REPAIR_PROMPT = """
你上一条回复不是合法JSON。
请严格按以下格式重写，且只输出JSON数组，不要输出任何解释、Markdown、<think>或前后缀文本：
[
  {
    "musical": "剧目名称",
    "reason": "推荐理由",
    "evidence_quotes": ["来自提供上下文的短句证据1", "短句证据2"]
  }
]
"""


# ======================== Prompt Loading ========================

PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "prompts"


def load_prompt(name: str) -> str:
    """Load a prompt template from configs/prompts/{name}.txt"""
    path = PROMPT_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_prompt_optional(name: str) -> str:
    """Load optional prompt template; return empty string if missing."""
    path = PROMPT_DIR / f"{name}.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def get_system_prompt(n: int) -> str:
    return load_prompt("system").format(n=n)


def get_user_prompt(signal: str) -> str:
    """Load user prompt template for a given signal condition."""
    return load_prompt(signal)


def get_few_shot_examples(signal: str) -> str:
    """
    Optional few-shot examples file:
      configs/prompts/few_shot_{signal}.txt
    """
    return load_prompt_optional(f"few_shot_{signal}")


# ======================== Formatting Helpers ========================

def format_preferences(prefs: list[dict]) -> str:
    lines = []
    for p in prefs:
        lines.append(f"- **{p['musical']}**: {p['reason']}")
    return "\n".join(lines) if lines else "(none provided)"


def format_retrieved(pairs: list[tuple[dict, float]]) -> str:
    lines = []
    for record, score in pairs:
        lines.append(f"- {record['musical']}: {record['reason']}")
    return "\n".join(lines) if lines else "(none)"


def format_kb(entries: list[tuple[dict, float]]) -> str:
    lines = []
    for entry, score in entries:
        lines.append(
            f"- **{entry.get('name', 'Unknown')}** "
            f"[Language: {entry.get('language_type', 'unknown')}] "
            f"[Format: {entry.get('format', 'unknown')}] "
            f"[Tradition: {entry.get('tradition', 'unknown')}]"
        )
    return "\n".join(lines) if lines else "(no KB entries retrieved)"

def format_user_profile(meta: dict) -> str:
    if not meta:
        return "(no profile metadata)"

    valued = meta.get("valued_elements", [])
    valued_text = ", ".join([f"{v['element']}({v['weight']})" for v in valued]) if valued else "none"

    lines = [
        f"- Experience: {meta.get('experience', 'unknown')}",
        f"- Tradition preference: {', '.join(meta.get('tradition_pref', [])) or 'none'}",
        f"- Avoided styles: {', '.join(meta.get('avoided_styles', [])) or 'none'}",
        f"- Valued elements (weighted): {valued_text}",
    ]
    return "\n".join(lines)


def format_semantic_dimensions(meta: dict) -> str:
    """
    Optional deconstructed semantic intent from user input.
    """
    dims = (meta or {}).get("semantic_dimensions", {}) if isinstance(meta, dict) else {}
    if not isinstance(dims, dict) or not dims:
        return "(none)"
    parts = []
    for key, label in (
        ("semantic", "语义维度"),
        ("emotional", "情感维度"),
        ("stagecraft", "舞美维度"),
        ("songs", "歌曲维度"),
    ):
        vals = dims.get(key, [])
        if isinstance(vals, list) and vals:
            parts.append(f"- {label}: {', '.join([str(v) for v in vals if str(v).strip()])}")
    return "\n".join(parts) if parts else "(none)"

# ======================== Generator ========================

class MeloGenerator:
    """Generate recommendations using a unified LLM backend (local or API)."""

    def __init__(
        self,
        llm,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        num_recommendations: int = 5,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        allowed_musical_names: Optional[list[str]] = None,
        exclude_user_mentioned: bool = True,
    ):
        self.llm = llm
        self.model_name = getattr(llm, 'model_name', 'unknown')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n = num_recommendations
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.allowed_musical_names = allowed_musical_names or []
        self.exclude_user_mentioned = exclude_user_mentioned
        self._allowed_name_map = {
            self._normalize_name(name): name for name in self.allowed_musical_names if self._normalize_name(name)
        }

    def generate(
        self,
        user: dict,
        retrieved: dict,
        signal: str,
    ) -> dict:
        """
        Generate recommendations for a user.

        Args:
            user: User dict with "likes" and "dislikes".
            retrieved: Output from MeloRetriever.retrieve().
            signal: "positive", "dual", "negative", or "baseline".

        Returns:
            {
                "recommendations": [{"musical": str, "reason": str}, ...],
                "raw_response": str,
                "model": str,
                "signal": str,
            }
        """
        template = get_user_prompt(signal)

        kwargs = {"n": self.n}
        kwargs["user_likes"] = format_preferences(user.get("likes", []))
        kwargs["user_dislikes"] = format_preferences(user.get("dislikes", []))
        kwargs["user_profile_summary"] = format_user_profile(user.get("meta", {}))
        kwargs["semantic_dimensions"] = format_semantic_dimensions(user.get("meta", {}))
        kwargs["retrieved_positive"] = format_retrieved(retrieved.get("positive_pairs", []))
        kwargs["retrieved_negative"] = format_retrieved(retrieved.get("negative_pairs", []))
        kwargs["kb_context"] = format_kb(retrieved.get("kb_entries", []))
        kwargs["avoidance_profile"] = retrieved.get("avoidance_profile", "(none)")
        kwargs["domain_lexicon_context"] = retrieved.get("domain_lexicon_context", "(未命中术语)")
        kwargs["few_shot_examples"] = get_few_shot_examples(signal)

        user_prompt = template.format(**kwargs)
        if kwargs["domain_lexicon_context"] and kwargs["domain_lexicon_context"] != "(未命中术语)":
            # Keep user original text unchanged; add glossary as side context.
            user_prompt += (
                "\n\n## 领域术语参考（仅辅助理解，不要机械复述）\n"
                f"{kwargs['domain_lexicon_context']}\n"
            )
        if kwargs["few_shot_examples"]:
            user_prompt += (
                "\n\n## 参考示例（few-shot，仅学习风格与约束，不要照抄剧名）\n"
                f"{kwargs['few_shot_examples']}\n"
            )

        messages = [
            {"role": "system", "content": get_system_prompt(self.n)},
            {"role": "user", "content": user_prompt},
        ]

        raw = self._call_with_retry(messages)
        recommendations = self._parse_recommendations(raw)

        # First-pass parse failed or produced placeholder entry -> force JSON repair.
        require_evidence = signal == "dual_enhanced"
        if not self._is_valid_recommendation_list(recommendations, require_evidence=require_evidence):
            logger.warning("First-pass JSON invalid, running JSON repair pass...")
            repaired_raw = self._force_json_repair(raw)
            repaired_recs = self._parse_recommendations(repaired_raw)
            if self._is_valid_recommendation_list(repaired_recs, require_evidence=require_evidence):
                raw = repaired_raw
                recommendations = repaired_recs

        # If parsing failed and temperature is 0, retry with slight temperature
        # bump to get a structurally different response (identical input at
        # temp 0 produces identical output, so retrying is futile without this).
        if (recommendations and recommendations[0].get("musical") == "PARSE_ERROR"
                and self.temperature == 0.0):
            logger.warning("JSON parse failed at temp 0.0 — retrying with temp 0.1")
            raw = self._call_with_retry(messages, temperature=0.1)
            recommendations = self._parse_recommendations(raw)

        # Validate: each recommendation must have 'musical' and 'reason'
        valid_recs = []
        for rec in recommendations:
            if (
                isinstance(rec, dict)
                and "musical" in rec
                and "reason" in rec
                and rec.get("musical") != "PARSE_ERROR"
            ):
                canonical = self._canonicalize_to_allowed_name(str(rec.get("musical", "")))
                if self._allowed_name_map and canonical is None:
                    logger.warning(f"Recommendation not in KB, skipped: {rec.get('musical')}")
                    continue
                if canonical is not None:
                    rec["musical"] = canonical
                valid_recs.append(rec)
            else:
                logger.warning(f"Invalid recommendation entry skipped: {rec}")

        # Exclude musicals explicitly mentioned by user input/preferences.
        if self.exclude_user_mentioned and valid_recs:
            mentioned = self._build_user_mentioned_set(user)
            before = len(valid_recs)
            valid_recs = self._exclude_user_mentioned(valid_recs, mentioned)
            if len(valid_recs) < before:
                logger.info(f"Filtered {before - len(valid_recs)} user-mentioned recommendations.")

        if not valid_recs:
            # Fallback: use retrieved KB candidates to keep outputs in-KB.
            kb_fallback = []
            mentioned = self._build_user_mentioned_set(user)
            for entry, _score in retrieved.get("kb_entries", []):
                name = entry.get("name", "")
                canonical = self._canonicalize_to_allowed_name(name)
                if not canonical:
                    continue
                if self.exclude_user_mentioned and self._normalize_name(canonical) in mentioned:
                    continue
                evidence_quotes = self._collect_candidate_evidence(canonical, retrieved, max_quotes=2)
                kb_fallback.append(
                    {
                        "musical": canonical,
                        "reason": self._build_kb_fallback_reason(canonical, entry, user),
                        "evidence_quotes": evidence_quotes,
                    }
                )
                if len(kb_fallback) >= self.n:
                    break

            if kb_fallback:
                valid_recs = kb_fallback
            else:
                logger.error(f"No valid recommendations parsed. Raw: {raw[:200]}")
                valid_recs = [
                    {
                        "musical": "推荐结果生成失败",
                        "reason": "模型输出未能解析为合法JSON，请重试。",
                    }
                ]

        return {
            "recommendations": valid_recs,
            "raw_response": raw,
            "model": self.model_name,
            "signal": signal,
        }

    def _normalize_name(self, name: str) -> str:
        s = (name or "").strip().lower()
        s = re.sub(r"[《》\"'`]", "", s)
        s = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _build_user_mentioned_set(self, user: dict) -> set[str]:
        ignored = {"用户正向偏好", "用户回避偏好", "偏好描述"}
        out: set[str] = set()
        for item in user.get("likes", []) + user.get("dislikes", []):
            raw = str(item.get("musical", "")).strip()
            if not raw or raw in ignored:
                continue
            n = self._normalize_name(raw)
            if n:
                out.add(n)
        return out

    def _exclude_user_mentioned(self, recs: list[dict], mentioned: set[str]) -> list[dict]:
        if not mentioned:
            return recs
        out = []
        for rec in recs:
            n = self._normalize_name(str(rec.get("musical", "")))
            if n and n in mentioned:
                continue
            out.append(rec)
        return out

    def _collect_candidate_evidence(self, candidate_name: str, retrieved: dict, max_quotes: int = 2) -> list[str]:
        key = self._normalize_name(candidate_name)
        quotes: list[str] = []
        for bucket in ("positive_pairs", "negative_pairs"):
            for record, _score in retrieved.get(bucket, []) or []:
                m = self._normalize_name(str(record.get("musical", "")))
                if m != key:
                    continue
                reason = str(record.get("reason", "")).strip()
                if reason:
                    quotes.append(reason)
                if len(quotes) >= max_quotes:
                    return quotes
        return quotes

    def _build_kb_fallback_reason(self, candidate_name: str, kb_entry: dict, user: dict) -> str:
        parts: list[str] = []
        lang = kb_entry.get("language_type")
        fmt = kb_entry.get("format")
        tradition = kb_entry.get("tradition")
        if lang:
            parts.append(f"语言类型为{lang}")
        if fmt:
            parts.append(f"形式为{fmt}")
        if tradition:
            parts.append(f"风格倾向{tradition}")

        likes = [x.get("musical", "") for x in user.get("likes", []) if x.get("musical") and x.get("musical") != "偏好描述"]
        dislikes = [x.get("musical", "") for x in user.get("dislikes", []) if x.get("musical")]
        if likes:
            parts.append(f"与您提到喜欢的作品（如{likes[0]}）在偏好方向上更接近")
        if dislikes:
            parts.append("并尽量避开您明确不喜欢的叙事取向")

        if not parts:
            return f"{candidate_name}在检索候选中的综合匹配度较高。"
        return f"{candidate_name}推荐给你的原因：{'；'.join(parts)}。"

    def _canonicalize_to_allowed_name(self, name: str) -> Optional[str]:
        if not self._allowed_name_map:
            return name

        n = self._normalize_name(name)
        if n in self._allowed_name_map:
            return self._allowed_name_map[n]

        # Fuzzy fallback for minor punctuation/spacing differences.
        for k, canonical in self._allowed_name_map.items():
            if n and (n in k or k in n):
                shorter = n if len(n) <= len(k) else k
                if len(shorter) >= 4:
                    return canonical
        return None

    def _call_with_retry(self, messages: list[dict], temperature: float = None) -> str:
        """Call LLM API with exponential backoff retry on transient failures."""
        if temperature is None:
            temperature = self.temperature
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return self.llm.chat(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                )
            except Exception as e:
                last_error = e
                wait = self.retry_delay * (2 ** attempt)
                logger.warning(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
        raise RuntimeError(f"API call failed after {self.max_retries} retries: {last_error}")

    def _is_valid_recommendation_list(self, recs: list[dict], require_evidence: bool = False) -> bool:
        """Strict validation for the expected recommendation JSON schema."""
        if not isinstance(recs, list) or not recs:
            return False
        for r in recs:
            if not isinstance(r, dict):
                return False
            if "musical" not in r or "reason" not in r:
                return False
            if not str(r["musical"]).strip() or not str(r["reason"]).strip():
                return False
            if str(r["musical"]).strip() == "PARSE_ERROR":
                return False
            if require_evidence:
                if "evidence_quotes" not in r:
                    return False
                if not isinstance(r["evidence_quotes"], list):
                    return False
                for q in r["evidence_quotes"]:
                    if not isinstance(q, str) or not q.strip():
                        return False
        return True

    def _force_json_repair(self, raw: str) -> str:
        """
        Second pass: ask the model to rewrite its own output as strict JSON.
        """
        messages = [
            {"role": "system", "content": "你是一个严格的JSON格式化器。"},
            {"role": "user", "content": f"{JSON_REPAIR_PROMPT}\n\n原始输出如下：\n{raw}"},
        ]
        return self._call_with_retry(messages, temperature=0.0)

    def _parse_recommendations(self, raw: str) -> list[dict]:
        """Extract JSON array from LLM response (robust to markdown fences)."""
        # Strip markdown code fences
        cleaned = re.sub(r"```(?:json)?\s*", "", raw)
        cleaned = re.sub(r"```\s*$", "", cleaned)

        # Find JSON array (lazy match to avoid capturing trailing brackets)
        json_match = re.search(r"\[.*?\]", cleaned, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

        # Retry with greedy match (handles nested arrays in JSON)
        json_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

        # Fallback: return raw as single entry
        return [{"musical": "PARSE_ERROR", "reason": raw}]
