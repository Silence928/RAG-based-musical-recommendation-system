"""
preference_profiler.py — Structured negative preference profile extraction.

Extracts structured "discomfort dimensions" from free-text dislike reasons,
then uses these as structured retrieval/filtering queries.

Inspired by: "DiscomfortFilter" concept — structuring vague dislike reasons
into concrete avoidance dimensions (genre, theme, style, narrative structure).

This bridges the gap between free-text reasons like "I found it boring and
pretentious" and structured filter queries like:
  {avoid_themes: ["pretentiousness"], avoid_style: ["slow-paced", "avant-garde"]}

Two modes:
  1. LLM-based extraction (higher quality, requires API call)
  2. Rule-based extraction (fast, no API, lower quality — fallback)
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger("melomatch.profiler")

PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "prompts"
JSON_REPAIR_PROMPT = """
请把下面这段输出重写为唯一一个JSON对象，不要输出任何额外文本：
{"avoid_genres": [...], "avoid_themes": [...], "avoid_styles": [...], "avoid_elements": [...], "tolerance_notes": [...]}
"""


def load_profile_prompt() -> str:
    """Load profile extraction prompt template."""
    path = PROMPT_DIR / "profile_extract.txt"
    if not path.exists():
        raise FileNotFoundError(f"Profile extraction prompt not found: {path}")
    return path.read_text(encoding="utf-8").strip()


# ======================== Structured Profile Schema ========================

PROFILE_SCHEMA = {
    "avoid_genres": [],       # e.g. ["jukebox", "revue", "dance-heavy"]
    "avoid_themes": [],       # e.g. ["pretentiousness", "shallow romance"]
    "avoid_styles": [],       # e.g. ["spectacle-over-substance", "non-linear"]
    "avoid_elements": [],     # e.g. ["no coherent plot", "excessive choreography"]
    "tolerance_notes": [],    # e.g. ["okay with dark themes if well-written"]
}


# ======================== Rule-Based Extraction (Fallback) ========================

# Keyword → dimension mapping for common dislike patterns.
# These are rough heuristics; LLM extraction is preferred.
KEYWORD_RULES = {
    # Genre signals
    "no plot": ("avoid_elements", "no coherent plot"),
    "no story": ("avoid_elements", "no coherent narrative"),
    "jukebox": ("avoid_genres", "jukebox"),
    "revue": ("avoid_genres", "revue"),
    # Style signals
    "boring": ("avoid_styles", "slow-paced"),
    "pretentious": ("avoid_themes", "pretentiousness"),
    "spectacle": ("avoid_styles", "spectacle-over-substance"),
    "too long": ("avoid_styles", "overly-long"),
    "confusing": ("avoid_styles", "non-linear-narrative"),
    "childish": ("avoid_styles", "juvenile"),
    "cheesy": ("avoid_styles", "overly-sentimental"),
    "repetitive": ("avoid_elements", "repetitive songs"),
    "predictable": ("avoid_elements", "predictable plot"),
    # Theme signals
    "depressing": ("avoid_themes", "unrelenting-darkness"),
    "dark": ("avoid_themes", "excessive-darkness"),
    "violent": ("avoid_themes", "graphic-violence"),
    "romance": ("avoid_themes", "shallow-romance"),
    "love triangle": ("avoid_themes", "love-triangle"),
}


def extract_profile_rule_based(dislike_reasons: list[str]) -> dict:
    """
    Fallback: extract structured profile from dislike reasons using keyword rules.

    Args:
        dislike_reasons: List of free-text dislike reasons.

    Returns:
        Structured profile dict following PROFILE_SCHEMA.
    """
    profile = {k: [] for k in PROFILE_SCHEMA}

    combined = " ".join(dislike_reasons).lower()

    for keyword, (dimension, value) in KEYWORD_RULES.items():
        if keyword in combined and value not in profile[dimension]:
            profile[dimension].append(value)

    return profile


# ======================== LLM-Based Extraction ========================

class PreferenceProfiler:
    """
    Extracts structured discomfort profiles from free-text dislike reasons
    using an LLM.
    """

    def __init__(
        self,
        llm=None,
        temperature: float = 0.0,
        max_retries: int = 2,
        use_llm: bool = True,
    ):
        """
        Args:
            llm: LLM backend (LocalLLM or APILLM) with .chat() method.
            temperature: 0.0 for deterministic.
            max_retries: Retry on failure.
            use_llm: If False, falls back to rule-based extraction.
        """
        self.use_llm = use_llm
        self.temperature = temperature
        self.max_retries = max_retries
        self.llm = llm
        self.model_name = getattr(llm, 'model_name', 'unknown') if llm else 'none'

        if use_llm:
            self._prompt_template = load_profile_prompt()
        else:
            self._prompt_template = None

    def extract_profile(
        self,
        user: dict,
    ) -> dict:
        """
        Extract a structured preference profile from a user's dislikes.

        Args:
            user: User dict with "dislikes" list of {"musical": str, "reason": str}.

        Returns:
            Structured profile dict following PROFILE_SCHEMA.
        """
        dislikes = user.get("dislikes", [])
        if not dislikes:
            return {k: [] for k in PROFILE_SCHEMA}

        reasons = [
            f"{d['musical']}: {d['reason']}"
            for d in dislikes
            if d.get("reason")
        ]

        if not reasons:
            return {k: [] for k in PROFILE_SCHEMA}

        if not self.use_llm:
            return extract_profile_rule_based([d.get("reason", "") for d in dislikes])

        return self._extract_with_llm(reasons)

    def _extract_with_llm(self, dislike_entries: list[str]) -> dict:
        """Call LLM to extract structured profile."""
        entries_text = "\n".join(f"- {e}" for e in dislike_entries)
        prompt = self._prompt_template.format(dislike_entries=entries_text)

        for attempt in range(self.max_retries):
            try:
                raw = self.llm.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=1024,
                )
                parsed = self._parse_profile(raw)
                if parsed is not None:
                    return parsed

                # Repair pass: keep thinking ability but force strict JSON at the end.
                repaired_raw = self._force_json_repair(raw)
                repaired = self._parse_profile(repaired_raw)
                if repaired is not None:
                    return repaired

                logger.warning(f"Could not parse profile response after repair: {self._sanitize_model_output(raw)[:200]}")
                return {k: [] for k in PROFILE_SCHEMA}
            except Exception as e:
                logger.warning(f"Profile extraction failed (attempt {attempt + 1}): {e}")

        # Fallback to rule-based
        logger.warning("LLM profile extraction failed, falling back to rule-based.")
        return extract_profile_rule_based(dislike_entries)

    def _parse_profile(self, raw: str) -> dict | None:
        """Parse LLM profile extraction response; returns None on failure."""
        cleaned = self._sanitize_model_output(raw)

        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                # Validate against schema
                profile = {k: [] for k in PROFILE_SCHEMA}
                for key in PROFILE_SCHEMA:
                    if key in parsed and isinstance(parsed[key], list):
                        profile[key] = [str(v) for v in parsed[key]]
                return profile
            except (json.JSONDecodeError, ValueError):
                pass

        return None

    def _sanitize_model_output(self, raw: str) -> str:
        """
        Remove <think> blocks and markdown wrappers before JSON parsing.
        """
        cleaned = raw or ""
        cleaned = re.sub(r"<think>.*?</think>\s*", "", cleaned, flags=re.DOTALL)
        if "<think>" in cleaned and "</think>" not in cleaned:
            cleaned = cleaned.split("<think>", 1)[0]
        cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"```\s*$", "", cleaned)
        return cleaned.strip()

    def _force_json_repair(self, raw: str) -> str:
        messages = [
            {"role": "system", "content": "你是严格的JSON格式化器。"},
            {"role": "user", "content": f"{JSON_REPAIR_PROMPT}\n\n原始输出：\n{raw}"},
        ]
        return self.llm.chat(messages=messages, temperature=0.0, max_tokens=256)

    def profile_to_query(self, profile: dict) -> str:
        """
        Convert a structured profile into a retrieval query string.
        This can be used as an additional negative query for the retriever.

        Args:
            profile: Structured profile dict.

        Returns:
            Query string summarizing avoidance dimensions.
        """
        parts = []
        for key, label in [
            ("avoid_genres", "Genres to avoid"),
            ("avoid_themes", "Themes to avoid"),
            ("avoid_styles", "Styles to avoid"),
            ("avoid_elements", "Elements to avoid"),
        ]:
            values = profile.get(key, [])
            if values:
                parts.append(f"{label}: {', '.join(values)}")

        return " | ".join(parts) if parts else ""

    def profile_to_prompt_section(self, profile: dict) -> str:
        """
        Format profile as a structured section for injection into generation prompts.

        Returns:
            Markdown-formatted profile section, or empty string if no avoidance dimensions.
        """
        lines = []
        for key, label in [
            ("avoid_genres", "🚫 Genres to avoid"),
            ("avoid_themes", "🚫 Themes to avoid"),
            ("avoid_styles", "🚫 Styles to avoid"),
            ("avoid_elements", "🚫 Specific elements to avoid"),
            ("tolerance_notes", "✅ Acceptable despite negatives"),
        ]:
            values = profile.get(key, [])
            if values:
                lines.append(f"**{label}:** {', '.join(values)}")

        return "\n".join(lines) if lines else ""
