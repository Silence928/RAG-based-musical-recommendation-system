"""
nli_filter.py — Post-retrieval NLI verification filter (ARAG-inspired).

After retrieval and before generation, checks each candidate KB entry against
the user's dislike reasons via natural language inference. Candidates that
contradict the user's negative preferences are penalized or removed.

Reference: "Agentic RAG for Personalized Recommendation" (2506.21931)
  - Their NLI Agent uses an LLM to check entailment between user preferences
    and candidate items, filtering out contradictions before final ranking.

We adapt this for MeloMatch:
  - Input: user's dislike reasons (free-text) + candidate musical metadata
  - Output: per-candidate avoidance score (0.0 = safe, 1.0 = strongly contradicts)
  - Candidates above a threshold are filtered or down-weighted.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger("melomatch.nli_filter")

PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "prompts"
JSON_REPAIR_PROMPT = """
请把下面这段输出重写为唯一一个JSON对象，不要输出其他任何文本：
{"score": <0.0到1.0之间的小数>, "reasoning": "<简短解释>"}
"""


def load_nli_prompt() -> str:
    """Load NLI check prompt template."""
    path = PROMPT_DIR / "nli_check.txt"
    if not path.exists():
        raise FileNotFoundError(f"NLI prompt template not found: {path}")
    return path.read_text(encoding="utf-8").strip()


class NLIFilter:
    """
    Filters candidate musicals by checking whether they conflict with
    a user's stated dislikes via LLM-based NLI.
    """

    def __init__(
        self,
        llm,
        threshold: float = 0.6,
        max_retries: int = 2,
    ):
        """
        Args:
            llm: LLM backend (LocalLLM or APILLM) with .chat() method.
            threshold: Avoidance score above which candidates are filtered.
            max_retries: Retry count on failure.
        """
        self.llm = llm
        self.model_name = getattr(llm, 'model_name', 'unknown')
        self.threshold = threshold
        self.max_retries = max_retries
        self._prompt_template = load_nli_prompt()

    def score_candidate(
        self,
        candidate: dict,
        dislike_reasons: list[str],
    ) -> dict:
        """
        Score a single candidate against the user's dislike reasons.

        Args:
            candidate: KB entry dict with at least "name", "synopsis", "genres", "themes".
            dislike_reasons: List of free-text dislike reasons from the user.

        Returns:
            {"score": float, "reasoning": str, "filtered": bool}
        """
        if not dislike_reasons:
            return {"score": 0.0, "reasoning": "No dislikes provided.", "filtered": False}

        # Build candidate description
        genres = ", ".join(candidate.get("genres", []))
        themes = ", ".join(candidate.get("themes", []))
        style = ", ".join(candidate.get("style", []))
        candidate_desc = (
            f"{candidate.get('name', 'Unknown')} ({candidate.get('era', '?')}): "
            f"{candidate.get('synopsis', 'N/A')} "
            f"[Genres: {genres}] [Themes: {themes}] [Style: {style}]"
        )

        reasons_text = "\n".join(f"- {r}" for r in dislike_reasons)

        prompt = self._prompt_template.format(
            dislike_reasons=reasons_text,
            candidate_description=candidate_desc,
        )

        for attempt in range(self.max_retries):
            try:
                raw = self.llm.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=1024,
                )
                parsed = self._parse_nli_response(raw)
                if parsed is not None:
                    return parsed

                # Repair pass: keep model thinking ability, but force final JSON object.
                repaired_raw = self._force_json_repair(raw)
                repaired = self._parse_nli_response(repaired_raw)
                if repaired is not None:
                    return repaired

                logger.warning(f"NLI parse failed after repair: {self._sanitize_model_output(raw)[:200]}")
                return {"score": 0.0, "reasoning": "Parse failed.", "filtered": False}
            except Exception as e:
                logger.warning(f"NLI call failed (attempt {attempt + 1}): {e}")

        # Fallback: don't filter on failure
        return {"score": 0.0, "reasoning": "NLI check failed.", "filtered": False}

    def filter_candidates(
        self,
        candidates: list[dict],
        dislike_reasons: list[str],
    ) -> list[tuple[dict, float]]:
        """
        Filter a list of candidate KB entries.

        Args:
            candidates: List of KB entry dicts.
            dislike_reasons: User's dislike reasons.

        Returns:
            List of (candidate, avoidance_score) tuples, sorted by score ascending
            (safest first). Candidates above threshold are excluded.
        """
        if not dislike_reasons:
            return [(c, 0.0) for c in candidates]

        results = []
        for candidate in candidates:
            result = self.score_candidate(candidate, dislike_reasons)
            if not result["filtered"] and result["score"] < self.threshold:
                results.append((candidate, result["score"]))
            else:
                logger.info(
                    f"NLI filtered: {candidate.get('name', '?')} "
                    f"(score={result['score']:.2f}, reason={result['reasoning'][:80]})"
                )

        # Sort by avoidance score ascending (safest candidates first)
        results.sort(key=lambda x: x[1])
        return results

    def _parse_nli_response(self, raw: str) -> dict | None:
        """Parse LLM NLI response. Returns None when parsing fails."""
        cleaned = self._sanitize_model_output(raw)

        # Try JSON extraction
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                score = float(parsed.get("score", 0.0))
                score = max(0.0, min(1.0, score))  # clamp
                reasoning = str(parsed.get("reasoning", ""))
                return {
                    "score": score,
                    "reasoning": reasoning,
                    "filtered": score >= self.threshold,
                }
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: try to extract a number
        num_match = re.search(r"(\d+\.?\d*)", cleaned)
        if num_match:
            score = float(num_match.group(1))
            score = max(0.0, min(1.0, score))
            return {"score": score, "reasoning": cleaned[:200], "filtered": score >= self.threshold}

        return None

    def _sanitize_model_output(self, raw: str) -> str:
        """
        Keep compatibility with models that output chain-of-thought.
        Removes <think> blocks and markdown wrappers before JSON parsing.
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
