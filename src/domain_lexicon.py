"""
domain_lexicon.py — Domain jargon lexicon loading and soft expansion.

Design goal:
- Do NOT hard-replace user words.
- Keep original text and append short explanations when jargon is detected.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def load_domain_lexicon(path: str) -> list[dict]:
    """
    Load domain lexicon entries from JSON or CSV.

    Supported JSON formats:
      1) [{"term": "...", "explanation": "...", "aliases": ["..."]}, ...]
      2) {"entries": [...same as above...]}

    Supported CSV columns:
      - term (required)
      - explanation (required)
      - aliases (optional, separated by |)
    """
    p = Path(path)
    if not p.exists():
        return []

    suffix = p.suffix.lower()
    if suffix == ".json":
        return _load_json_lexicon(p)
    if suffix == ".csv":
        return _load_csv_lexicon(p)
    raise ValueError(f"Unsupported lexicon file format: {p}")


def expand_text_with_lexicon(
    text: str,
    lexicon_entries: list[dict],
    max_expansions: int = 3,
) -> str:
    """
    Soft expansion (no replacement):
    - Keep original text unchanged.
    - Append a compact explanation tail if matched jargon exists.
    """
    if not text or not lexicon_entries:
        return text

    text_lower = text.lower()
    matches: list[tuple[str, str]] = []
    seen_terms: set[str] = set()

    for entry in lexicon_entries:
        term = str(entry.get("term", "")).strip()
        explanation = str(entry.get("explanation", "")).strip()
        aliases = [str(a).strip() for a in entry.get("aliases", []) if str(a).strip()]
        if not term or not explanation:
            continue

        keys = [term] + aliases
        hit = False
        for key in keys:
            if _contains_key(text, text_lower, key):
                hit = True
                break
        if not hit:
            continue

        canonical = term.lower()
        if canonical in seen_terms:
            continue
        seen_terms.add(canonical)
        matches.append((term, explanation))
        if len(matches) >= max_expansions:
            break

    if not matches:
        return text

    # Avoid appending multiple times if text already has expansion tail.
    if "术语解释：" in text:
        return text

    tails = [f"{term}={explanation}" for term, explanation in matches]
    return f"{text} 术语解释：{'；'.join(tails)}"


def match_lexicon_terms(
    text: str,
    lexicon_entries: list[dict],
    max_matches: int = 5,
) -> list[tuple[str, str]]:
    """
    Match terms from text without modifying the original text.
    Returns list of (term, explanation).
    """
    if not text or not lexicon_entries:
        return []

    text_lower = text.lower()
    matches: list[tuple[str, str]] = []
    seen_terms: set[str] = set()

    for entry in lexicon_entries:
        term = str(entry.get("term", "")).strip()
        explanation = str(entry.get("explanation", "")).strip()
        aliases = [str(a).strip() for a in entry.get("aliases", []) if str(a).strip()]
        if not term or not explanation:
            continue

        keys = [term] + aliases
        if any(_contains_key(text, text_lower, key) for key in keys):
            canonical = term.lower()
            if canonical in seen_terms:
                continue
            seen_terms.add(canonical)
            matches.append((term, explanation))
            if len(matches) >= max_matches:
                break

    return matches


def build_lexicon_context_block(
    texts: list[str],
    lexicon_entries: list[dict],
    max_terms: int = 8,
) -> str:
    """
    Build a compact glossary context for prompt grounding.
    """
    if not texts or not lexicon_entries:
        return "(未命中术语)"

    merged = " ".join(t for t in texts if t)
    matches = match_lexicon_terms(merged, lexicon_entries, max_matches=max_terms)
    if not matches:
        return "(未命中术语)"

    lines = [f"- {term}: {explanation}" for term, explanation in matches]
    return "\n".join(lines)


def _contains_key(text: str, text_lower: str, key: str) -> bool:
    if not key:
        return False
    # Latin keys: case-insensitive
    if key.isascii():
        return key.lower() in text_lower
    # Non-Latin keys: direct contains
    return key in text


def _load_json_lexicon(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        entries = payload.get("entries", [])
    elif isinstance(payload, list):
        entries = payload
    else:
        entries = []
    return _normalize_entries(entries)


def _load_csv_lexicon(path: Path) -> list[dict]:
    entries: list[dict] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            aliases_raw = str(row.get("aliases", "")).strip()
            aliases = [a.strip() for a in aliases_raw.split("|") if a.strip()]
            entries.append(
                {
                    "term": str(row.get("term", "")).strip(),
                    "explanation": str(row.get("explanation", "")).strip(),
                    "aliases": aliases,
                }
            )
    return _normalize_entries(entries)


def _normalize_entries(entries: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for entry in entries:
        term = str(entry.get("term", "")).strip()
        explanation = str(entry.get("explanation", "")).strip()
        aliases = entry.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []
        aliases = [str(a).strip() for a in aliases if str(a).strip()]
        if term and explanation:
            normalized.append(
                {
                    "term": term,
                    "explanation": explanation,
                    "aliases": aliases,
                }
            )
    return normalized
