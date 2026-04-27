"""
curate_kb.py — Expand the musical knowledge base from public sources.

Scrapes Wikipedia / IBDB / Douban for musical metadata.
Usage:
    cd E:/csc5051_final_proj
    python scripts/curate_kb.py [--source wikipedia] [--limit 200]
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config
from src.data.knowledge import (
    load_knowledge_base, save_knowledge_base, validate_entry, KB_SCHEMA_KEYS
)

# ======================== Wikipedia Scraper ========================
# TODO: Wikipedia is blocked by GFW. Replace with a GFW-accessible source
#       (Douban API, Baidu Baike, or manual curation from IBDB via university VPN).
#       All KB entries must come from verifiable public sources — no fabricated data.

# Notable musicals to curate (expand this list)
MUSICAL_LIST = [
    # Classic / Golden Age
    "Oklahoma!", "West Side Story", "The Sound of Music", "My Fair Lady",
    "Fiddler on the Roof", "Hello, Dolly!", "A Chorus Line", "Chicago",
    "Cabaret", "Guys and Dolls", "Annie", "The King and I", "South Pacific",
    "Grease", "Sweeney Todd",
    # Megamusicals
    "Les Misérables", "The Phantom of the Opera", "Cats", "Miss Saigon",
    "Evita", "Jesus Christ Superstar", "Joseph and the Amazing Technicolor Dreamcoat",
    # Contemporary / 2000s+
    "Hamilton", "Wicked", "Dear Evan Hansen", "Come From Away",
    "The Book of Mormon", "Avenue Q", "Spring Awakening", "In the Heights",
    "Hadestown", "Six", "Beetlejuice", "Mean Girls", "Moulin Rouge!",
    "The Greatest Showman", "Rent", "Next to Normal", "Fun Home",
    "Kinky Boots", "Matilda", "School of Rock", "Aladdin", "Frozen",
    "The Lion King", "Beauty and the Beast", "Mary Poppins",
    # Jukebox
    "Mamma Mia!", "Jersey Boys", "& Juliet", "Tina: The Tina Turner Musical",
    "MJ: The Musical", "Moulin Rouge!", "Ain't Too Proud",
    # Darker / Dramatic
    "The Last Five Years", "Assassins", "Parade", "Ragtime",
    "Once", "The Band's Visit", "Natasha, Pierre & The Great Comet of 1812",
    # Chinese musicals
    "蝶", "阿波罗尼亚", "赵氏孤儿", "白夜行",
    "摇滚莫扎特", "巴黎圣母院", "罗密欧与朱丽叶",
    # Korean / Japanese
    "Elisabeth", "Mozart!", "Death Note: The Musical",
    "Phantom (Korean)", "Jack the Ripper (Korean)",
]


def fetch_wikipedia_summary(title: str) -> dict | None:
    """
    Fetch musical metadata from Wikipedia REST API.
    Returns a partial KB entry or None.
    """
    import urllib.request
    import urllib.parse

    safe_title = urllib.parse.quote(title.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe_title}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MeloMatch/0.1 (academic research)"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if data.get("type") == "disambiguation":
            # Try with "(musical)" suffix
            safe_title_m = urllib.parse.quote(f"{title}_(musical)".replace(" ", "_"))
            url_m = f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe_title_m}"
            req_m = urllib.request.Request(url_m, headers={"User-Agent": "MeloMatch/0.1 (academic research)"})
            with urllib.request.urlopen(req_m, timeout=10) as resp_m:
                data = json.loads(resp_m.read().decode("utf-8"))

        extract = data.get("extract", "")
        if not extract:
            return None

        entry = {
            "id": re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_"),
            "name": data.get("title", title),
            "name_zh": "",  # Fill manually or via Douban
            "synopsis": extract[:500],
            "genres": [],  # Needs manual curation or NLP extraction
            "themes": [],
            "style": [],
            "era": "",
            "awards": [],
            "creators": [],
            "notable_cast": [],
            "tradition": "",
        }

        # Try to extract year from description
        year_match = re.search(r"\b(19|20)\d{2}\b", data.get("description", ""))
        if year_match:
            entry["era"] = year_match.group()

        return entry

    except Exception as e:
        print(f"  [WARN] Failed to fetch '{title}': {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Curate musical KB")
    parser.add_argument("--source", default="wikipedia", choices=["wikipedia"])
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    kb_path = str(Path(config["paths"]["knowledge_base"]) / "musicals.jsonl")

    # Load existing KB
    existing = []
    if Path(kb_path).exists():
        existing = load_knowledge_base(kb_path)
    existing_names = {e["name"].lower() for e in existing}
    existing_ids = {e["id"] for e in existing}

    print(f"[curate] Existing KB: {len(existing)} musicals")
    print(f"[curate] Target: {args.limit} musicals")
    print(f"[curate] Source: {args.source}")

    new_entries = []
    for title in MUSICAL_LIST:
        if len(existing) + len(new_entries) >= args.limit:
            break
        if title.lower() in existing_names:
            continue

        print(f"  Fetching: {title}")
        entry = fetch_wikipedia_summary(title)
        if entry and entry["id"] not in existing_ids:
            issues = validate_entry(entry)
            if issues:
                print(f"    [WARN] Validation: {issues} (adding anyway, needs manual curation)")
            new_entries.append(entry)
            existing_ids.add(entry["id"])

        time.sleep(0.5)  # Rate limiting

    all_entries = existing + new_entries
    save_knowledge_base(all_entries, kb_path)
    print(f"\n[curate] KB now has {len(all_entries)} musicals (+{len(new_entries)} new)")
    print(f"[curate] [WARN] New entries need manual curation for: genres, themes, style, awards, tradition")
    print(f"[curate] Run: python scripts/curate_kb.py to re-fetch, or edit {kb_path} directly")


if __name__ == "__main__":
    main()
