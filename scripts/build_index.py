"""
build_index.py — One-shot script: preprocess questionnaire data + build all indices.

Usage:
    cd E:/csc5051_final_proj
    python scripts/build_index.py [--config configs/config.yaml]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config
from src.data.preprocess import preprocess, build_preference_pairs
from src.data.knowledge import sync_kb_from_musical_tags
from src.retrieval.indexer import build_all_indices
from src.domain_lexicon import load_domain_lexicon


def main():
    parser = argparse.ArgumentParser(description="Build MeloMatch indices")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    # Step 1: Build KB from musical_tags.csv (single source of truth)
    kb_path = Path(config["paths"]["knowledge_base"]) / "musicals.jsonl"
    tags_path = Path(config["paths"]["raw_data"]) / "musical_tags.csv"
    print("=" * 60)
    print("Step 1: Knowledge Base (from musical_tags)")
    print("=" * 60)
    sync_kb_from_musical_tags(
        tags_csv_path=str(tags_path),
        kb_path=str(kb_path),
        overwrite=True,
    )

    # Step 2: Preprocess questionnaire data
    print("\n" + "=" * 60)
    print("Step 2: Preprocess Questionnaires")
    print("=" * 60)
    raw_dir = Path(config["paths"]["raw_data"])
    processed_dir = Path(config["paths"]["processed_data"])

    reviews_file = raw_dir / "subjective_reviews.csv"
    profile_file = raw_dir / "user_profile.csv"

    if not reviews_file.exists() or not profile_file.exists():
        print(f"[WARN] Missing required files in {raw_dir}.")
        print("  Required: subjective_reviews.csv and user_profile.csv")
        print("  Skipping preprocessing; will attempt to build indices from existing data.")
    else:
        lex_cfg = (
            config.get("text_expansion", {})
            .get("domain_lexicon", {})
        )
        lex_enabled = lex_cfg.get("enabled", False)
        lex_mode = lex_cfg.get("mode", "prompt_context")
        lex_path = lex_cfg.get("path", str(raw_dir / "domain_lexicon.json"))
        max_exp = int(lex_cfg.get("max_expansions_per_text", 3))
        use_inline = lex_enabled and lex_mode == "inline_expand"
        lexicon_entries = load_domain_lexicon(lex_path) if use_inline else []
        if lex_enabled:
            print(
                f"[preprocess] Domain lexicon enabled (mode={lex_mode}): "
                f"{len(lexicon_entries) if use_inline else 0} inline entries from {lex_path}"
            )

        users = preprocess(
            reviews_path=str(reviews_file),
            profile_path=str(profile_file),
            output_path=str(processed_dir / "users.jsonl"),
            lexicon_entries=lexicon_entries,
            max_lexicon_expansions=max_exp,
        )
        build_preference_pairs(users, str(processed_dir))

    # Step 3: Build indices
    print("\n" + "=" * 60)
    print("Step 3: Build Retrieval Indices")
    print("=" * 60)
    pos_path = processed_dir / "positive_pairs.jsonl"
    neg_path = processed_dir / "negative_pairs.jsonl"

    if pos_path.exists() and neg_path.exists():
        build_all_indices(args.config)
    else:
        print("[WARN] Processed data not found. Skipping index build.")
        print(f"  Expected: {pos_path} and {neg_path}")

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
