#!/usr/bin/env python3
"""Import Chatbot Arena (LMArena) rankings into models.json.

Fetches leaderboard data from the lmarena/arena-catalog GitHub repo and merges
ELO ratings + computed ranks into the rankings section of models.json.

Data sources:
  - leaderboard-text.json — text rankings (full, coding, math, creative_writing)
  - leaderboard-vision.json — vision rankings (full)

Usage:
    python scripts/import_arena_rankings.py
    python scripts/import_arena_rankings.py --dry-run
    python scripts/import_arena_rankings.py --source /path/to/dir
    python scripts/import_arena_rankings.py --overwrite
    python scripts/import_arena_rankings.py --categories all
"""

import argparse
import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/lmarena/arena-catalog/main/data"

# Mapping: (source_file, source_category) -> target ranking key
DEFAULT_CATEGORIES = {
    ("leaderboard-text.json", "full"): "chatbot_arena",
    ("leaderboard-text.json", "coding"): "chatbot_arena_coding",
    ("leaderboard-text.json", "math"): "chatbot_arena_math",
    ("leaderboard-text.json", "creative_writing"): "chatbot_arena_creative_writing",
    ("leaderboard-vision.json", "full"): "chatbot_arena_vision",
}

# Files to fetch
LEADERBOARD_FILES = ["leaderboard-text.json", "leaderboard-vision.json"]


def fetch_leaderboard(url: str) -> dict:
    """Download JSON from URL."""
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode())


def load_leaderboard_file(path: Path) -> dict:
    """Load leaderboard JSON from local file."""
    with open(path) as f:
        return json.load(f)


def build_model_lookup(models_data: dict) -> dict[str, list[str]]:
    """Build case-insensitive lookup: lowercase key -> list of actual keys."""
    lookup: dict[str, list[str]] = {}
    for key in models_data.get("models", {}):
        lower = key.lower()
        if lower not in lookup:
            lookup[lower] = []
        lookup[lower].append(key)
    return lookup


def import_arena_rankings(
    leaderboards: dict[str, dict],
    models_data: dict,
    categories: dict[tuple[str, str], str],
    merge: bool = True,
    as_of: str | None = None,
) -> tuple[int, int, list[str]]:
    """Import arena rankings into models_data.

    Returns (updated_model_count, skipped_count, unmatched_arena_names).
    """
    if as_of is None:
        as_of = date.today().isoformat()

    lookup = build_model_lookup(models_data)
    updated_keys: set[str] = set()
    all_unmatched: set[str] = set()

    for (file_name, src_category), target_key in categories.items():
        leaderboard = leaderboards.get(file_name)
        if leaderboard is None:
            print(f"Warning: {file_name} not loaded, skipping {target_key}")
            continue

        category_data = leaderboard.get(src_category)
        if category_data is None:
            print(f"Warning: category '{src_category}' not found in {file_name}, skipping {target_key}")
            continue

        # Sort by rating descending to compute ranks
        sorted_models = sorted(
            category_data.items(),
            key=lambda item: item[1].get("rating", 0),
            reverse=True,
        )

        for rank, (arena_name, arena_data) in enumerate(sorted_models, start=1):
            rating = arena_data.get("rating")
            if rating is None:
                continue

            elo = round(rating)

            # Case-insensitive match
            matched_keys = lookup.get(arena_name.lower(), [])
            if not matched_keys:
                all_unmatched.add(arena_name)
                continue

            ranking_entry = {
                "elo": elo,
                "rank": rank,
                "as_of": as_of,
            }

            for model_key in matched_keys:
                model = models_data["models"][model_key]
                if model.get("rankings") is None:
                    model["rankings"] = {}

                if merge and target_key in model["rankings"]:
                    existing = model["rankings"][target_key]
                    existing_date = existing.get("as_of", "")
                    if as_of >= existing_date:
                        if existing.get("elo") == elo and existing.get("rank") == rank:
                            continue
                        model["rankings"][target_key] = ranking_entry
                else:
                    model["rankings"][target_key] = ranking_entry

                updated_keys.add(model_key)

    skipped = len(all_unmatched)
    return len(updated_keys), skipped, sorted(all_unmatched)


def discover_all_categories(leaderboards: dict[str, dict]) -> dict[tuple[str, str], str]:
    """Build category mapping for all categories found in loaded leaderboards."""
    categories: dict[tuple[str, str], str] = {}
    for file_name, leaderboard in leaderboards.items():
        prefix = file_name.replace("leaderboard-", "").replace(".json", "")
        for src_category in leaderboard:
            if prefix == "text" and src_category == "full":
                target_key = "chatbot_arena"
            elif prefix == "text":
                target_key = f"chatbot_arena_{src_category}"
            else:
                # vision, text-style-control, etc.
                if src_category == "full":
                    target_key = f"chatbot_arena_{prefix}"
                else:
                    target_key = f"chatbot_arena_{prefix}_{src_category}"
            categories[(file_name, src_category)] = target_key
    return categories


def main():
    parser = argparse.ArgumentParser(description="Import Chatbot Arena rankings into models.json")
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--source", type=Path, default=None,
                        help="Local directory with leaderboard JSON files (default: fetch from GitHub)")
    parser.add_argument("--models", type=Path, default=repo_root / "models.json",
                        help="Path to models.json (default: repo root)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path (default: overwrite models.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing rankings instead of merging")
    parser.add_argument("--categories", default="default",
                        help="'default' for key categories, 'all' for every category in the data")
    parser.add_argument("--as-of", default=None,
                        help="Date for as_of field (default: today, format: YYYY-MM-DD)")
    args = parser.parse_args()

    # Load leaderboard data
    leaderboards: dict[str, dict] = {}
    for filename in LEADERBOARD_FILES:
        if args.source:
            filepath = args.source / filename
            if not filepath.exists():
                print(f"Warning: {filepath} not found, skipping")
                continue
            print(f"Loading {filepath}")
            leaderboards[filename] = load_leaderboard_file(filepath)
        else:
            url = f"{BASE_URL}/{filename}"
            print(f"Fetching {url}")
            try:
                leaderboards[filename] = fetch_leaderboard(url)
            except Exception as e:
                print(f"Error fetching {url}: {e}", file=sys.stderr)
                sys.exit(1)

    if not leaderboards:
        print("Error: no leaderboard data loaded", file=sys.stderr)
        sys.exit(1)

    # Determine categories
    if args.categories == "all":
        categories = discover_all_categories(leaderboards)
        print(f"Importing all {len(categories)} categories")
    else:
        categories = DEFAULT_CATEGORIES
        print(f"Importing {len(categories)} default categories")

    # Load models.json
    with open(args.models) as f:
        models_data = json.load(f)

    total_models = len(models_data.get("models", {}))
    print(f"Loaded {total_models} models from {args.models}")

    # Import
    updated, skipped, unmatched = import_arena_rankings(
        leaderboards, models_data,
        categories=categories,
        merge=not args.overwrite,
        as_of=args.as_of,
    )

    print(f"\nResults:")
    print(f"  Models updated: {updated}")
    print(f"  Arena models not in models.json: {skipped}")

    if unmatched:
        print(f"\nUnmatched arena models ({len(unmatched)}):")
        for name in unmatched[:20]:
            print(f"  - {name}")
        if len(unmatched) > 20:
            print(f"  ... and {len(unmatched) - 20} more")

    if args.dry_run:
        print("\nDry run — no changes written")
        return

    # Write output
    output_path = args.output or args.models
    with open(output_path, "w") as f:
        json.dump(models_data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\nWritten to {output_path}")


if __name__ == "__main__":
    main()
