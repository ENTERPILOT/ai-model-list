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
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.rankings import (
    ARENA_CATALOG_BASE_URL,
    ARENA_LEADERBOARD_FILENAMES,
    DEFAULT_ARENA_CATEGORIES,
    discover_all_categories,
    import_arena_rankings,
)


def fetch_leaderboard(url: str) -> dict:
    """Download JSON from URL."""
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode())


def load_leaderboard_file(path: Path) -> dict:
    """Load leaderboard JSON from local file."""
    with open(path) as f:
        return json.load(f)


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
    for filename in ARENA_LEADERBOARD_FILENAMES:
        if args.source:
            filepath = args.source / filename
            if not filepath.exists():
                print(f"Warning: {filepath} not found, skipping")
                continue
            print(f"Loading {filepath}")
            leaderboards[filename] = load_leaderboard_file(filepath)
        else:
            url = f"{ARENA_CATALOG_BASE_URL}/{filename}"
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
        categories = DEFAULT_ARENA_CATEGORIES
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
