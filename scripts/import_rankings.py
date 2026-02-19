#!/usr/bin/env python3
"""Import benchmark rankings into models.json.

Reads a rankings source file (JSON) and merges benchmark scores into the
models section of models.json.

Expected source format (JSON array):
[
  {
    "model": "gpt-4o",
    "benchmarks": {
      "chatbot_arena": { "elo": 1287, "rank": 3, "as_of": "2026-02-01" },
      "mmlu_pro": { "score": 0.887, "as_of": "2025-06-01" },
      "humaneval": { "score": 0.902, "as_of": "2025-06-01" }
    }
  }
]

Usage:
    python scripts/import_rankings.py <source_path_or_url>
    python scripts/import_rankings.py rankings.json --dry-run
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path


def load_source(source: str) -> list[dict]:
    """Load rankings data from a file path or URL."""
    if source.startswith("http://") or source.startswith("https://"):
        with urllib.request.urlopen(source) as response:
            return json.loads(response.read().decode())
    else:
        with open(source) as f:
            return json.load(f)


def import_rankings(source_data: list[dict], models_data: dict, merge: bool = True) -> tuple[int, int]:
    """Import rankings from source into models_data. Returns (updated_count, skipped_count)."""
    updated = 0
    skipped = 0

    for entry in source_data:
        model_name = entry.get("model", "")
        benchmarks = entry.get("benchmarks", {})

        if not model_name or not benchmarks:
            skipped += 1
            continue

        if model_name not in models_data.get("models", {}):
            skipped += 1
            continue

        model = models_data["models"][model_name]

        if model.get("rankings") is None:
            model["rankings"] = {}

        for bench_key, bench_data in benchmarks.items():
            if not isinstance(bench_data, dict):
                continue

            # Build ranking entry with expected fields
            ranking_entry = {}
            if "score" in bench_data:
                ranking_entry["score"] = bench_data["score"]
            if "elo" in bench_data:
                ranking_entry["elo"] = bench_data["elo"]
            if "rank" in bench_data:
                ranking_entry["rank"] = bench_data["rank"]
            if "as_of" in bench_data:
                ranking_entry["as_of"] = bench_data["as_of"]

            if not ranking_entry:
                continue

            if merge and bench_key in model["rankings"]:
                # Merge: update only new fields, prefer newer as_of
                existing = model["rankings"][bench_key]
                existing_date = existing.get("as_of", "")
                new_date = ranking_entry.get("as_of", "")
                if new_date >= existing_date:
                    model["rankings"][bench_key] = ranking_entry
            else:
                model["rankings"][bench_key] = ranking_entry

        updated += 1

    return updated, skipped


def main():
    parser = argparse.ArgumentParser(description="Import benchmark rankings into models.json")
    parser.add_argument("source", help="Path or URL to rankings source JSON")
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--models", type=Path, default=repo_root / "models.json",
                        help="Path to models.json (default: repo root)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path (default: overwrite models.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing values instead of merging")
    args = parser.parse_args()

    # Load source
    print(f"Loading rankings source: {args.source}")
    try:
        source_data = load_source(args.source)
    except Exception as e:
        print(f"Error loading source: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(source_data, list):
        print("Error: source must be a JSON array of model entries", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(source_data)} entries in source")

    # Load models.json
    with open(args.models) as f:
        models_data = json.load(f)

    # Import
    updated, skipped = import_rankings(source_data, models_data, merge=not args.overwrite)
    print(f"Updated: {updated}, Skipped: {skipped}")

    if args.dry_run:
        print("Dry run — no changes written")
        return

    # Write output
    output_path = args.output or args.models
    with open(output_path, "w") as f:
        json.dump(models_data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()
