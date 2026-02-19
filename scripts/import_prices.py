#!/usr/bin/env python3
"""Import pricing data from ai-model-price-list format into models.json.

Reads a pricing source file (JSON) and merges pricing + capabilities into
the existing models.json. The source format uses per-token costs; this script
converts to per-million-token costs.

Usage:
    python scripts/import_prices.py <source_path_or_url>
    python scripts/import_prices.py /path/to/pricing.json
    python scripts/import_prices.py https://raw.githubusercontent.com/.../pricing.json
    python scripts/import_prices.py <source> --dry-run
    python scripts/import_prices.py <source> --output /path/to/output.json
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path


MILLION = 1_000_000

# Field mapping: source field -> (target field, multiplier)
PRICING_FIELD_MAP = {
    "input_cost_per_token": ("input_per_mtok", MILLION),
    "output_cost_per_token": ("output_per_mtok", MILLION),
    "cache_read_input_token_cost": ("cached_input_per_mtok", MILLION),
}

# Capability mapping: source field -> target capability key
CAPABILITY_MAP = {
    "supports_function_calling": "function_calling",
    "supports_parallel_function_calling": "parallel_function_calling",
    "supports_streaming": "streaming",
    "supports_system_messages": "system_messages",
    "supports_vision": "vision",
    "supports_audio_input": "audio_input",
    "supports_audio_output": "audio_output",
    "supports_pdf_input": "pdf_input",
    "supports_response_schema": "response_schema",
    "supports_prompt_caching": "prompt_caching",
    "supports_web_search": "web_search",
    "supports_reasoning": "reasoning",
    "supports_assistant_prefill": "assistant_prefill",
}


def load_source(source: str) -> list[dict]:
    """Load pricing data from a file path or URL."""
    if source.startswith("http://") or source.startswith("https://"):
        with urllib.request.urlopen(source) as response:
            return json.loads(response.read().decode())
    else:
        with open(source) as f:
            return json.load(f)


def convert_pricing(source_entry: dict) -> dict | None:
    """Convert a source pricing entry to our pricing format."""
    pricing = {"currency": "USD"}
    has_any = False

    for src_field, (dst_field, multiplier) in PRICING_FIELD_MAP.items():
        value = source_entry.get(src_field)
        if value is not None and value > 0:
            pricing[dst_field] = round(value * multiplier, 6)
            has_any = True
        else:
            pricing[dst_field] = None

    # Fill in remaining pricing fields with null
    for field in ["reasoning_output_per_mtok", "per_image", "per_second_input",
                   "per_second_output", "per_character_input", "per_request",
                   "per_page", "tiers"]:
        pricing[field] = None

    return pricing if has_any else None


def convert_capabilities(source_entry: dict) -> dict | None:
    """Extract capability flags from a source entry."""
    caps = {}
    for src_field, dst_field in CAPABILITY_MAP.items():
        value = source_entry.get(src_field)
        if value is True:
            caps[dst_field] = True

    return caps if caps else None


def import_prices(source_data: list[dict], models_data: dict, merge: bool = True) -> tuple[int, int]:
    """Import pricing from source into models_data. Returns (updated_count, skipped_count)."""
    updated = 0
    skipped = 0

    for entry in source_data:
        model_name = entry.get("model_name", "")
        if not model_name:
            skipped += 1
            continue

        # Try to match by model_name in our models section
        # Source may use "provider/model" format or just "model"
        canonical = model_name.split("/")[-1] if "/" in model_name else model_name

        if canonical not in models_data.get("models", {}):
            skipped += 1
            continue

        model = models_data["models"][canonical]

        # Update pricing
        pricing = convert_pricing(entry)
        if pricing:
            if merge and model.get("pricing"):
                # Merge: only update null fields
                for key, value in pricing.items():
                    if value is not None and model["pricing"].get(key) is None:
                        model["pricing"][key] = value
            else:
                model["pricing"] = pricing

        # Update capabilities
        caps = convert_capabilities(entry)
        if caps:
            if model.get("capabilities") is None:
                model["capabilities"] = {}
            for key, value in caps.items():
                if merge and key in model["capabilities"]:
                    continue
                model["capabilities"][key] = value

        # Update limits
        if entry.get("max_input_tokens") and not model.get("context_window"):
            model["context_window"] = entry["max_input_tokens"]
        if entry.get("max_output_tokens") and not model.get("max_output_tokens"):
            model["max_output_tokens"] = entry["max_output_tokens"]

        # Update mode
        if entry.get("mode") and not model.get("mode"):
            model["mode"] = entry["mode"]

        updated += 1

    return updated, skipped


def main():
    parser = argparse.ArgumentParser(description="Import pricing from ai-model-price-list")
    parser.add_argument("source", help="Path or URL to pricing source JSON")
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
    print(f"Loading pricing source: {args.source}")
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
    updated, skipped = import_prices(source_data, models_data, merge=not args.overwrite)
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
