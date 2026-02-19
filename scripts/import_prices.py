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

# Valid mode values from the schema
VALID_MODES = {
    "chat", "completion", "embedding",
    "image_generation", "image_edit",
    "video_generation", "video_edit",
    "audio_speech", "audio_transcription",
    "rerank", "moderation", "ocr", "search",
    "responses", "code_interpreter",
}

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


# Brand prefix -> display form. Order matters: longer prefixes first.
_BRAND_PREFIXES = [
    ("text-embedding", "Text Embedding"),
    ("dall-e", "DALL-E"),
    ("deepseek", "DeepSeek"),
    ("gemini", "Gemini"),
    ("claude", "Claude"),
    ("mistral", "Mistral"),
    ("command", "Command"),
    ("whisper", "Whisper"),
    ("llama", "Llama"),
    ("grok", "Grok"),
    ("gpt", "GPT"),
    ("tts", "TTS"),
]

# Single-segment prefixes kept lowercase (model names like o1, o3, o4-mini)
_LOWERCASE_PREFIXES = {"o1", "o3", "o4"}


def generate_display_name(canonical: str) -> str:
    """Convert a canonical model name to a human-readable display name.

    Examples:
        gpt-4o-mini  -> GPT-4o Mini
        claude-opus-4-6 -> Claude Opus 4 6
        gemini-2.5-pro -> Gemini 2.5 Pro
        deepseek-r1 -> DeepSeek R1
        o4-mini -> o4 Mini
    """
    for prefix, brand in _BRAND_PREFIXES:
        if canonical == prefix:
            return brand
        if canonical.startswith(prefix + "-"):
            rest = canonical[len(prefix) + 1:]
            suffix = " ".join(seg.capitalize() for seg in rest.split("-"))
            return f"{brand} {suffix}"

    # Check for lowercase prefixes (o1, o3, o4)
    first_seg = canonical.split("-")[0]
    if first_seg in _LOWERCASE_PREFIXES:
        rest = canonical[len(first_seg) + 1:] if len(canonical) > len(first_seg) else ""
        if rest:
            suffix = " ".join(seg.capitalize() for seg in rest.split("-"))
            return f"{first_seg} {suffix}"
        return first_seg

    # Fallback: capitalize each segment
    return " ".join(seg.capitalize() for seg in canonical.split("-"))


def _infer_modalities(capabilities: dict | None) -> dict | None:
    """Infer input/output modalities from capability flags."""
    if not capabilities:
        return None

    inputs = ["text"]
    outputs = ["text"]

    if capabilities.get("vision"):
        inputs.append("image")
    if capabilities.get("audio_input"):
        inputs.append("audio")
    if capabilities.get("audio_output"):
        outputs.append("audio")

    # Only return if we have something beyond the default text/text
    if len(inputs) > 1 or len(outputs) > 1:
        return {"input": inputs, "output": outputs}
    return None


def create_model_entry(canonical: str, source_entry: dict) -> dict:
    """Build a full model dict conforming to the schema, auto-populated from source."""
    caps = convert_capabilities(source_entry)
    modalities = _infer_modalities(caps)

    return {
        "display_name": generate_display_name(canonical),
        "description": None,
        "owned_by": None,
        "family": None,
        "release_date": None,
        "deprecation_date": None,
        "tags": None,
        "mode": source_entry.get("mode", "chat") if source_entry.get("mode", "chat") in VALID_MODES else "chat",
        "modalities": modalities,
        "capabilities": caps,
        "context_window": source_entry.get("max_input_tokens") if isinstance(source_entry.get("max_input_tokens"), int) else None,
        "max_output_tokens": source_entry.get("max_output_tokens") if isinstance(source_entry.get("max_output_tokens"), int) else None,
        "max_images_per_request": None,
        "max_audio_length_seconds": None,
        "max_video_length_seconds": None,
        "max_pdf_size_mb": None,
        "pricing": convert_pricing(source_entry),
        "parameters": None,
        "rankings": None,
    }


def create_provider_model_entry(canonical: str) -> dict:
    """Build a minimal provider_model dict conforming to the schema."""
    return {
        "model_ref": canonical,
        "provider_model_id": None,
        "enabled": True,
        "pricing": None,
        "context_window": None,
        "max_output_tokens": None,
        "rate_limits": None,
        "endpoints": None,
        "regions": None,
    }


def import_prices(source_data: list[dict], models_data: dict, merge: bool = True) -> tuple[int, int, int]:
    """Import pricing from source into models_data. Returns (updated, skipped, created)."""
    updated = 0
    skipped = 0
    created = 0

    for entry in source_data:
        model_name = entry.get("model_name", "")
        if not model_name:
            skipped += 1
            continue

        # Extract provider prefix and canonical name
        if "/" in model_name:
            parts = model_name.split("/")
            provider = parts[0]
            canonical = parts[-1]
        else:
            provider = None
            canonical = model_name

        # Auto-create model if not found
        if canonical not in models_data.get("models", {}):
            models_data["models"][canonical] = create_model_entry(canonical, entry)
            created += 1

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

        # Update limits (only if source values are integers)
        if isinstance(entry.get("max_input_tokens"), int) and not model.get("context_window"):
            model["context_window"] = entry["max_input_tokens"]
        if isinstance(entry.get("max_output_tokens"), int) and not model.get("max_output_tokens"):
            model["max_output_tokens"] = entry["max_output_tokens"]

        # Update mode (only if valid)
        if entry.get("mode") in VALID_MODES and not model.get("mode"):
            model["mode"] = entry["mode"]

        # Auto-create provider_model if provider is known
        if provider and provider in models_data.get("providers", {}):
            pm_key = f"{provider}/{canonical}"
            if pm_key not in models_data.get("provider_models", {}):
                models_data["provider_models"][pm_key] = create_provider_model_entry(canonical)

        updated += 1

    return updated, skipped, created


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
    updated, skipped, created = import_prices(source_data, models_data, merge=not args.overwrite)
    print(f"Updated: {updated}, Skipped: {skipped}, Created: {created}")

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
