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
    "output_cost_per_reasoning_token": ("reasoning_output_per_mtok", MILLION),
    "output_cost_per_image": ("per_image", 1),
    "input_cost_per_character": ("per_character_input", 1),
    "ocr_cost_per_page": ("per_page", 1),
    "cache_creation_input_token_cost": ("cache_write_per_mtok", MILLION),
    "input_cost_per_token_batches": ("batch_input_per_mtok", MILLION),
    "output_cost_per_token_batches": ("batch_output_per_mtok", MILLION),
    "input_cost_per_audio_token": ("audio_input_per_mtok", MILLION),
    "output_cost_per_audio_token": ("audio_output_per_mtok", MILLION),
    "input_cost_per_image": ("input_per_image", 1),
}

# Capability mapping: source field -> target capability key
CAPABILITY_MAP = {
    "supports_function_calling": "function_calling",
    "supports_parallel_function_calling": "parallel_function_calling",
    "supports_native_streaming": "streaming",
    "supports_system_messages": "system_messages",
    "supports_vision": "vision",
    "supports_audio_input": "audio_input",
    "supports_audio_output": "audio_output",
    "supports_video_input": "video_input",
    "supports_computer_use": "computer_use",
    "supports_embedding_image_input": "image_input_embedding",
    "supports_pdf_input": "pdf_input",
    "supports_response_schema": "response_schema",
    "supports_prompt_caching": "prompt_caching",
    "supports_web_search": "web_search",
    "supports_reasoning": "reasoning",
    "supports_assistant_prefill": "assistant_prefill",
    "supports_tool_choice": "tool_choice",
}

# LiteLLM fields for per-second pricing, in priority order
_PER_SECOND_INPUT_FIELDS = [
    "input_cost_per_audio_per_second",
    "input_cost_per_video_per_second",
    "input_cost_per_second",
]
_PER_SECOND_OUTPUT_FIELDS = [
    "output_cost_per_video_per_second",
    "output_cost_per_second",
]

# Tiered pricing: (input_field, output_field, threshold)
_TIER_FIELDS = [
    ("input_cost_per_token_above_128k_tokens", "output_cost_per_token_above_128k_tokens", 128_000),
    ("input_cost_per_token_above_200k_tokens", "output_cost_per_token_above_200k_tokens", 200_000),
]

# Model prefixes that support the Responses API alongside their primary mode.
# Excludes realtime/audio/search variants which use specialized APIs.
_RESPONSES_CAPABLE_PREFIXES = [
    "gpt-4o", "gpt-4.1", "gpt-4.5", "gpt-5",
    "o1", "o3", "o4",
]
_RESPONSES_EXCLUDED_INFIXES = [
    "realtime", "audio", "search", "codex", "oss", "transcribe", "tts",
]

# Valid modality values
_VALID_INPUT_MODALITIES = {"text", "image", "audio", "video"}
_VALID_OUTPUT_MODALITIES = {"text", "image", "audio", "video"}


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

    # Direct field mappings
    for src_field, (dst_field, multiplier) in PRICING_FIELD_MAP.items():
        value = source_entry.get(src_field)
        if value is not None and value > 0:
            pricing[dst_field] = round(value * multiplier, 6)
            has_any = True

    # Per-second pricing (priority-based merge)
    for field in _PER_SECOND_INPUT_FIELDS:
        value = source_entry.get(field)
        if value is not None and value > 0:
            pricing["per_second_input"] = round(value, 10)
            has_any = True
            break

    for field in _PER_SECOND_OUTPUT_FIELDS:
        value = source_entry.get(field)
        if value is not None and value > 0:
            pricing["per_second_output"] = round(value, 10)
            has_any = True
            break

    # Tiered pricing
    tiers = _build_tiers(source_entry, pricing)
    if tiers:
        pricing["tiers"] = tiers
        has_any = True

    return pricing if has_any else None


def _build_tiers(source_entry: dict, base_pricing: dict) -> list[dict] | None:
    """Build tiered pricing from above_Nk fields."""
    tiers = []
    base_input = base_pricing.get("input_per_mtok")
    base_output = base_pricing.get("output_per_mtok")

    if base_input is None and base_output is None:
        return None

    for input_field, output_field, threshold in _TIER_FIELDS:
        elevated_input = source_entry.get(input_field)
        elevated_output = source_entry.get(output_field)

        if elevated_input is not None or elevated_output is not None:
            # Base tier
            tier1 = {
                "up_to_tokens": threshold,
                "input_per_mtok": base_input or 0,
                "output_per_mtok": base_output or 0,
            }
            # Elevated tier
            tier2_input = round(elevated_input * MILLION, 6) if elevated_input else (base_input or 0)
            tier2_output = round(elevated_output * MILLION, 6) if elevated_output else (base_output or 0)
            tier2 = {
                "up_to_tokens": 0,
                "input_per_mtok": tier2_input,
                "output_per_mtok": tier2_output,
            }
            tiers = [tier1, tier2]
            break  # Only use first matching tier set

    return tiers if tiers else None


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
        gen3a_turbo -> Gen3a Turbo
    """
    # Normalize underscores to hyphens for consistent processing
    normalized = canonical.replace("_", "-")

    for prefix, brand in _BRAND_PREFIXES:
        if normalized == prefix:
            return brand
        if normalized.startswith(prefix + "-"):
            rest = normalized[len(prefix) + 1:]
            suffix = " ".join(seg.capitalize() for seg in rest.split("-"))
            return f"{brand} {suffix}"

    # Check for lowercase prefixes (o1, o3, o4)
    first_seg = normalized.split("-")[0]
    if first_seg in _LOWERCASE_PREFIXES:
        rest = normalized[len(first_seg) + 1:] if len(normalized) > len(first_seg) else ""
        if rest:
            suffix = " ".join(seg.capitalize() for seg in rest.split("-"))
            return f"{first_seg} {suffix}"
        return first_seg

    # Fallback: capitalize each segment
    return " ".join(seg.capitalize() for seg in normalized.split("-"))


def convert_modalities(source_entry: dict, capabilities: dict | None) -> dict | None:
    """Convert modalities from source, falling back to inference from capabilities.

    Uses LiteLLM's supported_modalities / supported_output_modalities if present,
    otherwise infers from capability flags.
    """
    src_inputs = source_entry.get("supported_modalities")
    src_outputs = source_entry.get("supported_output_modalities")

    if src_inputs or src_outputs:
        inputs = [m for m in (src_inputs or ["text"]) if m in _VALID_INPUT_MODALITIES]
        outputs = [m for m in (src_outputs or ["text"]) if m in _VALID_OUTPUT_MODALITIES]
        if not inputs:
            inputs = ["text"]
        if not outputs:
            outputs = ["text"]
        return {"input": inputs, "output": outputs}

    return _infer_modalities(capabilities)


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
    if capabilities.get("video_input"):
        inputs.append("video")
    if capabilities.get("audio_output"):
        outputs.append("audio")

    # Only return if we have something beyond the default text/text
    if len(inputs) > 1 or len(outputs) > 1:
        return {"input": inputs, "output": outputs}
    return None


def _is_responses_capable(canonical: str) -> bool:
    """Return True if this model supports the Responses API alongside chat."""
    if not any(canonical.startswith(p) for p in _RESPONSES_CAPABLE_PREFIXES):
        return False
    if any(ex in canonical for ex in _RESPONSES_EXCLUDED_INFIXES):
        return False
    return True


def _infer_modes_from_modalities(modes: list[str], output_modalities: list[str]) -> None:
    """Add modes implied by output modalities (mutates modes in place)."""
    if "image" in output_modalities and "image_generation" not in modes:
        modes.append("image_generation")
    if "video" in output_modalities and "video_generation" not in modes:
        modes.append("video_generation")


def _build_modes(canonical: str, source_entry: dict) -> list[str]:
    """Build the modes list for a model from source data."""
    mode = source_entry.get("mode", "chat")
    modes = [mode] if mode in VALID_MODES else ["chat"]
    if "chat" in modes and "responses" not in modes and _is_responses_capable(canonical):
        modes.append("responses")
    # Infer additional modes from output modalities
    output_modalities = source_entry.get("supported_output_modalities", [])
    _infer_modes_from_modalities(modes, output_modalities)
    return modes


def create_model_entry(canonical: str, source_entry: dict) -> dict:
    """Build a full model dict conforming to the schema, auto-populated from source."""
    caps = convert_capabilities(source_entry)
    modalities = convert_modalities(source_entry, caps)

    entry = {
        "display_name": generate_display_name(canonical),
        "modes": _build_modes(canonical, source_entry),
    }

    # Optional fields — only include if they have actual values
    optional = {
        "deprecation_date": source_entry.get("deprecation_date"),
        "source_url": source_entry.get("source"),
        "modalities": modalities,
        "capabilities": caps,
        "context_window": source_entry.get("max_input_tokens") if isinstance(source_entry.get("max_input_tokens"), int) else None,
        "max_output_tokens": source_entry.get("max_output_tokens") if isinstance(source_entry.get("max_output_tokens"), int) else None,
        "max_images_per_request": source_entry.get("max_images_per_prompt") if isinstance(source_entry.get("max_images_per_prompt"), int) else None,
        "max_audio_length_seconds": _convert_audio_length(source_entry),
        "max_video_length_seconds": source_entry.get("max_video_length") if isinstance(source_entry.get("max_video_length"), (int, float)) else None,
        "max_pdf_size_mb": source_entry.get("max_pdf_size_mb") if isinstance(source_entry.get("max_pdf_size_mb"), (int, float)) else None,
        "max_videos_per_request": source_entry.get("max_videos_per_prompt") if isinstance(source_entry.get("max_videos_per_prompt"), int) else None,
        "max_audio_per_request": source_entry.get("max_audio_per_prompt") if isinstance(source_entry.get("max_audio_per_prompt"), int) else None,
        "output_vector_size": source_entry.get("output_vector_size") if isinstance(source_entry.get("output_vector_size"), int) else None,
        "pricing": convert_pricing(source_entry),
    }
    for key, value in optional.items():
        if value is not None:
            entry[key] = value

    return entry


def _convert_audio_length(source_entry: dict) -> int | None:
    """Convert audio length to seconds. LiteLLM uses max_audio_length_hours."""
    hours = source_entry.get("max_audio_length_hours")
    if isinstance(hours, (int, float)) and hours > 0:
        return int(round(hours * 3600))
    return None


def create_provider_model_entry(canonical: str, source_entry: dict | None = None) -> dict:
    """Build a provider_model dict conforming to the schema."""
    pm = {
        "model_ref": canonical,
        "enabled": True,
    }

    # Optional fields — only include if they have actual values
    if source_entry:
        optional = {
            "rate_limits": _extract_rate_limits(source_entry),
            "endpoints": _extract_list(source_entry, "supported_endpoints"),
            "regions": _extract_list(source_entry, "supported_regions"),
        }
        for key, value in optional.items():
            if value is not None:
                pm[key] = value

    return pm


def _extract_rate_limits(source_entry: dict) -> dict | None:
    """Extract rate limits from source entry."""
    if not source_entry:
        return None
    rpm = source_entry.get("rpm")
    tpm = source_entry.get("tpm")
    rpd = source_entry.get("rpd")
    if rpm is not None or tpm is not None or rpd is not None:
        return {
            "rpm": int(rpm) if isinstance(rpm, (int, float)) else None,
            "tpm": int(tpm) if isinstance(tpm, (int, float)) else None,
            "rpd": int(rpd) if isinstance(rpd, (int, float)) else None,
        }
    return None


def _extract_list(source_entry: dict, field: str) -> list | None:
    """Extract a list field from source, returning None if empty/missing."""
    if not source_entry:
        return None
    value = source_entry.get(field)
    if isinstance(value, list) and value:
        return value
    return None


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

        # Update modalities (use direct conversion)
        if not model.get("modalities"):
            modalities = convert_modalities(entry, model.get("capabilities"))
            if modalities:
                model["modalities"] = modalities

        # Update limits (only if source values are present and target is empty)
        if isinstance(entry.get("max_input_tokens"), int) and not model.get("context_window"):
            model["context_window"] = entry["max_input_tokens"]
        if isinstance(entry.get("max_output_tokens"), int) and not model.get("max_output_tokens"):
            model["max_output_tokens"] = entry["max_output_tokens"]
        if isinstance(entry.get("max_images_per_prompt"), int) and not model.get("max_images_per_request"):
            model["max_images_per_request"] = entry["max_images_per_prompt"]
        if isinstance(entry.get("max_video_length"), (int, float)) and not model.get("max_video_length_seconds"):
            model["max_video_length_seconds"] = entry["max_video_length"]
        if not model.get("max_audio_length_seconds"):
            audio_secs = _convert_audio_length(entry)
            if audio_secs:
                model["max_audio_length_seconds"] = audio_secs
        if isinstance(entry.get("max_pdf_size_mb"), (int, float)) and not model.get("max_pdf_size_mb"):
            model["max_pdf_size_mb"] = entry["max_pdf_size_mb"]
        if isinstance(entry.get("max_videos_per_prompt"), int) and not model.get("max_videos_per_request"):
            model["max_videos_per_request"] = entry["max_videos_per_prompt"]
        if isinstance(entry.get("max_audio_per_prompt"), int) and not model.get("max_audio_per_request"):
            model["max_audio_per_request"] = entry["max_audio_per_prompt"]
        if isinstance(entry.get("output_vector_size"), int) and not model.get("output_vector_size"):
            model["output_vector_size"] = entry["output_vector_size"]

        # Update deprecation_date
        if entry.get("deprecation_date") and not model.get("deprecation_date"):
            model["deprecation_date"] = entry["deprecation_date"]

        # Update source_url
        if entry.get("source") and not model.get("source_url"):
            model["source_url"] = entry["source"]

        # Update modes (only if valid and not already present)
        if entry.get("mode") in VALID_MODES:
            if not model.get("modes"):
                model["modes"] = [entry["mode"]]
            elif entry["mode"] not in model["modes"]:
                model["modes"].append(entry["mode"])

        # Add "responses" mode for known Responses-API-capable models
        modes = model.get("modes", [])
        if "chat" in modes and "responses" not in modes and _is_responses_capable(canonical):
            modes.append("responses")

        # Infer additional modes from output modalities (source or already-set)
        output_mods = (
            entry.get("supported_output_modalities")
            or model.get("modalities", {}).get("output", [])
        )
        _infer_modes_from_modalities(modes, output_mods)

        # Auto-create provider_model if provider is known
        if provider and provider in models_data.get("providers", {}):
            pm_key = f"{provider}/{canonical}"
            if pm_key not in models_data.get("provider_models", {}):
                models_data["provider_models"][pm_key] = create_provider_model_entry(canonical, entry)
            else:
                # Enrich existing provider_model with rate limits / endpoints / regions
                pm = models_data["provider_models"][pm_key]
                if pm.get("rate_limits") is None:
                    rate_limits = _extract_rate_limits(entry)
                    if rate_limits:
                        pm["rate_limits"] = rate_limits
                if pm.get("endpoints") is None:
                    endpoints = _extract_list(entry, "supported_endpoints")
                    if endpoints:
                        pm["endpoints"] = endpoints
                if pm.get("regions") is None:
                    regions = _extract_list(entry, "supported_regions")
                    if regions:
                        pm["regions"] = regions

            # Update provider's supported_modes if this model's mode is new
            mode = entry.get("mode")
            if mode and mode in VALID_MODES:
                provider_data = models_data["providers"][provider]
                supported = provider_data.get("supported_modes")
                if supported is not None and mode not in supported:
                    supported.append(mode)

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
