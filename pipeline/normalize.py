"""Normalize raw source payloads into shared evidence records."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from pipeline.types import SourceEvidence


USD_PER_TOKEN_TO_USD_PER_MTOK = 1_000_000
CENTS_PER_TOKEN_TO_USD_PER_MTOK = 10_000
PROVIDER_SLUG_ALIASES = {
    "x-ai": "xai",
    "aws": "bedrock",
    "azure_ai": "azure",
    "azure-openai": "azure",
    "google": "gemini",
    "fireworks-ai": "fireworks",
    "fireworks_ai": "fireworks",
    "mistral-ai": "mistral",
    "text-completion-openai": "openai",
    "together-ai": "together",
    "together_ai": "together",
    "vertex-ai": "vertex_ai",
    "vertex_ai-embedding-models": "vertex_ai",
    "vertex_ai-language-models": "vertex_ai",
    "vertex_ai-text-models": "vertex_ai",
}
KNOWN_PROVIDER_SLUGS = {
    "anthropic",
    "azure",
    "bedrock",
    "cerebras",
    "cohere",
    "deepinfra",
    "deepseek",
    "fireworks",
    "gemini",
    "google",
    "groq",
    "mistral",
    "openai",
    "ovhcloud",
    "runway",
    "together",
    "vertex_ai",
    "x-ai",
    "xai",
}
UNSTABLE_ALIAS_TOKENS = ("latest", "beta", "preview", "alpha", "experimental", "free")
DEPLOYMENT_TIER_TOKENS = {"free"}
DATE_SUFFIX_PATTERN = re.compile(r"(?:[-_.])(?:\d{4}|\d{8})$")
XAI_TOKEN_PRICE_DIVISOR = 10_000
XAI_UNIT_PRICE_DIVISOR = 1_000_000_000
DISPLAY_NAME_SPLIT_PATTERN = re.compile(r"[-_./:]+")
MODE_HINTS = (
    ("embedding", "embedding"),
    ("rerank", "rerank"),
    ("moderation", "moderation"),
    ("ocr", "ocr"),
    ("text-to-image", "image_generation"),
    ("image generation", "image_generation"),
    ("image generator", "image_generation"),
    ("text-to-video", "video_generation"),
    ("video generation", "video_generation"),
    ("text-to-speech", "audio_speech"),
    ("text to speech", "audio_speech"),
    ("speech synthesis", "audio_speech"),
    ("tts", "audio_speech"),
    ("speech-to-text", "audio_transcription"),
    ("speech to text", "audio_transcription"),
    ("transcription", "audio_transcription"),
    ("transcribe", "audio_transcription"),
    ("realtime", "realtime"),
)
DISPLAY_NAME_ACRONYMS = {"ai", "api", "asr", "gpt", "json", "ocr", "oss", "pdf", "tts", "ui", "ux"}
SUPPORTED_MODALITIES = {"text", "image", "audio", "video"}
MODALITY_ORDER = {"text": 0, "image": 1, "audio": 2, "video": 3}
OPENAI_RESPONSES_MODEL_PATTERN = re.compile(r"^(?:chatgpt-4o(?:[.-]|$)|gpt-(?:4(?:[.-]|$)|4o(?:[.-]|$)|5(?:[.-]|$)))")
ENDPOINT_MODE_HINTS = {
    "/v1/images/generations": "image_generation",
    "/v1/images/edits": "image_edit",
    "/v1/images/variations": "image_edit",
    "/v1/audio/speech": "audio_speech",
    "/v1/audio/transcriptions": "audio_transcription",
    "/v1/audio/translations": "audio_transcription",
    "/v1/embeddings": "embedding",
    "/v1/responses": "responses",
    "/v1/realtime": "realtime",
}
MODE_MODALITY_HINTS = {
    "image_generation": {"input": {"text"}, "output": {"image"}},
    "image_edit": {"input": {"text", "image"}, "output": {"image"}},
    "video_generation": {"input": {"text"}, "output": {"video"}},
    "video_edit": {"input": {"text", "video"}, "output": {"video"}},
    "audio_speech": {"input": {"text"}, "output": {"audio"}},
    "audio_transcription": {"input": {"audio"}, "output": {"text"}},
}
OWNER_BY_MODEL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^claude"), "anthropic"),
    (re.compile(r"^(?:chatgpt|codex|curie|babbage|ada|davinci|dall-e|gpt(?:[-.]|$)|o[134](?:[-.]|$)|text-embedding|text-moderation|sora)"), "openai"),
    (re.compile(r"^(?:gemini|gemma)"), "gemini"),
    (re.compile(r"^grok"), "xai"),
    (re.compile(r"^deepseek"), "deepseek"),
    (re.compile(r"^(?:command|embed|rerank)"), "cohere"),
    (re.compile(r"^(?:codestral|devstral|magistral|ministral|mistral|mixtral|open-mistral|pixtral)"), "mistral"),
)
BEDROCK_PREFIX_PATTERN = re.compile(r"^(?:(?:regional|global|us|eu|ap|sa|ca|me|af)\.)*anthropic\.")
BEDROCK_SUFFIX_PATTERN = re.compile(r"-v\d+:\d+$")


def normalize_provider_slug(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = value.strip().lower()
    if not normalized:
        return None

    return PROVIDER_SLUG_ALIASES.get(normalized, normalized)


def split_provider_model_name(model_name: str, fallback_provider: str | None = None) -> tuple[str | None, str | None]:
    if "/" in model_name:
        provider_slug, _, canonical_hint = model_name.partition("/")
        return normalize_provider_slug(provider_slug), _strip_nested_provider_prefixes(canonical_hint or None)

    return normalize_provider_slug(fallback_provider), model_name or None


def _strip_nested_provider_prefixes(model_name: str | None) -> str | None:
    canonical_hint = model_name
    while canonical_hint and "/" in canonical_hint:
        provider_candidate, _, remainder = canonical_hint.partition("/")
        normalized = normalize_provider_slug(provider_candidate)
        if normalized not in KNOWN_PROVIDER_SLUGS:
            break
        canonical_hint = remainder or None
    return canonical_hint


def is_rejected_model_id(
    model_id: str,
    rejection_policy: Mapping[str, Sequence[str]] | None = None,
) -> bool:
    rejections = rejection_policy or {}
    if model_id in rejections.get("exact_model_ids", ()):
        return True

    return any(model_id.startswith(prefix) for prefix in rejections.get("prefixes", ()))


def _to_float(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return float(stripped)

    return None


def _scale_price(value: float, multiplier: int) -> float:
    return float(Decimal(str(value)) * Decimal(multiplier))


def _duration_seconds_from_hours(value: Any) -> int | None:
    hours = _to_float(value)
    if hours is None:
        return None
    return int(round(hours * 3600))


def _merge_pricing_maps(*maps: Mapping[str, Any] | None) -> dict[str, Any] | None:
    pricing: dict[str, Any] = {}
    for candidate in maps:
        if not isinstance(candidate, Mapping):
            continue
        for key, value in candidate.items():
            if value is None:
                continue
            pricing[key] = value

    if not pricing:
        return None

    pricing.setdefault("currency", "USD")
    return pricing if len(pricing) > 1 else None


def _scale_cents_to_usd(value: Any) -> float | None:
    numeric = _to_float(value)
    if numeric is None:
        return None
    return float(Decimal(str(numeric)) / Decimal(100))


def infer_owned_by(model_id: str, display_name: str | None = None) -> str | None:
    normalized_model_id = model_id.strip().lower()
    candidates = [normalized_model_id]
    if display_name:
        candidates.append(display_name.strip().lower())

    for candidate in candidates:
        for pattern, owner in OWNER_BY_MODEL_PATTERNS:
            if pattern.match(candidate):
                return owner

    return None


def sanitize_provider_canonical_hint(model_id: str, provider_slug: str) -> str:
    if provider_slug == "bedrock":
        stripped = BEDROCK_PREFIX_PATTERN.sub("", model_id)
        stripped = BEDROCK_SUFFIX_PATTERN.sub("", stripped)
        return stripped
    return model_id


def _build_tiered_token_pricing(entry: Mapping[str, Any], pricing: Mapping[str, Any]) -> list[dict[str, float | int]] | None:
    base_input = _to_float(pricing.get("input_per_mtok"))
    base_output = _to_float(pricing.get("output_per_mtok"))
    if base_input is None or base_output is None:
        return None

    context_window = entry.get("max_input_tokens")
    input_pattern = re.compile(r"^input_cost_per_token_above_(\d+)k_tokens$")
    output_pattern = re.compile(r"^output_cost_per_token_above_(\d+)k_tokens$")
    input_thresholds = {
        int(match.group(1)) * 1_000: _scale_price(value, USD_PER_TOKEN_TO_USD_PER_MTOK)
        for key, raw_value in entry.items()
        if (match := input_pattern.match(key)) and (value := _to_float(raw_value)) is not None
    }
    output_thresholds = {
        int(match.group(1)) * 1_000: _scale_price(value, USD_PER_TOKEN_TO_USD_PER_MTOK)
        for key, raw_value in entry.items()
        if (match := output_pattern.match(key)) and (value := _to_float(raw_value)) is not None
    }

    tiers: list[dict[str, float | int]] = []
    for threshold in sorted(input_thresholds.keys() & output_thresholds.keys()):
        if context_window is not None and threshold > context_window:
            continue
        if not tiers:
            tiers.append(
                {
                    "up_to_tokens": threshold,
                    "input_per_mtok": base_input,
                    "output_per_mtok": base_output,
                }
            )
        upper_bound = context_window if isinstance(context_window, int) and context_window >= threshold else threshold
        tiers.append(
            {
                "up_to_tokens": upper_bound,
                "input_per_mtok": input_thresholds[threshold],
                "output_per_mtok": output_thresholds[threshold],
            }
        )

    return tiers or None


def _is_image_generation_summary_row(entry: Mapping[str, Any]) -> bool:
    name = entry.get("name")
    return isinstance(name, str) and "(image gen)" in name.lower()


def _pricing_from_usd_per_token(input_value: Any, output_value: Any) -> dict[str, float | str] | None:
    input_cost = _to_float(input_value)
    output_cost = _to_float(output_value)
    if input_cost is None and output_cost is None:
        return None

    pricing: dict[str, float | str] = {"currency": "USD"}
    if input_cost is not None:
        pricing["input_per_mtok"] = _scale_price(input_cost, USD_PER_TOKEN_TO_USD_PER_MTOK)
    if output_cost is not None:
        pricing["output_per_mtok"] = _scale_price(output_cost, USD_PER_TOKEN_TO_USD_PER_MTOK)
    return pricing


def _pricing_from_usd_per_mtok(input_value: Any, output_value: Any) -> dict[str, float | str] | None:
    input_cost = _to_float(input_value)
    output_cost = _to_float(output_value)
    if input_cost is None and output_cost is None:
        return None

    pricing: dict[str, float | str] = {"currency": "USD"}
    if input_cost is not None:
        pricing["input_per_mtok"] = input_cost
    if output_cost is not None:
        pricing["output_per_mtok"] = output_cost
    return pricing


def _pricing_from_portkey(entry: Mapping[str, Any]) -> dict[str, float | str] | None:
    pricing_config = entry.get("pricing_config") or {}
    payg = pricing_config.get("pay_as_you_go", {}) or {}
    batch = pricing_config.get("batch_config", {}) or {}
    pricing: dict[str, Any] = {"currency": "USD"}

    cents_token_fields = {
        ("request_token", "price"): "input_per_mtok",
        ("request_text_token", "price"): "input_per_mtok",
        ("response_token", "price"): "output_per_mtok",
        ("response_text_token", "price"): "output_per_mtok",
        ("cache_read_input_token", "price"): "cached_input_per_mtok",
        ("cache_read_text_input_token", "price"): "cached_input_per_mtok",
        ("cached_text_input_token", "price"): "cached_input_per_mtok",
        ("cache_write_input_token", "price"): "cache_write_per_mtok",
        ("cache_write_text_input_token", "price"): "cache_write_per_mtok",
        ("request_audio_token", "price"): "audio_input_per_mtok",
        ("response_audio_token", "price"): "audio_output_per_mtok",
        ("request_image_token", "price"): "input_image_per_mtok",
        ("response_image_token", "price"): "output_image_per_mtok",
        ("cache_read_image_input_token", "price"): "cached_input_image_per_mtok",
        ("cached_image_input_token", "price"): "cached_input_image_per_mtok",
    }
    for (section, field), target_field in cents_token_fields.items():
        numeric_value = _to_float(payg.get(section, {}).get(field))
        if numeric_value is not None:
            pricing[target_field] = _scale_price(numeric_value, CENTS_PER_TOKEN_TO_USD_PER_MTOK)

    batch_token_fields = {
        ("request_token", "price"): "batch_input_per_mtok",
        ("request_text_token", "price"): "batch_input_per_mtok",
        ("response_token", "price"): "batch_output_per_mtok",
        ("response_text_token", "price"): "batch_output_per_mtok",
    }
    for (section, field), target_field in batch_token_fields.items():
        numeric_value = _to_float(batch.get(section, {}).get(field))
        if numeric_value is not None:
            pricing[target_field] = _scale_price(numeric_value, CENTS_PER_TOKEN_TO_USD_PER_MTOK)

    image_generation_prices: list[dict[str, Any]] = []
    default_per_image: float | None = None
    image_prices = payg.get("image")
    if isinstance(image_prices, Mapping):
        for quality, quality_payload in image_prices.items():
            if not isinstance(quality_payload, Mapping):
                continue
            for size, price_payload in quality_payload.items():
                if not isinstance(price_payload, Mapping):
                    continue
                price = _scale_cents_to_usd(price_payload.get("price"))
                if price is None:
                    continue
                variant: dict[str, Any] = {"price": price}
                if quality != "default":
                    variant["quality"] = quality
                if size != "default":
                    variant["size"] = size
                image_generation_prices.append(variant)
                if quality == "default" and size == "default":
                    default_per_image = price
                elif default_per_image is None and quality == "standard" and size == "default":
                    default_per_image = price
                elif default_per_image is None and quality == "default" and size == "1024x1024":
                    default_per_image = price

    if image_generation_prices:
        pricing["image_generation_prices"] = image_generation_prices
        if default_per_image is not None:
            pricing["per_image"] = default_per_image
        elif len({variant["price"] for variant in image_generation_prices}) == 1:
            pricing["per_image"] = image_generation_prices[0]["price"]

    pricing = {
        key: value
        for key, value in pricing.items()
        if not (isinstance(value, (int, float)) and value == 0)
    }

    return pricing if len(pricing) > 1 else None


def _pricing_from_extended_fields(entry: Mapping[str, Any]) -> dict[str, Any] | None:
    pricing: dict[str, Any] = {"currency": "USD"}

    token_cost_fields = {
        "cache_read_input_token_cost": "cached_input_per_mtok",
        "cache_creation_input_token_cost": "cache_write_per_mtok",
        "output_cost_per_reasoning_token": "reasoning_output_per_mtok",
        "input_cost_per_audio_token": "audio_input_per_mtok",
        "output_cost_per_audio_token": "audio_output_per_mtok",
        "input_cost_per_image_token": "input_image_per_mtok",
        "output_cost_per_image_token": "output_image_per_mtok",
        "cache_read_input_image_token_cost": "cached_input_image_per_mtok",
    }
    for source_field, target_field in token_cost_fields.items():
        value = _to_float(entry.get(source_field))
        if value is not None:
            pricing[target_field] = _scale_price(value, USD_PER_TOKEN_TO_USD_PER_MTOK)

    raw_pricing = entry.get("pricing")
    if isinstance(raw_pricing, Mapping):
        openrouter_fields = {
            "input_cache_read": "cached_input_per_mtok",
            "input_cache_write": "cache_write_per_mtok",
        }
        for source_field, target_field in openrouter_fields.items():
            value = _to_float(raw_pricing.get(source_field))
            if value is not None:
                pricing[target_field] = _scale_price(value, USD_PER_TOKEN_TO_USD_PER_MTOK)

    if _is_image_generation_summary_row(entry):
        image_input = _to_float(entry.get("input"))
        image_output = _to_float(entry.get("output"))
        if image_input is not None:
            pricing["input_image_per_mtok"] = image_input
        if image_output is not None:
            pricing["output_image_per_mtok"] = image_output
    cached_input = _to_float(entry.get("input_cached"))
    if cached_input is not None:
        pricing["cached_input_per_mtok"] = cached_input

    mode = entry.get("mode")
    input_per_image = _to_float(entry.get("input_cost_per_image"))
    output_per_image = _to_float(entry.get("output_cost_per_image"))
    if output_per_image is not None:
        pricing["per_image"] = output_per_image
    elif input_per_image is not None:
        if mode == "image_generation":
            pricing["per_image"] = input_per_image
        else:
            pricing["input_per_image"] = input_per_image

    direct_price_fields = {
        "input_cost_per_second": "per_second_input",
        "output_cost_per_second": "per_second_output",
        "input_cost_per_audio_per_second": "per_second_input",
        "output_cost_per_audio_per_second": "per_second_output",
        "input_cost_per_character": "per_character_input",
    }
    for source_field, target_field in direct_price_fields.items():
        value = _to_float(entry.get(source_field))
        if value is not None:
            pricing[target_field] = value

    return pricing if len(pricing) > 1 else None


def _pricing_from_catalog_prices(value: Any) -> dict[str, float | str] | None:
    pricing_payload = value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for candidate in value:
            if not isinstance(candidate, Mapping):
                continue
            nested_pricing = candidate.get("prices")
            if isinstance(nested_pricing, Mapping):
                pricing_payload = nested_pricing
                break
        else:
            return None

    if not isinstance(pricing_payload, Mapping):
        return None

    pricing: dict[str, float | str] = {"currency": "USD"}
    field_map = {
        "input_mtok": "input_per_mtok",
        "output_mtok": "output_per_mtok",
        "cache_read_mtok": "cached_input_per_mtok",
        "cache_write_mtok": "cache_write_per_mtok",
        "batch_input_mtok": "batch_input_per_mtok",
        "batch_output_mtok": "batch_output_per_mtok",
        "input_audio_mtok": "audio_input_per_mtok",
        "output_audio_mtok": "audio_output_per_mtok",
        "per_second_input": "per_second_input",
        "per_second_output": "per_second_output",
        "per_character_input": "per_character_input",
        "per_character_output": "per_character_output",
        "per_image": "per_image",
        "input_per_image": "input_per_image",
    }
    for source_field, target_field in field_map.items():
        raw_value = pricing_payload.get(source_field)
        numeric_value = _to_float(raw_value)
        if numeric_value is None and isinstance(raw_value, Mapping):
            numeric_value = _to_float(raw_value.get("base"))
        if numeric_value is not None:
            pricing[target_field] = numeric_value

    image_generation_prices = pricing_payload.get("image_generation_prices")
    if isinstance(image_generation_prices, Sequence) and not isinstance(image_generation_prices, (str, bytes, bytearray)):
        normalized_image_generation_prices = []
        for entry in image_generation_prices:
            if not isinstance(entry, Mapping):
                continue
            price = _to_float(entry.get("price"))
            resolution = entry.get("resolution")
            if price is None or not isinstance(resolution, str) or not resolution:
                continue
            normalized_image_generation_prices.append(
                {
                    "resolution": resolution,
                    "price": price,
                }
            )
        if normalized_image_generation_prices:
            pricing["image_generation_prices"] = normalized_image_generation_prices

    return pricing if len(pricing) > 1 else None


def _extract_exact_match_ids(match_spec: Any) -> list[str]:
    matches: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            exact_value = node.get("equals")
            if isinstance(exact_value, str) and exact_value:
                matches.append(exact_value)
            for key in ("or", "and"):
                nested = node.get(key)
                if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes, bytearray)):
                    for child in nested:
                        visit(child)

    visit(match_spec)
    return matches


def _canonical_rank(model_id: str, raw_model_id: str) -> tuple[int, int, int, int, str]:
    lowered = model_id.lower()
    unstable = int(any(token in lowered for token in UNSTABLE_ALIAS_TOKENS))
    dated = int(bool(DATE_SUFFIX_PATTERN.search(lowered)))
    exact_raw = 0 if model_id == raw_model_id else 1
    return (unstable, dated, len(model_id), exact_raw, model_id)


def _choose_canonical_model_id(model: Mapping[str, Any]) -> str:
    raw_model_id = model["id"]
    candidates = {
        candidate
        for candidate in [raw_model_id, *_extract_exact_match_ids(model.get("match"))]
        if isinstance(candidate, str) and candidate
    }
    return min(candidates, key=lambda candidate: _canonical_rank(candidate, raw_model_id))


def _choose_preferred_canonical_hint(raw_model_id: str, *candidates: str | None) -> str | None:
    valid_candidates = {candidate for candidate in candidates if isinstance(candidate, str) and candidate}
    trimmed_raw_model_id = _strip_deployment_tier_suffix(raw_model_id)
    if trimmed_raw_model_id:
        valid_candidates.add(trimmed_raw_model_id)
    for candidate in tuple(valid_candidates):
        trimmed_candidate = _strip_deployment_tier_suffix(candidate)
        if trimmed_candidate:
            valid_candidates.add(trimmed_candidate)
    if not valid_candidates:
        return None
    return min(valid_candidates, key=lambda candidate: _canonical_rank(candidate, raw_model_id))


def _strip_deployment_tier_suffix(model_id: str | None) -> str | None:
    if not isinstance(model_id, str) or not model_id:
        return None
    base, separator, suffix = model_id.rpartition(":")
    if not separator or not base:
        return None
    if suffix.lower() not in DEPLOYMENT_TIER_TOKENS:
        return None
    return base


def _display_name_from_model_id(model_id: str) -> str:
    tokens = [token for token in DISPLAY_NAME_SPLIT_PATTERN.split(model_id) if token]
    display_tokens: list[str] = []

    for token in tokens:
        lowered = token.lower()
        if lowered in DISPLAY_NAME_ACRONYMS:
            display_tokens.append(lowered.upper())
        elif lowered.startswith("grok"):
            display_tokens.append(token.replace("grok", "Grok", 1))
        elif lowered.startswith("gpt"):
            display_tokens.append(token.replace("gpt", "GPT", 1))
        elif any(char.isdigit() for char in token) and any(char.isalpha() for char in token):
            display_tokens.append(token.upper() if lowered in DISPLAY_NAME_ACRONYMS else token)
        elif token.replace(".", "", 1).isdigit():
            display_tokens.append(token)
        else:
            display_tokens.append(token.title())

    return " ".join(display_tokens) if display_tokens else model_id


def _ordered_unique_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _sorted_modalities(values: Iterable[str]) -> list[str]:
    normalized = {value for value in values if value in SUPPORTED_MODALITIES}
    return sorted(normalized, key=lambda value: (MODALITY_ORDER.get(value, len(MODALITY_ORDER)), value))


def _infer_mode_from_values(*values: Any) -> str:
    for value in values:
        if not isinstance(value, str):
            continue
        lowered = value.lower()
        if lowered in {
            "chat",
            "responses",
            "embedding",
            "rerank",
            "moderation",
            "ocr",
            "image_generation",
            "image_edit",
            "video_generation",
            "video_edit",
            "audio_speech",
            "audio_transcription",
            "code_interpreter",
            "realtime",
        }:
            return lowered
        for token, mode in MODE_HINTS:
            if token in lowered:
                return mode
    return "chat"


def _modes_from_endpoints(endpoints: Sequence[str]) -> list[str]:
    return _ordered_unique_strings(
        ENDPOINT_MODE_HINTS[endpoint]
        for endpoint in endpoints
        if endpoint in ENDPOINT_MODE_HINTS
    )


def _infer_modalities_from_modes(modes: Sequence[str]) -> dict[str, list[str]] | None:
    inferred: dict[str, set[str]] = {}
    for mode in modes:
        mode_modalities = MODE_MODALITY_HINTS.get(mode)
        if not mode_modalities:
            continue
        for direction, values in mode_modalities.items():
            inferred.setdefault(direction, set()).update(values)

    if not inferred:
        return None

    return {
        direction: _sorted_modalities(values)
        for direction, values in inferred.items()
        if values
    } or None


def _infer_modalities_from_pricing(
    pricing: Mapping[str, Any] | None,
    modes: Sequence[str],
) -> dict[str, list[str]] | None:
    if not isinstance(pricing, Mapping):
        return None

    inferred: dict[str, set[str]] = {}

    if any(key in pricing for key in ("input_image_per_mtok", "cached_input_image_per_mtok", "input_per_image")):
        inferred.setdefault("input", set()).add("image")
    if any(key in pricing for key in ("output_image_per_mtok", "per_image", "image_generation_prices")):
        inferred.setdefault("output", set()).add("image")

    mode_set = set(modes)
    if "video_generation" in mode_set or "video_edit" in mode_set:
        if "per_second_input" in pricing:
            inferred.setdefault("input", set()).add("video")
        if "per_second_output" in pricing:
            inferred.setdefault("output", set()).add("video")

    if "audio_transcription" in mode_set and "per_second_input" in pricing:
        inferred.setdefault("input", set()).add("audio")
    if "audio_speech" in mode_set and "per_second_output" in pricing:
        inferred.setdefault("output", set()).add("audio")

    if not inferred:
        return None

    return {
        direction: _sorted_modalities(values)
        for direction, values in inferred.items()
        if values
    } or None


def _merge_modalities(*modality_maps: Mapping[str, Sequence[str]] | None) -> dict[str, list[str]] | None:
    merged: dict[str, set[str]] = {}
    for modality_map in modality_maps:
        if not isinstance(modality_map, Mapping):
            continue
        for direction in ("input", "output"):
            values = modality_map.get(direction)
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
                continue
            normalized_values = {
                value
                for value in values
                if isinstance(value, str) and value in SUPPORTED_MODALITIES
            }
            if normalized_values:
                merged.setdefault(direction, set()).update(normalized_values)

    if not merged:
        return None

    return {
        direction: _sorted_modalities(values)
        for direction, values in merged.items()
        if values
    } or None


def _apply_inferred_modalities(fields: dict[str, Any]) -> None:
    modes = fields.get("modes", [])
    pricing = fields.get("pricing")
    fields_modalities = fields.get("modalities")
    merged = _merge_modalities(
        fields_modalities if isinstance(fields_modalities, Mapping) else None,
        _infer_modalities_from_modes(modes if isinstance(modes, Sequence) and not isinstance(modes, (str, bytes, bytearray)) else []),
        _infer_modalities_from_pricing(pricing if isinstance(pricing, Mapping) else None, modes if isinstance(modes, Sequence) and not isinstance(modes, (str, bytes, bytearray)) else []),
    )
    if merged:
        fields["modalities"] = merged


def extract_official_catalog_fields(
    model: Mapping[str, Any],
    provider_slug: str,
    *,
    canonical_model_id: str | None = None,
    owner_providers: set[str] | None = None,
) -> dict[str, Any]:
    model_id = canonical_model_id or model.get("id") or model.get("name") or provider_slug
    mode = _infer_mode_from_values(
        model.get("mode"),
        model_id,
        model.get("name"),
        model.get("description"),
    )
    modes = [mode]
    if provider_slug in {"openai", "azure"} and mode == "chat" and OPENAI_RESPONSES_MODEL_PATTERN.match(str(model_id)):
        modes.append("responses")
    fields: dict[str, Any] = {
        "modes": _ordered_unique_strings(modes),
    }
    display_name = model.get("name")
    normalized_display_name = (
        display_name if isinstance(display_name, str) and display_name else _display_name_from_model_id(str(model_id))
    )
    fields["display_name"] = normalized_display_name

    inferred_owner = infer_owned_by(str(model_id), normalized_display_name)
    if inferred_owner is not None:
        fields["owned_by"] = inferred_owner
    elif owner_providers is None or provider_slug in owner_providers:
        fields["owned_by"] = provider_slug

    description = model.get("description")
    if isinstance(description, str) and description:
        fields["description"] = description

    context_window = model.get("context_window")
    if context_window is not None:
        fields["context_window"] = context_window

    pricing = _pricing_from_catalog_prices(model.get("prices"))
    if pricing is not None:
        fields["pricing"] = pricing

    _apply_inferred_modalities(fields)

    return fields


def _parse_xai_numeric(value: Any) -> float | None:
    if isinstance(value, str) and value.startswith("$n"):
        return float(value[2:])
    return _to_float(value)


def _parse_xai_integer(value: Any) -> int | None:
    numeric = _parse_xai_numeric(value)
    if numeric is None:
        return None
    return int(numeric)


def _xai_price_per_mtok(value: Any) -> float | None:
    numeric = _parse_xai_numeric(value)
    if numeric is None:
        return None
    return float(Decimal(str(numeric)) / Decimal(XAI_TOKEN_PRICE_DIVISOR))


def _xai_price_per_unit(value: Any) -> float | None:
    numeric = _parse_xai_numeric(value)
    if numeric is None:
        return None
    return float(Decimal(str(numeric)) / Decimal(XAI_UNIT_PRICE_DIVISOR))


def _pricing_from_xai_language_model(model: Mapping[str, Any]) -> dict[str, Any] | None:
    input_price = _xai_price_per_mtok(model.get("promptTextTokenPrice"))
    output_price = _xai_price_per_mtok(model.get("completionTextTokenPrice"))
    cached_price = _xai_price_per_mtok(model.get("cachedPromptTokenPrice"))
    if input_price is None and output_price is None and cached_price is None:
        return None

    pricing: dict[str, Any] = {"currency": "USD"}
    if input_price is not None:
        pricing["input_per_mtok"] = input_price
    if output_price is not None:
        pricing["output_per_mtok"] = output_price
    if cached_price is not None:
        pricing["cached_input_per_mtok"] = cached_price

    long_context_threshold = _parse_xai_integer(model.get("longContextThreshold"))
    max_prompt_length = _parse_xai_integer(model.get("maxPromptLength"))
    long_input_price = _xai_price_per_mtok(model.get("promptTextTokenPriceLongContext"))
    long_output_price = _xai_price_per_mtok(model.get("completionTokenPriceLongContext"))
    base_input_price = pricing.get("input_per_mtok")
    base_output_price = pricing.get("output_per_mtok")
    if (
        isinstance(long_context_threshold, int)
        and isinstance(max_prompt_length, int)
        and max_prompt_length > long_context_threshold
        and isinstance(base_input_price, float)
        and isinstance(base_output_price, float)
        and isinstance(long_input_price, float)
        and isinstance(long_output_price, float)
        and (long_input_price != base_input_price or long_output_price != base_output_price)
    ):
        pricing["tiers"] = [
            {
                "up_to_tokens": long_context_threshold,
                "input_per_mtok": base_input_price,
                "output_per_mtok": base_output_price,
            },
            {
                "up_to_tokens": max_prompt_length,
                "input_per_mtok": long_input_price,
                "output_per_mtok": long_output_price,
            },
        ]

    return pricing


def _pricing_from_xai_image_model(model: Mapping[str, Any]) -> dict[str, Any] | None:
    per_image = _xai_price_per_unit(model.get("imagePrice"))
    input_per_image = _xai_price_per_unit(model.get("pricePerInputImage"))
    if per_image is None and input_per_image is None:
        return None

    pricing: dict[str, Any] = {"currency": "USD"}
    if per_image is not None:
        pricing["per_image"] = per_image
    if input_per_image is not None:
        pricing["input_per_image"] = input_per_image
    return pricing


def _pricing_from_xai_video_model(model: Mapping[str, Any]) -> dict[str, Any] | None:
    output_prices = [
        _xai_price_per_unit(entry.get("pricePerSecond"))
        for entry in model.get("resolutionPricing", [])
        if isinstance(entry, Mapping)
    ]
    output_prices = [price for price in output_prices if price is not None]
    input_per_image = _xai_price_per_unit(model.get("pricePerInputImage"))
    per_second_input = _xai_price_per_unit(model.get("pricePerInputVideoSecond"))
    if not output_prices and input_per_image is None and per_second_input is None:
        return None

    pricing: dict[str, Any] = {"currency": "USD"}
    if output_prices:
        pricing["per_second_output"] = min(output_prices)
    if input_per_image is not None:
        pricing["input_per_image"] = input_per_image
    if per_second_input is not None:
        pricing["per_second_input"] = per_second_input
    return pricing


def _display_name_from_xai_model_id(model_id: str) -> str:
    parts = model_id.split("-")
    display_parts: list[str] = []
    index = 0
    while index < len(parts):
        current = parts[index]
        if current.isdigit() and index + 1 < len(parts) and parts[index + 1].isdigit():
            display_parts.append(f"{current}.{parts[index + 1]}")
            index += 2
            continue

        if current.startswith("grok"):
            display_parts.append(current.replace("grok", "Grok", 1))
        elif current.replace(".", "", 1).isdigit():
            display_parts.append(current)
        else:
            display_parts.append(current.title())
        index += 1

    return " ".join(display_parts)


def _canonical_model_input(raw_model_id: str, aliases: Sequence[str]) -> dict[str, Any]:
    return {
        "id": raw_model_id,
        "match": {
            "or": [{"equals": alias} for alias in aliases if isinstance(alias, str) and alias]
        },
    }


def _build_xai_model_records(
    raw_model_id: str,
    aliases: Sequence[str],
    fields: dict[str, Any],
    evidence_ref: str,
    rejection_policy: Mapping[str, Sequence[str]] | None,
) -> list[SourceEvidence]:
    canonical_model_id = _choose_canonical_model_id(_canonical_model_input(raw_model_id, aliases))
    records = [
        SourceEvidence(
            source_name="xai_official_docs",
            source_model_id=canonical_model_id,
            provider_slug="xai",
            canonical_hint=canonical_model_id,
            fields=fields,
            confidence="official",
            evidence_ref=evidence_ref,
            rejected=is_rejected_model_id(canonical_model_id, rejection_policy),
        )
    ]

    provider_aliases = []
    if raw_model_id != canonical_model_id:
        provider_aliases.append(raw_model_id)
    provider_aliases.extend(
        alias
        for alias in aliases
        if isinstance(alias, str) and alias and alias != canonical_model_id and alias != raw_model_id
    )

    seen: set[str] = set()
    for alias in provider_aliases:
        if alias in seen:
            continue
        seen.add(alias)
        records.append(
            SourceEvidence(
                source_name="xai_official_docs",
                source_model_id=f"xai/{alias}",
                provider_slug="xai",
                canonical_hint=canonical_model_id,
                fields=fields,
                confidence="official",
                evidence_ref=evidence_ref,
                rejected=is_rejected_model_id(alias, rejection_policy),
            )
        )

    return records


def normalize_xai_models_official_rows(
    payload: Mapping[str, Any],
    *,
    rejection_policy: Mapping[str, Sequence[str]] | None = None,
    **_: Any,
) -> list[SourceEvidence]:
    evidence_ref = payload.get("source_url") or "xai_models_official.json"
    records: list[SourceEvidence] = []

    for model in payload.get("language_models", []):
        if not isinstance(model, Mapping):
            continue
        raw_model_id = model.get("name")
        if not isinstance(raw_model_id, str) or not raw_model_id:
            continue

        fields: dict[str, Any] = {
            "display_name": _display_name_from_xai_model_id(_choose_canonical_model_id(_canonical_model_input(raw_model_id, model.get("aliases", [])))),
            "owned_by": "xai",
            "modes": ["chat"],
            "source_url": evidence_ref,
        }
        context_window = _parse_xai_integer(model.get("maxPromptLength"))
        if context_window is not None:
            fields["context_window"] = context_window
        pricing = _pricing_from_xai_language_model(model)
        if pricing is not None:
            fields["pricing"] = pricing
        _apply_inferred_modalities(fields)

        records.extend(
            _build_xai_model_records(
                raw_model_id,
                model.get("aliases", []),
                fields,
                evidence_ref,
                rejection_policy,
            )
        )

    for model in payload.get("image_generation_models", []):
        if not isinstance(model, Mapping):
            continue
        raw_model_id = model.get("name")
        if not isinstance(raw_model_id, str) or not raw_model_id:
            continue

        fields: dict[str, Any] = {
            "display_name": _display_name_from_xai_model_id(raw_model_id),
            "owned_by": "xai",
            "modes": ["image_generation"],
            "source_url": evidence_ref,
        }
        pricing = _pricing_from_xai_image_model(model)
        if pricing is not None:
            fields["pricing"] = pricing
        _apply_inferred_modalities(fields)

        records.extend(
            _build_xai_model_records(
                raw_model_id,
                model.get("aliases", []),
                fields,
                evidence_ref,
                rejection_policy,
            )
        )

    for model in payload.get("video_generation_models", []):
        if not isinstance(model, Mapping):
            continue
        raw_model_id = model.get("name")
        if not isinstance(raw_model_id, str) or not raw_model_id:
            continue

        fields: dict[str, Any] = {
            "display_name": _display_name_from_xai_model_id(raw_model_id),
            "owned_by": "xai",
            "modes": ["video_generation"],
            "source_url": evidence_ref,
        }
        pricing = _pricing_from_xai_video_model(model)
        if pricing is not None:
            fields["pricing"] = pricing
        _apply_inferred_modalities(fields)

        records.extend(
            _build_xai_model_records(
                raw_model_id,
                model.get("aliases", []),
                fields,
                evidence_ref,
                rejection_policy,
            )
        )

    return records


def extract_supported_fields(entry: Mapping[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    endpoints: list[str] = []
    supported_endpoints = entry.get("supported_endpoints")
    if isinstance(supported_endpoints, Sequence) and not isinstance(supported_endpoints, (str, bytes, bytearray)):
        endpoints = [endpoint for endpoint in supported_endpoints if isinstance(endpoint, str) and endpoint]
        if endpoints:
            fields["endpoints"] = endpoints

    display_name = entry.get("name")
    if isinstance(display_name, str) and display_name:
        fields["display_name"] = display_name
    elif isinstance(entry.get("id"), str) and entry.get("id"):
        fields["display_name"] = _display_name_from_model_id(entry["id"])

    inferred_modes: list[str] = []
    mode = entry.get("mode")
    if isinstance(mode, str) and mode:
        inferred_modes.append(mode)
    elif isinstance(entry.get("architecture"), Mapping):
        architecture_modality = entry["architecture"].get("modality")
        if isinstance(architecture_modality, str) and architecture_modality:
            inferred_modes.append(_infer_mode_from_values(architecture_modality))
    inferred_modes.extend(_modes_from_endpoints(endpoints))
    if inferred_modes:
        fields["modes"] = _ordered_unique_strings(inferred_modes)

    source_url = entry.get("source")
    if isinstance(source_url, str) and source_url:
        fields["source_url"] = source_url

    top_provider = entry.get("top_provider")
    if isinstance(top_provider, Mapping):
        context_window = top_provider.get("context_length")
        if context_window is not None:
            fields["context_window"] = context_window

        max_output_tokens = top_provider.get("max_completion_tokens")
        if max_output_tokens is not None:
            fields["max_output_tokens"] = max_output_tokens
    elif entry.get("max_input_tokens") is not None:
        fields["context_window"] = entry.get("max_input_tokens")

    if entry.get("max_input_tokens") is not None and "context_window" not in fields:
        fields["context_window"] = entry.get("max_input_tokens")

    max_output_tokens = entry.get("max_output_tokens")
    if max_output_tokens is None:
        max_output_tokens = entry.get("max_tokens")
    if max_output_tokens is not None:
        fields["max_output_tokens"] = max_output_tokens

    output_vector_size = entry.get("output_vector_size")
    if output_vector_size is not None:
        fields["output_vector_size"] = output_vector_size

    max_images_per_request = entry.get("max_images_per_prompt")
    if max_images_per_request is not None:
        fields["max_images_per_request"] = max_images_per_request

    max_audio_per_request = entry.get("max_audio_per_prompt")
    if max_audio_per_request is not None:
        fields["max_audio_per_request"] = max_audio_per_request

    max_videos_per_request = entry.get("max_videos_per_prompt")
    if max_videos_per_request is not None:
        fields["max_videos_per_request"] = max_videos_per_request

    max_pdf_size_mb = entry.get("max_pdf_size_mb")
    if max_pdf_size_mb is not None:
        fields["max_pdf_size_mb"] = max_pdf_size_mb

    max_audio_length_seconds = _duration_seconds_from_hours(entry.get("max_audio_length_hours"))
    if max_audio_length_seconds is not None:
        fields["max_audio_length_seconds"] = max_audio_length_seconds

    pricing = _merge_pricing_maps(
        _pricing_from_usd_per_token(entry.get("input_cost_per_token"), entry.get("output_cost_per_token")),
        _pricing_from_usd_per_token(
            entry.get("pricing", {}).get("prompt"),
            entry.get("pricing", {}).get("completion"),
        ),
        None if _is_image_generation_summary_row(entry) else _pricing_from_usd_per_mtok(entry.get("input"), entry.get("output")),
        _pricing_from_portkey(entry),
        _pricing_from_extended_fields(entry),
    )
    if pricing is not None:
        tiers = _build_tiered_token_pricing(entry, pricing)
        if tiers:
            pricing["tiers"] = tiers
        fields["pricing"] = pricing

    supported_modalities = entry.get("supported_modalities")
    supported_output_modalities = entry.get("supported_output_modalities")
    architecture = entry.get("architecture")
    input_modalities = supported_modalities
    output_modalities = supported_output_modalities
    if isinstance(architecture, Mapping):
        input_modalities = input_modalities or architecture.get("input_modalities")
        output_modalities = output_modalities or architecture.get("output_modalities")

    modalities: dict[str, list[str]] = {}
    normalized_input_modalities = [
        modality
        for modality in (input_modalities or [])
        if modality in SUPPORTED_MODALITIES
    ]
    normalized_output_modalities = [
        modality
        for modality in (output_modalities or [])
        if modality in SUPPORTED_MODALITIES
    ]
    if normalized_input_modalities:
        modalities["input"] = normalized_input_modalities
    if normalized_output_modalities:
        modalities["output"] = normalized_output_modalities
    if modalities:
        fields["modalities"] = modalities
    _apply_inferred_modalities(fields)

    capabilities: dict[str, bool] = {}
    capability_flags = {
        "supports_function_calling": "function_calling",
        "supports_parallel_function_calling": "parallel_function_calling",
        "supports_system_messages": "system_messages",
        "supports_vision": "vision",
        "supports_audio_input": "audio_input",
        "supports_audio_output": "audio_output",
        "supports_video_input": "video_input",
        "supports_pdf_input": "pdf_input",
        "supports_json_mode": "json_mode",
        "supports_structured_output": "structured_output",
        "supports_response_schema": "response_schema",
        "supports_reasoning": "reasoning",
        "supports_prompt_caching": "prompt_caching",
        "supports_web_search": "web_search",
        "supports_computer_use": "computer_use",
        "supports_tool_choice": "tool_choice",
    }
    for source_field, target_field in capability_flags.items():
        if entry.get(source_field) is True:
            capabilities[target_field] = True

    supported_parameters = entry.get("supported_parameters")
    if isinstance(supported_parameters, Sequence) and not isinstance(supported_parameters, (str, bytes, bytearray)):
        parameter_set = {value for value in supported_parameters if isinstance(value, str)}
        if {"tools", "tool_choice"} & parameter_set:
            capabilities.setdefault("function_calling", True)
        if "tool_choice" in parameter_set:
            capabilities.setdefault("tool_choice", True)
        if {"structured_outputs", "response_format"} & parameter_set:
            capabilities.setdefault("structured_output", True)
            capabilities.setdefault("response_schema", True)
        if {"reasoning", "reasoning_effort", "include_reasoning"} & parameter_set:
            capabilities.setdefault("reasoning", True)
    if capabilities:
        fields["capabilities"] = capabilities

    rate_limits: dict[str, Any] = {}
    for source_field in ("rpm", "tpm"):
        value = entry.get(source_field)
        if value is not None:
            rate_limits[source_field] = value
    if rate_limits:
        fields["rate_limits"] = rate_limits

    return fields


def normalize_litellm_entry(
    entry: dict[str, Any],
    *,
    rejection_policy: Mapping[str, Sequence[str]] | None = None,
    evidence_ref: str = "litellm_model_prices.json",
) -> SourceEvidence:
    model_name = entry["model_name"]
    provider_slug, canonical_hint = split_provider_model_name(model_name, entry.get("litellm_provider"))
    normalized_entry = dict(entry)
    normalized_entry.setdefault("id", canonical_hint or model_name)
    return SourceEvidence(
        source_name="litellm",
        source_model_id=model_name,
        provider_slug=provider_slug,
        canonical_hint=canonical_hint,
        fields=extract_supported_fields(normalized_entry),
        confidence="low",
        evidence_ref=entry.get("source") or evidence_ref,
        rejected=is_rejected_model_id(model_name, rejection_policy),
    )


def normalize_litellm_rows(
    rows: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    rejection_policy: Mapping[str, Sequence[str]] | None = None,
    evidence_ref: str = "litellm_model_prices.json",
) -> list[SourceEvidence]:
    if isinstance(rows, Mapping):
        entries = []
        for model_name, payload in rows.items():
            entry = dict(payload)
            entry.setdefault("model_name", model_name)
            entries.append(
                normalize_litellm_entry(
                    entry,
                    rejection_policy=rejection_policy,
                    evidence_ref=evidence_ref,
                )
            )
        return entries

    return [
        normalize_litellm_entry(
            dict(entry),
            rejection_policy=rejection_policy,
            evidence_ref=evidence_ref,
        )
        for entry in rows
    ]


def normalize_openrouter_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    rejection_policy: Mapping[str, Sequence[str]] | None = None,
    evidence_ref: str = "openrouter_models.json",
) -> list[SourceEvidence]:
    records: list[SourceEvidence] = []
    for row in rows:
        source_model_id = row["id"]
        _, source_model_hint = split_provider_model_name(source_model_id)
        _, canonical_slug_hint = split_provider_model_name(row.get("canonical_slug") or source_model_id)
        raw_model_id = source_model_hint or source_model_id
        canonical_hint = _choose_preferred_canonical_hint(raw_model_id, source_model_hint, canonical_slug_hint)
        records.append(
            SourceEvidence(
                source_name="openrouter",
                source_model_id=source_model_id,
                provider_slug="openrouter",
                canonical_hint=canonical_hint,
                fields=extract_supported_fields(row),
                confidence="low",
                evidence_ref=evidence_ref,
                rejected=is_rejected_model_id(source_model_id, rejection_policy),
            )
        )
    return records


def normalize_llm_prices_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    rejection_policy: Mapping[str, Sequence[str]] | None = None,
    evidence_ref: str = "llm_prices_current.json",
) -> list[SourceEvidence]:
    records: list[SourceEvidence] = []
    for row in rows:
        source_model_id = row["id"]
        provider_slug, canonical_hint = split_provider_model_name(source_model_id, row.get("vendor"))
        records.append(
            SourceEvidence(
                source_name="llm_prices",
                source_model_id=source_model_id,
                provider_slug=provider_slug,
                canonical_hint=canonical_hint,
                fields=extract_supported_fields(row),
                confidence="low",
                evidence_ref=evidence_ref,
                rejected=is_rejected_model_id(source_model_id, rejection_policy),
            )
        )
    return records


def normalize_portkey_files(
    files: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    rejection_policy: Mapping[str, Sequence[str]] | None = None,
    evidence_ref_by_file: Mapping[str, str] | None = None,
) -> list[SourceEvidence]:
    records: list[SourceEvidence] = []
    source_refs = evidence_ref_by_file or {}
    for filename, models in files.items():
        provider_slug = normalize_provider_slug(Path(filename).stem)
        evidence_ref = source_refs.get(filename, f"portkey/{Path(filename).name}")
        for model_id, payload in models.items():
            if model_id == "default":
                continue

            _, canonical_hint = split_provider_model_name(model_id, provider_slug)
            normalized_payload = dict(payload)
            normalized_payload.setdefault("id", canonical_hint or model_id)
            records.append(
                SourceEvidence(
                    source_name="portkey",
                    source_model_id=model_id,
                    provider_slug=provider_slug,
                    canonical_hint=canonical_hint,
                    fields=extract_supported_fields(normalized_payload),
                    confidence="low",
                    evidence_ref=evidence_ref,
                    rejected=is_rejected_model_id(model_id, rejection_policy),
                )
            )
    return records


def normalize_pydantic_genai_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    rejection_policy: Mapping[str, Sequence[str]] | None = None,
    allowed_providers: Sequence[str] | None = None,
    owner_providers: Sequence[str] | None = None,
    skip_providers: Sequence[str] | None = None,
    evidence_ref: str = "pydantic_genai_prices.json",
) -> list[SourceEvidence]:
    allowed_provider_set = {normalize_provider_slug(value) for value in allowed_providers or ()}
    owner_provider_set = {normalize_provider_slug(value) for value in owner_providers or ()}
    skipped_provider_set = {normalize_provider_slug(value) for value in skip_providers or ()}
    records: list[SourceEvidence] = []

    for provider in rows:
        provider_slug = normalize_provider_slug(provider.get("id"))
        if allowed_provider_set and provider_slug not in allowed_provider_set:
            continue
        if provider_slug in skipped_provider_set:
            continue

        provider_evidence_ref = next(
            (
                url
                for url in provider.get("pricing_urls", [])
                if isinstance(url, str) and url
            ),
            evidence_ref,
        )

        for model in provider.get("models", []):
            raw_model_id = model.get("id")
            if not isinstance(raw_model_id, str) or not raw_model_id:
                continue

            canonical_model_id = sanitize_provider_canonical_hint(_choose_canonical_model_id(model), provider_slug)
            fields = extract_official_catalog_fields(
                model,
                provider_slug,
                canonical_model_id=canonical_model_id,
                owner_providers=owner_provider_set or None,
            )
            if isinstance(provider_evidence_ref, str) and provider_evidence_ref:
                fields.setdefault("source_url", provider_evidence_ref)
            records.append(
                SourceEvidence(
                    source_name="official",
                    source_model_id=canonical_model_id,
                    provider_slug=provider_slug,
                    canonical_hint=canonical_model_id,
                    fields=fields,
                    confidence="official",
                    evidence_ref=provider_evidence_ref,
                    rejected=is_rejected_model_id(canonical_model_id, rejection_policy),
                )
            )

            provider_aliases = {
                alias
                for alias in [raw_model_id, *_extract_exact_match_ids(model.get("match"))]
                if isinstance(alias, str) and alias and alias != canonical_model_id
            }
            for provider_alias in sorted(provider_aliases):
                records.append(
                    SourceEvidence(
                        source_name="official",
                        source_model_id=f"{provider_slug}/{provider_alias}",
                        provider_slug=provider_slug,
                        canonical_hint=canonical_model_id,
                        fields=fields,
                        confidence="official",
                        evidence_ref=provider_evidence_ref,
                        rejected=is_rejected_model_id(provider_alias, rejection_policy),
                    )
                )

    return records


NORMALIZER_BY_SOURCE = {
    "litellm": normalize_litellm_rows,
    "xai_models_official": normalize_xai_models_official_rows,
    "pydantic_genai": normalize_pydantic_genai_rows,
    "deepseek_official": normalize_pydantic_genai_rows,
    "runway_official": normalize_pydantic_genai_rows,
    "google_speech_official": normalize_pydantic_genai_rows,
    "openrouter": normalize_openrouter_rows,
    "llm_prices": normalize_llm_prices_rows,
    "portkey": normalize_portkey_files,
}
