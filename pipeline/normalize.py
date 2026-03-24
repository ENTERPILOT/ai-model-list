"""Normalize raw source payloads into shared evidence records."""

from __future__ import annotations

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
}
UNSTABLE_ALIAS_TOKENS = ("latest", "beta", "preview", "alpha", "experimental", "free")
DATE_SUFFIX_PATTERN = re.compile(r"(?:[-_.])(?:\d{4}|\d{8})$")
XAI_TOKEN_PRICE_DIVISOR = 10_000
XAI_UNIT_PRICE_DIVISOR = 1_000_000_000


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
        return normalize_provider_slug(provider_slug), canonical_hint or None

    return normalize_provider_slug(fallback_provider), model_name or None


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
    return round(value * multiplier, 12)


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
    request_token = _to_float(payg.get("request_token", {}).get("price"))
    response_token = _to_float(payg.get("response_token", {}).get("price"))
    if request_token is None and response_token is None:
        return None

    pricing: dict[str, float | str] = {"currency": "USD"}
    if request_token is not None:
        pricing["input_per_mtok"] = _scale_price(request_token, CENTS_PER_TOKEN_TO_USD_PER_MTOK)
    if response_token is not None:
        pricing["output_per_mtok"] = _scale_price(response_token, CENTS_PER_TOKEN_TO_USD_PER_MTOK)
    return pricing


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
    }
    for source_field, target_field in field_map.items():
        value = _to_float(pricing_payload.get(source_field))
        if value is not None:
            pricing[target_field] = value

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


def extract_official_catalog_fields(model: Mapping[str, Any], provider_slug: str) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "modes": ["chat"],
        "owned_by": provider_slug,
    }

    display_name = model.get("name")
    if isinstance(display_name, str) and display_name:
        fields["display_name"] = display_name

    description = model.get("description")
    if isinstance(description, str) and description:
        fields["description"] = description

    context_window = model.get("context_window")
    if context_window is not None:
        fields["context_window"] = context_window

    pricing = _pricing_from_catalog_prices(model.get("prices"))
    if pricing is not None:
        fields["pricing"] = pricing

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
    return round(numeric / XAI_TOKEN_PRICE_DIVISOR, 12)


def _xai_price_per_unit(value: Any) -> float | None:
    numeric = _parse_xai_numeric(value)
    if numeric is None:
        return None
    return round(numeric / XAI_UNIT_PRICE_DIVISOR, 12)


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

    display_name = entry.get("name")
    if isinstance(display_name, str) and display_name:
        fields["display_name"] = display_name

    mode = entry.get("mode")
    if isinstance(mode, str) and mode:
        fields["modes"] = [mode]

    top_provider = entry.get("top_provider")
    if isinstance(top_provider, Mapping):
        context_window = top_provider.get("context_length")
        if context_window is not None:
            fields["context_window"] = context_window

        max_output_tokens = top_provider.get("max_completion_tokens")
        if max_output_tokens is not None:
            fields["max_output_tokens"] = max_output_tokens

    pricing = (
        _pricing_from_usd_per_token(entry.get("input_cost_per_token"), entry.get("output_cost_per_token"))
        or _pricing_from_usd_per_token(
            entry.get("pricing", {}).get("prompt"),
            entry.get("pricing", {}).get("completion"),
        )
        or _pricing_from_usd_per_mtok(entry.get("input"), entry.get("output"))
        or _pricing_from_portkey(entry)
    )
    if pricing is not None:
        fields["pricing"] = pricing

    return fields


def normalize_litellm_entry(
    entry: dict[str, Any],
    *,
    rejection_policy: Mapping[str, Sequence[str]] | None = None,
    evidence_ref: str = "litellm_model_prices.json",
) -> SourceEvidence:
    model_name = entry["model_name"]
    provider_slug, canonical_hint = split_provider_model_name(model_name, entry.get("litellm_provider"))
    return SourceEvidence(
        source_name="litellm",
        source_model_id=model_name,
        provider_slug=provider_slug,
        canonical_hint=canonical_hint,
        fields=extract_supported_fields(entry),
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
        provider_slug, canonical_hint = split_provider_model_name(row.get("canonical_slug") or source_model_id)
        records.append(
            SourceEvidence(
                source_name="openrouter",
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
            records.append(
                SourceEvidence(
                    source_name="portkey",
                    source_model_id=model_id,
                    provider_slug=provider_slug,
                    canonical_hint=canonical_hint,
                    fields=extract_supported_fields(payload),
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
    evidence_ref: str = "pydantic_genai_prices.json",
) -> list[SourceEvidence]:
    allowed_provider_set = {normalize_provider_slug(value) for value in allowed_providers or ()}
    records: list[SourceEvidence] = []

    for provider in rows:
        provider_slug = normalize_provider_slug(provider.get("id"))
        if provider_slug != "xai":
            continue
        if allowed_provider_set and provider_slug not in allowed_provider_set:
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

            canonical_model_id = _choose_canonical_model_id(model)
            fields = extract_official_catalog_fields(model, provider_slug)
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

            if raw_model_id == canonical_model_id:
                continue

            records.append(
                SourceEvidence(
                    source_name="official",
                    source_model_id=f"{provider_slug}/{raw_model_id}",
                    provider_slug=provider_slug,
                    canonical_hint=canonical_model_id,
                    fields=fields,
                    confidence="official",
                    evidence_ref=provider_evidence_ref,
                    rejected=is_rejected_model_id(raw_model_id, rejection_policy),
                )
            )

    return records


NORMALIZER_BY_SOURCE = {
    "litellm": normalize_litellm_rows,
    "xai_models_official": normalize_xai_models_official_rows,
    "pydantic_genai": normalize_pydantic_genai_rows,
    "openrouter": normalize_openrouter_rows,
    "llm_prices": normalize_llm_prices_rows,
    "portkey": normalize_portkey_files,
}
