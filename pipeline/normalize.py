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
    "pydantic_genai": normalize_pydantic_genai_rows,
    "openrouter": normalize_openrouter_rows,
    "llm_prices": normalize_llm_prices_rows,
    "portkey": normalize_portkey_files,
}
