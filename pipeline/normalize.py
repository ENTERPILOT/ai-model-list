"""Normalize raw source payloads into shared evidence records."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pipeline.types import SourceEvidence


USD_PER_TOKEN_TO_USD_PER_MTOK = 1_000_000
CENTS_PER_TOKEN_TO_USD_PER_MTOK = 10_000


def normalize_provider_slug(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = value.strip().lower()
    return normalized or None


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
    payg = entry.get("pricing_config", {}).get("pay_as_you_go", {})
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


NORMALIZER_BY_SOURCE = {
    "litellm": normalize_litellm_rows,
    "openrouter": normalize_openrouter_rows,
    "llm_prices": normalize_llm_prices_rows,
    "portkey": normalize_portkey_files,
}
