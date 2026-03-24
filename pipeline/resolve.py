"""Resolve normalized evidence into canonical models, provider models, and quarantine."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import re
from typing import Any, Iterable

from pipeline.rules import is_canonical_model_key, sort_candidates_by_authority
from pipeline.types import SourceEvidence


MODEL_FIELD_NAMES = {
    "aliases",
    "display_name",
    "description",
    "owned_by",
    "family",
    "release_date",
    "deprecation_date",
    "source_url",
    "tags",
    "modes",
    "modalities",
    "capabilities",
    "context_window",
    "max_output_tokens",
    "max_images_per_request",
    "max_audio_length_seconds",
    "max_video_length_seconds",
    "max_pdf_size_mb",
    "max_videos_per_request",
    "max_audio_per_request",
    "output_vector_size",
    "pricing",
    "parameters",
    "rankings",
}

REQUIRED_MODEL_FIELDS = {"display_name", "modes"}
FALLBACK_MODEL_FIELD_NAMES = {
    "display_name",
    "description",
    "owned_by",
    "family",
    "release_date",
    "deprecation_date",
    "source_url",
    "tags",
    "modes",
    "modalities",
    "capabilities",
    "output_vector_size",
}
DISPLAY_NAME_FALLBACK_SUPPORT_FIELDS = {
    "pricing",
    "context_window",
    "max_output_tokens",
    "output_vector_size",
    "modalities",
    "capabilities",
    "description",
    "source_url",
}

PROVIDER_MODEL_FIELD_NAMES = {
    "pricing",
    "context_window",
    "max_output_tokens",
    "rate_limits",
    "endpoints",
    "regions",
}
DISPLAY_NAME_SPLIT_PATTERN = re.compile(r"[-_./:]+")
DISPLAY_NAME_ACRONYMS = {"ai", "api", "asr", "gpt", "json", "ocr", "oss", "pdf", "tts", "ui", "ux"}


def choose_field_value(field_name: str, candidates: list[SourceEvidence], policy: dict[str, Any]) -> Any:
    ranked = sort_candidates_by_authority(field_name, candidates, policy)
    return ranked[0].fields.get(field_name) if ranked else None


def resolve_registry(
    evidence: Iterable[SourceEvidence],
    curated: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    registry = {
        "providers": deepcopy(curated.get("providers", {})),
        "models": {},
        "provider_models": {},
    }
    report = {"quarantine": []}

    alias_map = curated.get("canonical_aliases", {})
    policy = curated.get("source_policies", {})

    evidence_clusters: dict[str, list[SourceEvidence]] = defaultdict(list)

    for record in evidence:
        if record.rejected:
            report["quarantine"].append(
                {
                    "source_model_id": record.source_model_id,
                    "reason": "rejected",
                    "evidence_ref": record.evidence_ref,
                }
            )
            continue

        canonical_key = resolve_canonical_key(record, alias_map)
        evidence_clusters[canonical_key].append(record)

    for canonical_key, records in evidence_clusters.items():
        canonical_records = select_canonical_records(canonical_key, records, alias_map)
        if not should_admit_canonical_model(canonical_key, canonical_records):
            quarantine_records(report["quarantine"], records)
            continue

        canonical_model = build_model_record(canonical_key, canonical_records, records, policy)
        if not has_required_model_fields(canonical_model):
            quarantine_records(report["quarantine"], records)
            continue

        registry["models"][canonical_key] = canonical_model

        provider_clusters = build_provider_clusters(canonical_key, records, canonical_records)
        for provider_model_key, provider_records in provider_clusters.items():
            registry["provider_models"][provider_model_key] = build_provider_model_record(
                provider_model_key,
                canonical_key,
                provider_records,
                policy,
            )

    return registry, report


def build_model_record(
    canonical_key: str,
    canonical_records: list[SourceEvidence],
    support_records: list[SourceEvidence],
    policy: dict[str, Any],
) -> dict[str, Any]:
    model: dict[str, Any] = {}
    for field_name in sorted((_collect_field_names(canonical_records) & MODEL_FIELD_NAMES) - {"aliases"}):
        value = choose_field_value(field_name, canonical_records, policy)
        if value is not None:
            model[field_name] = value

    missing_fallback_fields = FALLBACK_MODEL_FIELD_NAMES - model.keys()
    for field_name in sorted((_collect_field_names(support_records) & missing_fallback_fields) - {"aliases"}):
        value = choose_field_value(field_name, support_records, policy)
        if value is not None:
            model[field_name] = value

    aliases = collect_model_aliases(canonical_key, support_records)
    if aliases:
        model["aliases"] = aliases

    if not model.get("display_name") and can_fallback_display_name(canonical_records, support_records):
        model["display_name"] = _display_name_from_model_id(canonical_key)
    return model


def build_provider_model_record(
    provider_model_key: str,
    canonical_key: str,
    records: list[SourceEvidence],
    policy: dict[str, Any],
) -> dict[str, Any]:
    provider_model = {
        "model_ref": canonical_key,
        "enabled": True,
    }
    retained_provider_model_id = choose_provider_model_id(provider_model_key, canonical_key, records)
    if retained_provider_model_id is not None:
        provider_model["provider_model_id"] = retained_provider_model_id

    for field_name in sorted(_collect_field_names(records) & PROVIDER_MODEL_FIELD_NAMES):
        value = choose_field_value(field_name, records, policy)
        if value is not None:
            provider_model[field_name] = value
    return provider_model


def should_admit_canonical_model(canonical_key: str, records: list[SourceEvidence]) -> bool:
    return bool(records) and is_canonical_model_key(canonical_key)


def has_required_model_fields(model: dict[str, Any]) -> bool:
    display_name = model.get("display_name")
    modes = model.get("modes")
    return bool(display_name) and isinstance(modes, list) and bool(modes)


def select_canonical_records(
    canonical_key: str,
    records: list[SourceEvidence],
    alias_map: dict[str, Any],
) -> list[SourceEvidence]:
    exact_records = [record for record in records if is_exact_canonical_record(record, canonical_key)]
    if exact_records:
        return exact_records
    return [record for record in records if is_clean_alias_record(record, canonical_key, alias_map)]


def build_provider_clusters(
    canonical_key: str,
    records: list[SourceEvidence],
    canonical_records: list[SourceEvidence],
) -> dict[str, list[SourceEvidence]]:
    provider_clusters: dict[str, list[SourceEvidence]] = defaultdict(list)
    aggregate_provider_slugs = {record.provider_slug for record in canonical_records if record.provider_slug is not None}

    for record in records:
        provider_model_key = build_provider_model_key(record)
        if provider_model_key is not None:
            provider_clusters[provider_model_key].append(record)

        if record.provider_slug in aggregate_provider_slugs:
            aggregate_key = f"{record.provider_slug}/{canonical_key}"
            if aggregate_key != provider_model_key:
                provider_clusters[aggregate_key].append(record)

    for provider_slug in aggregate_provider_slugs:
        provider_clusters.setdefault(f"{provider_slug}/{canonical_key}", [])

    return provider_clusters


def is_exact_canonical_record(
    record: SourceEvidence,
    canonical_key: str,
) -> bool:
    return (
        record.source_model_id == canonical_key
        and is_canonical_model_key(record.source_model_id)
        and not has_unapproved_clean_alias_hint(record)
    )


def is_clean_alias_record(
    record: SourceEvidence,
    canonical_key: str,
    alias_map: dict[str, Any],
) -> bool:
    approved_alias = lookup_alias(record.source_model_id, alias_map)
    return (
        record.source_model_id != canonical_key
        and is_canonical_model_key(record.source_model_id)
        and approved_alias == canonical_key
    )


def has_unapproved_clean_alias_hint(record: SourceEvidence) -> bool:
    return (
        is_canonical_model_key(record.source_model_id)
        and isinstance(record.canonical_hint, str)
        and is_canonical_model_key(record.canonical_hint)
        and record.canonical_hint != record.source_model_id
    )

def build_provider_model_key(
    record: SourceEvidence,
) -> str | None:
    if record.provider_slug is None:
        return None
    if record.source_model_id.startswith(f"{record.provider_slug}/"):
        return record.source_model_id
    return f"{record.provider_slug}/{record.source_model_id}"


def choose_provider_model_id(
    provider_model_key: str,
    canonical_key: str,
    records: list[SourceEvidence],
) -> str | None:
    provider_slug, _, _ = provider_model_key.partition("/")

    source_ids = {
        _strip_provider_prefix(record.source_model_id, provider_slug)
        for record in records
        if record.source_model_id != canonical_key and record.source_model_id != provider_model_key
    }
    if len(source_ids) == 1:
        provider_model_id = next(iter(source_ids))
        key_suffix = provider_model_key.split("/", 1)[1]
        if provider_model_id == key_suffix:
            return None
        return provider_model_id
    return None


def _strip_provider_prefix(source_model_id: str, provider_slug: str) -> str:
    prefix = f"{provider_slug}/"
    if source_model_id.startswith(prefix):
        return source_model_id[len(prefix):]
    return source_model_id


def resolve_canonical_key(record: SourceEvidence, alias_map: dict[str, Any]) -> str:
    approved_source_alias = canonicalize_model_id(record.source_model_id, alias_map)
    if is_canonical_model_key(record.source_model_id):
        return approved_source_alias or record.source_model_id

    for candidate in (record.canonical_hint, record.source_model_id):
        if not candidate:
            continue

        resolved = canonicalize_model_id(candidate, alias_map)
        if resolved:
            return resolved

    return record.canonical_hint or record.source_model_id


def canonicalize_model_id(model_id: str, alias_map: dict[str, Any]) -> str:
    return lookup_alias(model_id, alias_map) or model_id


def lookup_alias(model_id: str, alias_map: dict[str, Any]) -> str | None:
    resolved = alias_map.get(model_id)
    if isinstance(resolved, str):
        return resolved
    if isinstance(resolved, dict):
        for key in ("canonical", "canonical_key", "model"):
            value = resolved.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def quarantine_records(quarantine: list[dict[str, Any]], records: list[SourceEvidence]) -> None:
    for record in records:
        quarantine.append(
            {
                "source_model_id": record.source_model_id,
                "reason": "unapproved_alias_or_low_confidence_only",
                "evidence_ref": record.evidence_ref,
            }
        )


def _collect_field_names(records: list[SourceEvidence]) -> set[str]:
    field_names: set[str] = set()
    for record in records:
        field_names.update(record.fields)
    return field_names


def collect_model_aliases(canonical_key: str, records: list[SourceEvidence]) -> list[str]:
    aliases: set[str] = set()
    for record in records:
        for candidate in _alias_candidates(record):
            if candidate and candidate != canonical_key:
                aliases.add(candidate)
    return sorted(aliases, key=lambda value: (value.count("/"), len(value), value))


def _alias_candidates(record: SourceEvidence) -> set[str]:
    candidates = {record.source_model_id}
    if record.provider_slug is not None:
        if not record.source_model_id.startswith(f"{record.provider_slug}/"):
            candidates.add(f"{record.provider_slug}/{record.source_model_id}")
        stripped = _strip_provider_prefix(record.source_model_id, record.provider_slug)
        candidates.add(stripped)
    if record.canonical_hint:
        candidates.add(record.canonical_hint)
    return {candidate for candidate in candidates if candidate}


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
        elif token.replace(".", "", 1).isdigit():
            display_tokens.append(token)
        else:
            display_tokens.append(token.title())

    return " ".join(display_tokens) if display_tokens else model_id


def can_fallback_display_name(
    canonical_records: list[SourceEvidence],
    support_records: list[SourceEvidence],
) -> bool:
    candidate_records = canonical_records or support_records
    return any(DISPLAY_NAME_FALLBACK_SUPPORT_FIELDS & set(record.fields) for record in candidate_records)
