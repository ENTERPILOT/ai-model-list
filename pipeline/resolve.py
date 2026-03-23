"""Resolve normalized evidence into canonical models, provider models, and quarantine."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Iterable

from pipeline.rules import is_canonical_model_key, sort_candidates_by_authority
from pipeline.types import SourceEvidence


MODEL_FIELD_NAMES = {
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

PROVIDER_MODEL_FIELD_NAMES = {
    "pricing",
    "context_window",
    "max_output_tokens",
    "rate_limits",
    "endpoints",
    "regions",
}


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

        canonical_model = build_model_record(canonical_records, policy)
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


def build_model_record(records: list[SourceEvidence], policy: dict[str, Any]) -> dict[str, Any]:
    model: dict[str, Any] = {}
    for field_name in sorted(_collect_field_names(records) & MODEL_FIELD_NAMES):
        value = choose_field_value(field_name, records, policy)
        if value is not None:
            model[field_name] = value
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
    return bool(records) and is_canonical_model_key(canonical_key) and any(
        record.confidence == "official" for record in records
    )


def has_required_model_fields(model: dict[str, Any]) -> bool:
    display_name = model.get("display_name")
    modes = model.get("modes")
    return bool(display_name) and isinstance(modes, list) and bool(modes)


def select_canonical_records(
    canonical_key: str,
    records: list[SourceEvidence],
    alias_map: dict[str, Any],
) -> list[SourceEvidence]:
    exact_records = [record for record in records if is_exact_canonical_record(record, canonical_key, alias_map)]
    if exact_records:
        return exact_records
    return [record for record in records if is_clean_alias_record(record, canonical_key, alias_map)]


def build_provider_clusters(
    canonical_key: str,
    records: list[SourceEvidence],
    canonical_records: list[SourceEvidence],
) -> dict[str, list[SourceEvidence]]:
    provider_clusters: dict[str, list[SourceEvidence]] = defaultdict(list)
    has_exact_canonical = any(record.source_model_id == canonical_key for record in canonical_records)

    for record in records:
        provider_model_key = build_provider_model_key(record, canonical_key, has_exact_canonical, canonical_records)
        if provider_model_key is not None:
            provider_clusters[provider_model_key].append(record)

    return provider_clusters


def is_exact_canonical_record(
    record: SourceEvidence,
    canonical_key: str,
    alias_map: dict[str, Any],
) -> bool:
    return (
        record.source_model_id == canonical_key
        and is_canonical_model_key(record.source_model_id)
        and not has_unapproved_clean_alias_hint(record, alias_map)
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


def has_unapproved_clean_alias_hint(record: SourceEvidence, alias_map: dict[str, Any]) -> bool:
    return (
        is_canonical_model_key(record.source_model_id)
        and record.canonical_hint not in (None, "", record.source_model_id)
        and lookup_alias(record.source_model_id, alias_map) is None
    )


def is_provider_deployment_record(record: SourceEvidence) -> bool:
    return not is_canonical_model_key(record.source_model_id)


def build_provider_model_key(
    record: SourceEvidence,
    canonical_key: str,
    has_exact_canonical: bool,
    canonical_records: list[SourceEvidence],
) -> str | None:
    if record.provider_slug is None:
        return None
    if record in canonical_records:
        return None
    if is_provider_deployment_record(record):
        return f"{record.provider_slug}/{canonical_key}"
    if has_exact_canonical and is_canonical_model_key(record.source_model_id):
        return f"{record.provider_slug}/{canonical_key}"
    return None


def choose_provider_model_id(
    provider_model_key: str,
    canonical_key: str,
    records: list[SourceEvidence],
) -> str | None:
    source_ids = {
        record.source_model_id
        for record in records
        if record.source_model_id != canonical_key and record.source_model_id != provider_model_key
    }
    if len(source_ids) == 1:
        return next(iter(source_ids))
    return None


def resolve_canonical_key(record: SourceEvidence, alias_map: dict[str, Any]) -> str:
    approved_source_alias = lookup_alias(record.source_model_id, alias_map)
    if is_canonical_model_key(record.source_model_id):
        return approved_source_alias or record.source_model_id

    for candidate in (record.canonical_hint, record.source_model_id):
        if not candidate:
            continue

        resolved = lookup_alias(candidate, alias_map)
        if resolved:
            return resolved

    return record.canonical_hint or record.source_model_id


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
