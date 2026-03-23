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
    canonical_clusters: dict[str, list[SourceEvidence]] = defaultdict(list)
    provider_clusters: dict[str, list[SourceEvidence]] = defaultdict(list)

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

        if is_canonical_evidence_for_key(record, canonical_key):
            canonical_clusters[canonical_key].append(record)

        if is_provider_scoped_id(record.source_model_id):
            provider_clusters[record.source_model_id].append(record)

    admitted_models: set[str] = set()

    for canonical_key, records in evidence_clusters.items():
        canonical_records = canonical_clusters.get(canonical_key, [])
        if not should_admit_canonical_model(canonical_key, canonical_records):
            quarantine_records(report["quarantine"], records)
            continue

        registry["models"][canonical_key] = build_model_record(canonical_records, policy)
        admitted_models.add(canonical_key)

    for provider_model_id, records in provider_clusters.items():
        canonical_key = resolve_canonical_key(records[0], alias_map)
        if canonical_key not in admitted_models:
            continue

        registry["provider_models"][provider_model_id] = build_provider_model_record(
            provider_model_id,
            canonical_key,
            records,
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
    provider_model_id: str,
    canonical_key: str,
    records: list[SourceEvidence],
    policy: dict[str, Any],
) -> dict[str, Any]:
    provider_model = {
        "model_ref": canonical_key,
        "enabled": True,
    }

    for field_name in sorted(_collect_field_names(records) & PROVIDER_MODEL_FIELD_NAMES):
        value = choose_field_value(field_name, records, policy)
        if value is not None:
            provider_model[field_name] = value
    return provider_model


def should_admit_canonical_model(canonical_key: str, records: list[SourceEvidence]) -> bool:
    return bool(records) and is_canonical_model_key(canonical_key) and any(
        record.confidence == "official" for record in records
    )


def is_canonical_evidence_for_key(record: SourceEvidence, canonical_key: str) -> bool:
    return record.source_model_id == canonical_key and is_canonical_model_key(record.source_model_id)


def is_provider_scoped_id(model_id: str) -> bool:
    return "/" in model_id or ":" in model_id


def resolve_canonical_key(record: SourceEvidence, alias_map: dict[str, Any]) -> str:
    candidates = [
        record.canonical_hint,
        record.source_model_id,
    ]
    for candidate in candidates:
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
