"""Shared resolver rules for canonical model admission and field authority."""

from __future__ import annotations

import re
from typing import Any

from pipeline.types import SourceEvidence


DEFAULT_FIELD_AUTHORITY = {
    "owned_by": ["official"],
    "display_name": ["official"],
    "pricing": ["official", "portkey", "llm_prices", "openrouter", "litellm"],
}

CONFIDENCE_RANK = {
    "official": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}

REGION_PATTERN = re.compile(
    r"(?:^|[-_.])(?:"
    r"(?:us|eu|ap|sa|ca|me|af)-(?:north|south|east|west|central|southeast|northeast|northwest|southwest)\d*"
    r"|eastus\d*|westus\d*|centralus|northcentralus|southcentralus"
    r"|westeurope|northeurope|uksouth|ukwest"
    r")(?:$|[-_.])"
)

ARTIFACT_NAMES = {"default", "sample_spec", "search_api", "model_router"}
ARTIFACT_SUFFIXES = (".json", ".yaml", ".yml", ".txt", ".csv")


def sort_candidates_by_authority(
    field_name: str,
    candidates: list[SourceEvidence],
    policy: dict[str, Any] | None,
) -> list[SourceEvidence]:
    configured_authority = (policy or {}).get("field_authority", {})
    preferred = configured_authority.get(field_name, DEFAULT_FIELD_AUTHORITY.get(field_name, ["official"]))

    def candidate_rank(candidate: SourceEvidence) -> tuple[int, int, int, int, str, str]:
        authority_tokens = (candidate.source_name, candidate.confidence)
        preferred_rank = next(
            (index for index, token in enumerate(preferred) if token in authority_tokens),
            len(preferred),
        )
        missing_rank = 0 if field_name in candidate.fields else 1
        confidence_rank = CONFIDENCE_RANK.get(candidate.confidence, len(CONFIDENCE_RANK))
        canonical_rank = 0 if candidate.source_model_id == (candidate.canonical_hint or candidate.source_model_id) else 1
        return (
            missing_rank,
            preferred_rank,
            confidence_rank,
            canonical_rank,
            candidate.source_name,
            candidate.source_model_id,
        )

    return sorted(candidates, key=candidate_rank)


def looks_like_region_scoped_deployment(model_id: str) -> bool:
    lowered = model_id.lower()
    return bool(REGION_PATTERN.search(lowered))


def contains_source_artifact_shape(model_id: str) -> bool:
    lowered = model_id.lower()
    return (
        lowered in ARTIFACT_NAMES
        or lowered.endswith(ARTIFACT_SUFFIXES)
        or lowered.startswith(("http://", "https://"))
        or "\\" in model_id
        or model_id.endswith("/")
        or any(char.isspace() for char in model_id)
    )


def is_canonical_model_key(model_id: str) -> bool:
    return (
        "/" not in model_id
        and ":" not in model_id
        and not looks_like_region_scoped_deployment(model_id)
        and not contains_source_artifact_shape(model_id)
    )
