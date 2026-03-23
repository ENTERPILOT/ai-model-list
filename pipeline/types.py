from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SourceEvidence:
    source_name: str
    source_model_id: str
    provider_slug: str | None
    canonical_hint: str | None
    fields: dict[str, Any]
    confidence: str
    evidence_ref: str
    rejected: bool = False
