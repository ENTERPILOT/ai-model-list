from pipeline.resolve import resolve_registry
from pipeline.types import SourceEvidence


def test_resolve_routes_provider_specific_ids_to_provider_models() -> None:
    evidence = [
        SourceEvidence(
            source_name="official",
            source_model_id="grok-4",
            provider_slug="xai",
            canonical_hint="grok-4",
            fields={"display_name": "Grok 4", "owned_by": "xai", "modes": ["chat"]},
            confidence="official",
            evidence_ref="https://docs.x.ai/docs/models",
        ),
        SourceEvidence(
            source_name="litellm",
            source_model_id="xai/grok-4",
            provider_slug="xai",
            canonical_hint="grok-4",
            fields={
                "pricing": {
                    "currency": "USD",
                    "input_per_mtok": 3.0,
                    "output_per_mtok": 15.0,
                }
            },
            confidence="low",
            evidence_ref="litellm_model_prices.json",
        ),
    ]

    registry, report = resolve_registry(evidence, curated={})

    assert "grok-4" in registry["models"]
    assert "xai/grok-4" in registry["provider_models"]
    assert not report["quarantine"]


def test_resolve_quarantines_unapproved_alias_cluster() -> None:
    evidence = [
        SourceEvidence(
            source_name="litellm",
            source_model_id="claude-opus-41",
            provider_slug="anthropic",
            canonical_hint="claude-opus-41",
            fields={"modes": ["chat"]},
            confidence="low",
            evidence_ref="litellm_model_prices.json",
        )
    ]

    registry, report = resolve_registry(evidence, curated={})

    assert "claude-opus-41" not in registry["models"]
    assert report["quarantine"]
