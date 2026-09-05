"""Curated pricing overrides: correcting a wrong upstream price, and only that."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.loaders import load_curated_config
from pipeline.overrides import apply_pricing_overrides
from pipeline.report import build_markdown_report, build_report
from pipeline.resolve import resolve_registry
from pipeline.types import SourceEvidence


REPO_ROOT = Path(__file__).resolve().parent.parent


def _registry() -> dict:
    return {
        "models": {
            "gpt-5.6-luna": {
                "display_name": "GPT-5.6 Luna",
                "source_url": "https://developers.openai.com/api/docs/pricing",
                "pricing": {
                    "currency": "USD",
                    "input_per_mtok": 1.0,
                    "output_per_mtok": 6.0,
                    "batch_input_per_mtok": 0.1,
                    "tiers": [{"input_per_mtok": 0.2, "output_per_mtok": 1.2, "up_to_tokens": 272000}],
                },
            },
        },
        "provider_models": {
            "openai/gpt-5.6-luna": {
                "model_ref": "gpt-5.6-luna",
                "pricing": {"currency": "USD", "input_per_mtok": 1.0, "output_per_mtok": 6.0},
                "pricing_source_url": "https://llmprices.dev/",
                "source_urls": ["https://llmprices.dev/"],
            },
            "openai/gpt-5-6-luna": {
                "model_ref": "gpt-5.6-luna",
                "pricing": {"currency": "USD", "input_per_mtok": 1.0, "output_per_mtok": 6.0},
            },
            "openrouter/openai/gpt-5.6-luna": {
                "model_ref": "gpt-5.6-luna",
                "pricing": {"currency": "USD", "input_per_mtok": 0.25, "output_per_mtok": 1.5},
            },
        },
    }


def _override(**kwargs) -> dict:
    entry = {
        "model": "gpt-5.6-luna",
        "providers": ["openai"],
        "pricing": {"input_per_mtok": 0.2, "output_per_mtok": 1.2},
        "source_url": "https://developers.openai.com/api/docs/pricing",
    }
    entry.update(kwargs)
    return {"overrides": [entry]}


def test_override_corrects_canonical_and_named_provider_records():
    registry = _registry()

    findings = apply_pricing_overrides(registry, _override())

    assert registry["models"]["gpt-5.6-luna"]["pricing"]["input_per_mtok"] == 0.2
    assert registry["models"]["gpt-5.6-luna"]["pricing"]["output_per_mtok"] == 1.2
    assert registry["provider_models"]["openai/gpt-5.6-luna"]["pricing"]["input_per_mtok"] == 0.2
    assert registry["provider_models"]["openai/gpt-5-6-luna"]["pricing"]["input_per_mtok"] == 0.2
    assert findings[0]["status"] == "applied"
    assert findings[0]["changed_fields"] == ["input_per_mtok", "output_per_mtok"]


def test_override_leaves_unlisted_providers_on_their_own_published_rate():
    """A vendor's price says nothing about what a reseller charges for the model."""
    registry = _registry()

    apply_pricing_overrides(registry, _override())

    reseller = registry["provider_models"]["openrouter/openai/gpt-5.6-luna"]["pricing"]
    assert reseller["input_per_mtok"] == 0.25
    assert reseller["output_per_mtok"] == 1.5


def test_override_touches_only_the_fields_it_names():
    registry = _registry()

    apply_pricing_overrides(registry, _override())

    pricing = registry["models"]["gpt-5.6-luna"]["pricing"]
    assert pricing["batch_input_per_mtok"] == 0.1
    assert pricing["tiers"] == [{"input_per_mtok": 0.2, "output_per_mtok": 1.2, "up_to_tokens": 272000}]
    assert pricing["currency"] == "USD"


def test_override_repoints_pricing_provenance_at_the_reviewed_source():
    registry = _registry()

    apply_pricing_overrides(registry, _override())

    corrected = registry["provider_models"]["openai/gpt-5.6-luna"]
    assert corrected["pricing_source_url"] == "https://developers.openai.com/api/docs/pricing"
    assert "https://developers.openai.com/api/docs/pricing" in corrected["source_urls"]
    assert "https://llmprices.dev/" in corrected["source_urls"]


def test_override_that_changes_nothing_reports_itself_as_stale():
    """Upstream caught up — the entry is now dead weight and should be retired."""
    registry = _registry()
    registry["models"]["gpt-5.6-luna"]["pricing"]["input_per_mtok"] = 0.2
    registry["models"]["gpt-5.6-luna"]["pricing"]["output_per_mtok"] = 1.2
    for key in ("openai/gpt-5.6-luna", "openai/gpt-5-6-luna"):
        registry["provider_models"][key]["pricing"].update({"input_per_mtok": 0.2, "output_per_mtok": 1.2})

    findings = apply_pricing_overrides(registry, _override())

    assert findings[0]["status"] == "noop"
    assert findings[0]["changed_targets"] == []


def test_override_for_a_model_that_no_longer_resolves_is_reported_unmatched():
    findings = apply_pricing_overrides({"models": {}, "provider_models": {}}, _override())

    assert findings[0]["status"] == "unmatched"


def test_malformed_override_entries_are_skipped():
    registry = _registry()
    overrides = {"overrides": [{"model": "gpt-5.6-luna"}, {"pricing": {"input_per_mtok": 9.0}}, "nonsense"]}

    assert apply_pricing_overrides(registry, overrides) == []
    assert registry["models"]["gpt-5.6-luna"]["pricing"]["input_per_mtok"] == 1.0


def test_missing_override_config_is_a_no_op():
    registry = _registry()

    assert apply_pricing_overrides(registry, None) == []
    assert registry["models"]["gpt-5.6-luna"]["pricing"]["input_per_mtok"] == 1.0


def test_resolve_registry_applies_curated_overrides():
    evidence = [
        SourceEvidence(
            source_name="official",
            source_model_id="gpt-5.6-luna",
            provider_slug="openai",
            canonical_hint="gpt-5.6-luna",
            fields={
                "display_name": "GPT-5.6 Luna",
                "modes": ["chat"],
                "pricing": {"currency": "USD", "input_per_mtok": 1.0, "output_per_mtok": 6.0},
            },
            confidence="official",
            evidence_ref="pydantic_genai_prices.json",
        ),
    ]
    curated = {
        "providers": {"openai": {"display_name": "OpenAI", "api_type": "openai"}},
        "source_policies": {},
        "canonical_aliases": {},
        "rejections": {},
        "pricing_overrides": _override(),
    }

    registry, report = resolve_registry(evidence, curated)

    assert registry["models"]["gpt-5.6-luna"]["pricing"]["input_per_mtok"] == 0.2
    assert registry["provider_models"]["openai/gpt-5.6-luna"]["pricing"]["output_per_mtok"] == 1.2
    assert report["pricing_overrides"][0]["status"] == "applied"


def test_report_summarizes_applied_and_stale_overrides():
    report = build_report(
        pricing_overrides=[
            {
                "model": "gpt-5.6-luna",
                "status": "applied",
                "changed_targets": ["gpt-5.6-luna"],
                "changed_fields": ["input_per_mtok"],
            },
            {"model": "gone", "status": "unmatched", "changed_targets": [], "changed_fields": []},
        ]
    )

    assert report["summary"]["pricing_overrides_applied"] == 1
    assert report["summary"]["pricing_overrides_stale"] == 1

    markdown = build_markdown_report(report)
    assert "## Pricing Overrides" in markdown
    assert "gpt-5.6-luna: corrected input_per_mtok on gpt-5.6-luna" in markdown
    assert "gone: unmatched" in markdown


def test_report_omits_the_section_when_no_overrides_are_configured():
    markdown = build_markdown_report(build_report())

    assert "Pricing Overrides" not in markdown


def test_curated_override_file_is_well_formed():
    curated = load_curated_config(REPO_ROOT / "registry" / "curated")
    overrides = curated.get("pricing_overrides", {}).get("overrides", [])

    assert overrides, "expected at least one curated pricing override"
    for entry in overrides:
        assert isinstance(entry["model"], str) and entry["model"]
        assert entry["pricing"], "an override must name the fields it corrects"
        assert entry["source_url"].startswith("https://")
        assert entry["reason"], "an override must record why it exists"
        assert entry["verified_on"], "an override must record when it was last checked"


def test_shipped_registry_matches_the_curated_overrides():
    """The committed artifacts must already reflect every override.

    ``models.json`` is regenerated on a schedule and auto-merged, so a build that
    would still change these prices means the shipped registry is out of date.
    """
    curated = load_curated_config(REPO_ROOT / "registry" / "curated")
    registry = json.loads((REPO_ROOT / "models.json").read_text(encoding="utf-8"))

    findings = apply_pricing_overrides(registry, curated.get("pricing_overrides"))

    assert [entry for entry in findings if entry["status"] == "applied"] == []
