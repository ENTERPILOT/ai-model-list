"""Parsing OpenAI's published pricing page into a price-only evidence overlay."""

from __future__ import annotations

import pytest

from pipeline.normalize import normalize_openai_docs_rows
from pipeline.openai_docs import build_openai_models_snapshot


SOURCE_URL = "https://developers.openai.com/api/docs/pricing"

# Trimmed to the shape that matters: four tier tables, in page order, sharing one
# column layout. Only Standard and Batch are read; Flex and Fast must be ignored.
PRICING_MARKDOWN = """# Pricing

Standard

### Standard pricing data

| Model | Short context input | Short context cached input | Short context cache writes | Short context output | Long context input | Long context cached input | Long context cache writes | Long context output |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-5.6-luna | $0.20 | $0.02 | $0.25 | $1.20 | $0.40 | $0.04 | $0.50 | $1.80 |
| gpt-5.5 (<272K context length) | $5.00 | $0.50 | - | $30.00 | $10.00 | $1.00 | - | $45.00 |

Some prose about regional processing that is not a table.

Batch

### Batch pricing data

| Model | Short context input | Short context cached input | Short context cache writes | Short context output | Long context input | Long context cached input | Long context cache writes | Long context output |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-5.6-luna | $0.10 | $0.01 | $0.125 | $0.60 | $0.20 | $0.02 | $0.25 | $0.90 |
| gpt-5.5 (<272K context length) | $2.50 | $0.25 | - | $15.00 | $5.00 | $0.50 | - | $22.50 |

Flex

### Flex pricing data

| Model | Short context input | Short context cached input | Short context cache writes | Short context output | Long context input | Long context cached input | Long context cache writes | Long context output |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-5.6-luna | $0.10 | $0.01 | $0.125 | $0.60 | $0.20 | $0.02 | $0.25 | $0.90 |

Fast mode

### Fast pricing data

| Model | Short context input | Short context cached input | Short context cache writes | Short context output | Long context input | Long context cached input | Long context cache writes | Long context output |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-5.6-luna | $0.40 | $0.04 | $0.50 | $2.40 | $0.80 | $0.08 | $1.00 | $3.60 |

### Grouped Pricing Table data

| Model | Modality | Input | Cached input | Output / cost |
| --- | --- | --- | --- | --- |
| gpt-realtime-2.1 | Audio | $32.00 | $0.40 | $64.00 |
"""


def _models() -> dict[str, dict]:
    snapshot = build_openai_models_snapshot(PRICING_MARKDOWN, SOURCE_URL)
    assert len(snapshot) == 1
    assert snapshot[0]["id"] == "openai"
    assert snapshot[0]["pricing_urls"] == [SOURCE_URL]
    return {model["id"]: model for model in snapshot[0]["models"]}


def test_reads_short_context_standard_rates_as_the_headline_price():
    prices = _models()["gpt-5.6-luna"]["prices"]

    assert prices["input_mtok"] == 0.2
    assert prices["output_mtok"] == 1.2
    assert prices["cache_read_mtok"] == 0.02
    assert prices["cache_write_mtok"] == 0.25


def test_reads_batch_rates_from_the_batch_table():
    prices = _models()["gpt-5.6-luna"]["prices"]

    assert prices["batch_input_mtok"] == 0.1
    assert prices["batch_output_mtok"] == 0.6


def test_flex_and_fast_tables_never_leak_into_the_published_rates():
    """A section that runs past its own table silently picks up the next tier."""
    prices = _models()["gpt-5.6-luna"]["prices"]

    assert 0.4 not in prices.values(), "Fast mode input rate leaked into the snapshot"
    assert 2.4 not in prices.values(), "Fast mode output rate leaked into the snapshot"


def test_context_length_annotations_are_stripped_from_model_ids():
    models = _models()

    assert "gpt-5.5" in models
    assert not any("(" in model_id for model_id in models)


def test_absent_prices_are_omitted_rather_than_zeroed():
    prices = _models()["gpt-5.5"]["prices"]

    assert "cache_write_mtok" not in prices
    assert prices["input_mtok"] == 5.0


def test_only_ids_the_page_publishes_are_emitted():
    """Inventing an id spelling would invent a provider model OpenAI may not serve."""
    for model in _models().values():
        assert model["match"] == {"or": [{"equals": model["id"]}]}


def test_unparseable_page_raises_so_the_snapshot_is_skipped():
    with pytest.raises(ValueError):
        build_openai_models_snapshot("# Pricing\n\nNo tables here.\n", SOURCE_URL)


def test_normalizer_emits_pricing_only_evidence():
    """Catalog metadata must keep coming from the feed — the page has none."""
    snapshot = build_openai_models_snapshot(PRICING_MARKDOWN, SOURCE_URL)

    records = normalize_openai_docs_rows(snapshot, allowed_providers=["openai"], owner_providers=["openai"])

    assert records
    for record in records:
        assert record.source_name == "openai_official"
        assert record.confidence == "official"
        assert record.provider_slug == "openai"
        assert set(record.fields) == {"pricing"}, "overlay must not compete for catalog fields"

    luna = next(r for r in records if r.source_model_id == "gpt-5.6-luna")
    assert luna.fields["pricing"]["input_per_mtok"] == 0.2
    assert luna.fields["pricing"]["output_per_mtok"] == 1.2
    assert luna.fields["pricing"]["batch_input_per_mtok"] == 0.1


def test_overlay_outranks_the_aggregated_feed_for_pricing():
    from pipeline.rules import sort_candidates_by_authority
    from pipeline.types import SourceEvidence

    feed = SourceEvidence(
        source_name="official",
        source_model_id="gpt-5.6-luna",
        provider_slug="openai",
        canonical_hint="gpt-5.6-luna",
        fields={"pricing": {"input_per_mtok": 1.0}},
        confidence="official",
        evidence_ref="pydantic_genai_prices.json",
    )
    overlay = SourceEvidence(
        source_name="openai_official",
        source_model_id="gpt-5.6-luna",
        provider_slug="openai",
        canonical_hint="gpt-5.6-luna",
        fields={"pricing": {"input_per_mtok": 0.2}},
        confidence="official",
        evidence_ref="openai_models_official.json",
    )
    policy = {"field_authority": {"pricing": ["openai_official", "official", "portkey"]}}

    ranked = sort_candidates_by_authority("pricing", [feed, overlay], policy)

    assert ranked[0] is overlay
