from pipeline.normalize import (
    NORMALIZER_BY_SOURCE,
    normalize_litellm_entry,
    normalize_llm_prices_rows,
    normalize_openrouter_rows,
    normalize_portkey_files,
)


def test_normalize_litellm_entry_extracts_provider_and_canonical_hint() -> None:
    entry = {
        "model_name": "xai/grok-4",
        "litellm_provider": "xai",
        "mode": "chat",
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
    }

    record = normalize_litellm_entry(entry, rejection_policy={})

    assert record.provider_slug == "xai"
    assert record.canonical_hint == "grok-4"
    assert record.source_name == "litellm"


def test_normalize_litellm_entry_marks_sample_spec_as_rejected() -> None:
    entry = {
        "model_name": "sample_spec",
        "litellm_provider": "openai",
        "mode": "chat",
    }

    record = normalize_litellm_entry(
        entry,
        rejection_policy={
            "exact_model_ids": ["sample_spec"],
            "prefixes": ["1024-x-1024/"],
        },
    )

    assert record.rejected is True


def test_normalize_litellm_rows_supports_dict_shaped_sources() -> None:
    rows = {
        "xai/grok-4": {
            "litellm_provider": "xai",
            "mode": "chat",
            "input_cost_per_token": 0.000003,
            "output_cost_per_token": 0.000015,
        }
    }

    record = NORMALIZER_BY_SOURCE["litellm"](rows, rejection_policy={})[0]

    assert record.source_model_id == "xai/grok-4"
    assert record.provider_slug == "xai"
    assert record.canonical_hint == "grok-4"
    assert record.fields["modes"] == ["chat"]


def test_normalize_openrouter_rows_normalizes_pricing_and_metadata() -> None:
    rows = [
        {
            "id": "anthropic/claude-sonnet-4.6",
            "canonical_slug": "anthropic/claude-4.6-sonnet-20260217",
            "name": "Anthropic: Claude Sonnet 4.6",
            "pricing": {"prompt": "0.000003", "completion": "0.000015"},
            "top_provider": {
                "context_length": 1_000_000,
                "max_completion_tokens": 128_000,
            },
        }
    ]

    record = normalize_openrouter_rows(
        rows,
        evidence_ref="sources/openrouter/2026-03-23.json",
        rejection_policy={},
    )[0]

    assert record.provider_slug == "anthropic"
    assert record.canonical_hint == "claude-4.6-sonnet-20260217"
    assert record.evidence_ref == "sources/openrouter/2026-03-23.json"
    assert record.fields["display_name"] == "Anthropic: Claude Sonnet 4.6"
    assert record.fields["pricing"] == {
        "currency": "USD",
        "input_per_mtok": 3.0,
        "output_per_mtok": 15.0,
    }
    assert record.fields["context_window"] == 1_000_000
    assert record.fields["max_output_tokens"] == 128_000


def test_normalize_llm_prices_rows_normalizes_vendor_rows() -> None:
    rows = [
        {
            "id": "claude-3.7-sonnet",
            "vendor": "anthropic",
            "name": "Claude 3.7 Sonnet",
            "input": 3,
            "output": 15,
        }
    ]

    record = normalize_llm_prices_rows(rows, rejection_policy={})[0]

    assert record.provider_slug == "anthropic"
    assert record.canonical_hint == "claude-3.7-sonnet"
    assert record.fields["display_name"] == "Claude 3.7 Sonnet"
    assert record.fields["pricing"] == {
        "currency": "USD",
        "input_per_mtok": 3.0,
        "output_per_mtok": 15.0,
    }


def test_normalize_portkey_files_skips_default_and_converts_cents() -> None:
    files = {
        "anthropic.json": {
            "default": {
                "pricing_config": {
                    "pay_as_you_go": {
                        "request_token": {"price": 0},
                        "response_token": {"price": 0},
                    }
                }
            },
            "claude-2.1": {
                "pricing_config": {
                    "pay_as_you_go": {
                        "request_token": {"price": 0.0008},
                        "response_token": {"price": 0.0024},
                    }
                }
            },
        }
    }

    records = normalize_portkey_files(files, rejection_policy={})

    assert len(records) == 1
    assert records[0].provider_slug == "anthropic"
    assert records[0].canonical_hint == "claude-2.1"
    assert records[0].evidence_ref == "portkey/anthropic.json"
    assert records[0].fields["pricing"] == {
        "currency": "USD",
        "input_per_mtok": 8.0,
        "output_per_mtok": 24.0,
    }


def test_normalizer_registry_covers_current_source_set() -> None:
    assert set(NORMALIZER_BY_SOURCE) == {"litellm", "openrouter", "llm_prices", "portkey"}
