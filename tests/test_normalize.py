from pipeline.normalize import (
    NORMALIZER_BY_SOURCE,
    normalize_litellm_entry,
    normalize_llm_prices_rows,
    normalize_openrouter_rows,
    normalize_portkey_files,
    normalize_pydantic_genai_rows,
    normalize_xai_models_official_rows,
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


def test_normalize_litellm_entry_extracts_richer_fields_and_nested_provider_hints() -> None:
    entry = {
        "model_name": "groq/openai/gpt-oss-120b",
        "litellm_provider": "groq",
        "mode": "chat",
        "max_input_tokens": 131_072,
        "max_output_tokens": 32_766,
        "supported_modalities": ["text", "image", "audio"],
        "supported_output_modalities": ["text"],
        "supports_function_calling": True,
        "supports_reasoning": True,
        "supports_web_search": True,
        "supported_endpoints": ["/v1/chat/completions"],
        "rpm": 1_000,
        "input_cost_per_token": 0.00000015,
        "cache_read_input_token_cost": 0.000000075,
        "output_cost_per_token": 0.0000006,
        "output_cost_per_reasoning_token": 0.0000025,
    }

    record = normalize_litellm_entry(entry, rejection_policy={})

    assert record.provider_slug == "groq"
    assert record.canonical_hint == "gpt-oss-120b"
    assert record.fields["display_name"] == "GPT OSS 120b"
    assert record.fields["context_window"] == 131_072
    assert record.fields["max_output_tokens"] == 32_766
    assert record.fields["modalities"] == {
        "input": ["text", "image", "audio"],
        "output": ["text"],
    }
    assert record.fields["capabilities"] == {
        "function_calling": True,
        "reasoning": True,
        "web_search": True,
    }
    assert record.fields["endpoints"] == ["/v1/chat/completions"]
    assert record.fields["rate_limits"] == {"rpm": 1_000}
    assert record.fields["pricing"] == {
        "currency": "USD",
        "input_per_mtok": 0.15,
        "cached_input_per_mtok": 0.075,
        "output_per_mtok": 0.6,
        "reasoning_output_per_mtok": 2.5,
    }


def test_normalize_litellm_entry_converts_audio_hours_to_integer_seconds() -> None:
    entry = {
        "model_name": "gemini/gemini-flash-latest",
        "litellm_provider": "gemini",
        "mode": "chat",
        "max_audio_length_hours": 8.4,
    }

    record = normalize_litellm_entry(entry, rejection_policy={})

    assert record.fields["max_audio_length_seconds"] == 30240
    assert isinstance(record.fields["max_audio_length_seconds"], int)


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

    record = normalize_llm_prices_rows(
        rows,
        rejection_policy={},
        evidence_ref="sources/llm-prices/2026-03-23.json",
    )[0]

    assert record.provider_slug == "anthropic"
    assert record.canonical_hint == "claude-3.7-sonnet"
    assert record.evidence_ref == "sources/llm-prices/2026-03-23.json"
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


def test_normalize_pydantic_genai_rows_promotes_xai_model_to_exact_canonical_record() -> None:
    rows = [
        {
            "id": "x-ai",
            "pricing_urls": ["https://docs.x.ai/docs/models"],
            "models": [
                {
                    "id": "grok-4-0709",
                    "name": "Grok 4",
                    "description": "Flagship model.",
                    "context_window": 256_000,
                    "match": {
                        "or": [
                            {"equals": "grok-4-0709"},
                            {"equals": "grok-4"},
                            {"equals": "grok-4-latest"},
                        ]
                    },
                    "prices": {
                        "input_mtok": 3,
                        "cache_read_mtok": 0.75,
                        "output_mtok": 15,
                    },
                }
            ],
        }
    ]

    records = normalize_pydantic_genai_rows(
        rows,
        rejection_policy={},
        allowed_providers=["xai"],
    )

    assert [record.source_model_id for record in records] == ["grok-4", "xai/grok-4-0709", "xai/grok-4-latest"]
    assert records[0].source_name == "official"
    assert records[0].provider_slug == "xai"
    assert records[0].canonical_hint == "grok-4"
    assert records[0].evidence_ref == "https://docs.x.ai/docs/models"
    assert records[0].fields == {
        "context_window": 256_000,
        "description": "Flagship model.",
        "display_name": "Grok 4",
        "modes": ["chat"],
        "owned_by": "xai",
        "pricing": {
            "currency": "USD",
            "input_per_mtok": 3.0,
            "cached_input_per_mtok": 0.75,
            "output_per_mtok": 15.0,
        },
    }
    assert records[1].canonical_hint == "grok-4"
    assert records[2].canonical_hint == "grok-4"


def test_normalize_pydantic_genai_rows_supports_google_official_models() -> None:
    rows = [
        {
            "id": "google",
            "pricing_urls": ["https://ai.google.dev/pricing"],
            "models": [
                {
                    "id": "gemini-embedding-001",
                    "match": {"equals": "gemini-embedding-001"},
                    "prices": {"input_mtok": 0.15},
                },
                {
                    "id": "gemini-2.5-flash",
                    "name": "Gemini 2.5 Flash",
                    "match": {
                        "or": [
                            {"equals": "gemini-2.5-flash"},
                            {"equals": "gemini-2.5-flash-latest"},
                        ]
                    },
                    "prices": {"input_mtok": 0.3, "output_mtok": 2.5},
                },
            ],
        }
    ]

    records = normalize_pydantic_genai_rows(
        rows,
        rejection_policy={},
        allowed_providers=["gemini"],
        owner_providers=["gemini"],
    )

    by_id = {record.source_model_id: record for record in records}
    assert by_id["gemini-embedding-001"].fields == {
        "display_name": "Gemini Embedding 001",
        "modes": ["embedding"],
        "owned_by": "gemini",
        "pricing": {
            "currency": "USD",
            "input_per_mtok": 0.15,
        },
    }
    assert by_id["gemini-2.5-flash"].fields["display_name"] == "Gemini 2.5 Flash"
    assert by_id["gemini/gemini-2.5-flash-latest"].canonical_hint == "gemini-2.5-flash"


def test_normalize_xai_models_official_rows_adds_new_xai_models_and_modalities() -> None:
    payload = {
        "source_url": "https://docs.x.ai/developers/models?cluster=us-east-1",
        "language_models": [
            {
                "name": "grok-4.20-0309-reasoning",
                "aliases": ["grok-4.20", "grok-4.20-beta"],
                "maxPromptLength": 2_000_000,
                "promptTextTokenPrice": "$n20000",
                "cachedPromptTokenPrice": "$n2000",
                "completionTextTokenPrice": "$n60000",
                "longContextThreshold": "$n200000",
                "promptTextTokenPriceLongContext": "$n40000",
                "completionTokenPriceLongContext": "$n120000",
            }
        ],
        "image_generation_models": [
            {
                "name": "grok-imagine-image",
                "aliases": ["grok-imagine-image-2026-03-02"],
                "imagePrice": "$n200000000",
                "pricePerInputImage": "$n20000000",
            }
        ],
        "video_generation_models": [
            {
                "name": "grok-imagine-video",
                "aliases": [],
                "resolutionPricing": [
                    {"pricePerSecond": "$n500000000"},
                    {"pricePerSecond": "$n700000000"},
                ],
                "pricePerInputImage": "$n20000000",
                "pricePerInputVideoSecond": "$n100000000",
            }
        ],
    }

    records = normalize_xai_models_official_rows(payload, rejection_policy={})

    by_id = {record.source_model_id: record for record in records}
    assert by_id["grok-4.20"].fields == {
        "context_window": 2_000_000,
        "display_name": "Grok 4.20",
        "modes": ["chat"],
        "owned_by": "xai",
        "pricing": {
            "cached_input_per_mtok": 0.2,
            "currency": "USD",
            "input_per_mtok": 2.0,
            "output_per_mtok": 6.0,
            "tiers": [
                {
                    "up_to_tokens": 200000,
                    "input_per_mtok": 2.0,
                    "output_per_mtok": 6.0,
                },
                {
                    "up_to_tokens": 2000000,
                    "input_per_mtok": 4.0,
                    "output_per_mtok": 12.0,
                },
            ],
        },
        "source_url": "https://docs.x.ai/developers/models?cluster=us-east-1",
    }
    assert by_id["xai/grok-4.20-0309-reasoning"].canonical_hint == "grok-4.20"
    assert by_id["grok-imagine-image"].fields["modes"] == ["image_generation"]
    assert by_id["grok-imagine-image"].fields["pricing"] == {
        "currency": "USD",
        "input_per_image": 0.02,
        "per_image": 0.2,
    }
    assert by_id["grok-imagine-video"].fields["modes"] == ["video_generation"]
    assert by_id["grok-imagine-video"].fields["pricing"] == {
        "currency": "USD",
        "input_per_image": 0.02,
        "per_second_input": 0.1,
        "per_second_output": 0.5,
    }


def test_normalizer_registry_covers_current_source_set() -> None:
    assert set(NORMALIZER_BY_SOURCE) == {"litellm", "xai_models_official", "pydantic_genai", "openrouter", "llm_prices", "portkey"}
