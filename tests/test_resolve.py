from pipeline.resolve import canonicalize_model_id, resolve_registry
from pipeline.types import SourceEvidence


def test_canonicalize_model_id_maps_claude_opus_41_aliases() -> None:
    aliases = {
        "claude-opus-41": "claude-opus-4-1",
        "claude-opus-4.1": "claude-opus-4-1",
    }

    assert canonicalize_model_id("claude-opus-41", aliases) == "claude-opus-4-1"
    assert canonicalize_model_id("claude-opus-4.1", aliases) == "claude-opus-4-1"
    assert canonicalize_model_id("claude-opus-4-1", aliases) == "claude-opus-4-1"


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
    assert "pricing" not in registry["models"]["grok-4"]
    assert registry["provider_models"]["xai/grok-4"]["pricing"] == {
        "currency": "USD",
        "input_per_mtok": 3.0,
        "output_per_mtok": 15.0,
    }
    assert "provider_model_id" not in registry["provider_models"]["xai/grok-4"]
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


def test_resolve_does_not_promote_clean_alias_from_canonical_hint_without_curated_alias() -> None:
    evidence = [
        SourceEvidence(
            source_name="official",
            source_model_id="grok-four",
            provider_slug="xai",
            canonical_hint="grok-4",
            fields={"display_name": "Grok 4", "owned_by": "xai", "modes": ["chat"]},
            confidence="official",
            evidence_ref="https://docs.x.ai/docs/models",
        )
    ]

    registry, report = resolve_registry(evidence, curated={})

    assert "grok-4" not in registry["models"]
    assert "grok-four" not in registry["models"]
    assert report["quarantine"]


def test_resolve_admits_clean_alias_records_via_curated_aliases() -> None:
    evidence = [
        SourceEvidence(
            source_name="official",
            source_model_id="grok-four",
            provider_slug="xai",
            canonical_hint="grok-four",
            fields={"display_name": "Grok 4", "owned_by": "xai", "modes": ["chat"]},
            confidence="official",
            evidence_ref="https://docs.x.ai/docs/models",
        )
    ]

    registry, report = resolve_registry(
        evidence,
        curated={"canonical_aliases": {"grok-four": "grok-4"}},
    )

    assert "grok-4" in registry["models"]
    assert registry["models"]["grok-4"]["display_name"] == "Grok 4"
    assert registry["models"]["grok-4"]["aliases"] == ["grok-four", "xai/grok-four"]
    assert registry["provider_models"]["xai/grok-4"] == {
        "enabled": True,
        "model_ref": "grok-4",
        "provider_model_id": "grok-four",
    }
    assert registry["provider_models"]["xai/grok-four"] == {
        "enabled": True,
        "model_ref": "grok-4",
    }
    assert not report["quarantine"]


def test_resolve_admits_exact_canonical_record_even_with_different_hint() -> None:
    evidence = [
        SourceEvidence(
            source_name="official",
            source_model_id="grok-4",
            provider_slug="xai",
            canonical_hint="xai/grok-4",
            fields={"display_name": "Grok 4", "owned_by": "xai", "modes": ["chat"]},
            confidence="official",
            evidence_ref="https://docs.x.ai/docs/models",
        )
    ]

    registry, report = resolve_registry(evidence, curated={})

    assert "grok-4" in registry["models"]
    assert registry["models"]["grok-4"]["display_name"] == "Grok 4"
    assert not report["quarantine"]


def test_resolve_admits_exact_canonical_record_without_official_confidence() -> None:
    evidence = [
        SourceEvidence(
            source_name="openrouter",
            source_model_id="grok-4",
            provider_slug="xai",
            canonical_hint="grok-4",
            fields={"display_name": "Grok 4", "modes": ["chat"]},
            confidence="low",
            evidence_ref="openrouter_models.json",
        )
    ]

    registry, report = resolve_registry(evidence, curated={})

    assert "grok-4" in registry["models"]
    assert registry["models"]["grok-4"]["display_name"] == "Grok 4"
    assert not report["quarantine"]


def test_resolve_adds_provider_model_for_exact_canonical_record() -> None:
    evidence = [
        SourceEvidence(
            source_name="litellm",
            source_model_id="chatgpt-4o-latest",
            provider_slug="openai",
            canonical_hint="chatgpt-4o-latest",
            fields={"display_name": "ChatGPT 4o Latest", "modes": ["chat"]},
            confidence="low",
            evidence_ref="litellm_model_prices.json",
        )
    ]

    registry, report = resolve_registry(evidence, curated={})

    assert "chatgpt-4o-latest" in registry["models"]
    assert registry["provider_models"]["openai/chatgpt-4o-latest"] == {
        "enabled": True,
        "model_ref": "chatgpt-4o-latest",
    }
    assert not report["quarantine"]


def test_resolve_quarantines_alias_only_cluster_missing_required_model_fields() -> None:
    evidence = [
        SourceEvidence(
            source_name="official",
            source_model_id="grok-four",
            provider_slug="xai",
            canonical_hint="grok-four",
            fields={"pricing": {"currency": "USD", "input_per_mtok": 4.0, "output_per_mtok": 20.0}},
            confidence="official",
            evidence_ref="https://docs.x.ai/docs/models",
        )
    ]

    registry, report = resolve_registry(
        evidence,
        curated={"canonical_aliases": {"grok-four": "grok-4"}},
    )

    assert "grok-4" not in registry["models"]
    assert report["quarantine"]


def test_resolve_routes_clean_alias_overrides_only_to_provider_models_when_exact_canonical_exists() -> None:
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
            source_name="official",
            source_model_id="grok-four-release",
            provider_slug="xai",
            canonical_hint="grok-four-release",
            fields={"pricing": {"currency": "USD", "input_per_mtok": 4.0, "output_per_mtok": 20.0}},
            confidence="official",
            evidence_ref="https://docs.x.ai/docs/models",
        ),
    ]

    registry, report = resolve_registry(
        evidence,
        curated={"canonical_aliases": {"grok-four-release": "grok-4"}},
    )

    assert "xai/grok-4" in registry["provider_models"]
    assert "xai/grok-four-release" in registry["provider_models"]
    assert "pricing" not in registry["models"]["grok-4"]
    assert registry["provider_models"]["xai/grok-4"]["pricing"] == {
        "currency": "USD",
        "input_per_mtok": 4.0,
        "output_per_mtok": 20.0,
    }
    assert registry["provider_models"]["xai/grok-4"]["provider_model_id"] == "grok-four-release"
    assert not report["quarantine"]


def test_resolve_routes_deployment_style_aliases_without_separator_to_provider_models() -> None:
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
            source_name="official",
            source_model_id="grok-4-us-east",
            provider_slug="xai",
            canonical_hint="grok-4-us-east",
            fields={"pricing": {"currency": "USD", "input_per_mtok": 5.0, "output_per_mtok": 25.0}},
            confidence="official",
            evidence_ref="https://docs.x.ai/docs/models",
        ),
    ]

    registry, report = resolve_registry(
        evidence,
        curated={"canonical_aliases": {"grok-4-us-east": "grok-4"}},
    )

    assert "xai/grok-4" in registry["provider_models"]
    assert "xai/grok-4-us-east" in registry["provider_models"]
    assert registry["provider_models"]["xai/grok-4"]["provider_model_id"] == "grok-4-us-east"
    assert registry["provider_models"]["xai/grok-4"]["pricing"] == {
        "currency": "USD",
        "input_per_mtok": 5.0,
        "output_per_mtok": 25.0,
    }
    assert "pricing" not in registry["models"]["grok-4"]
    assert not report["quarantine"]


def test_resolve_omits_provider_model_id_for_multi_alias_provider_cluster() -> None:
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
            source_name="official",
            source_model_id="grok-four-release",
            provider_slug="xai",
            canonical_hint="grok-four-release",
            fields={"pricing": {"currency": "USD", "input_per_mtok": 4.0, "output_per_mtok": 20.0}},
            confidence="official",
            evidence_ref="https://docs.x.ai/docs/models",
        ),
        SourceEvidence(
            source_name="official",
            source_model_id="grok-four-preview",
            provider_slug="xai",
            canonical_hint="grok-four-preview",
            fields={"context_window": 256000},
            confidence="official",
            evidence_ref="https://docs.x.ai/docs/models",
        ),
    ]

    registry, report = resolve_registry(
        evidence,
        curated={
            "canonical_aliases": {
                "grok-four-release": "grok-4",
                "grok-four-preview": "grok-4",
            }
        },
    )

    assert "xai/grok-4" in registry["provider_models"]
    assert "provider_model_id" not in registry["provider_models"]["xai/grok-4"]
    assert not report["quarantine"]


def test_resolve_collects_aliases_and_exact_provider_keys_without_merging_variants() -> None:
    evidence = [
        SourceEvidence(
            source_name="official",
            source_model_id="gpt-oss-120b",
            provider_slug="fireworks",
            canonical_hint="gpt-oss-120b",
            fields={"display_name": "GPT-OSS 120B", "modes": ["chat"]},
            confidence="official",
            evidence_ref="https://fireworks.ai/pricing",
        ),
        SourceEvidence(
            source_name="openrouter",
            source_model_id="openai/gpt-oss-120b",
            provider_slug="openai",
            canonical_hint="gpt-oss-120b",
            fields={"description": "OpenAI open-weight model"},
            confidence="low",
            evidence_ref="openrouter_models.json",
        ),
        SourceEvidence(
            source_name="litellm",
            source_model_id="groq/openai/gpt-oss-120b",
            provider_slug="groq",
            canonical_hint="gpt-oss-120b",
            fields={"pricing": {"currency": "USD", "input_per_mtok": 0.15, "output_per_mtok": 0.6}},
            confidence="low",
            evidence_ref="litellm_model_prices.json",
        ),
    ]

    registry, report = resolve_registry(evidence, curated={})

    assert registry["models"]["gpt-oss-120b"]["aliases"] == [
        "openai/gpt-oss-120b",
        "fireworks/gpt-oss-120b",
        "groq/openai/gpt-oss-120b",
    ]
    assert "fireworks/gpt-oss-120b" in registry["provider_models"]
    assert registry["provider_models"]["openai/gpt-oss-120b"] == {
        "enabled": True,
        "model_ref": "gpt-oss-120b",
    }
    assert registry["provider_models"]["groq/openai/gpt-oss-120b"] == {
        "enabled": True,
        "model_ref": "gpt-oss-120b",
        "pricing": {"currency": "USD", "input_per_mtok": 0.15, "output_per_mtok": 0.6},
    }
    assert not report["quarantine"]


def test_resolve_admits_clean_provider_prefixed_record_without_curated_alias() -> None:
    evidence = [
        SourceEvidence(
            source_name="openrouter",
            source_model_id="qwen/qwen3-vl-32b-instruct",
            provider_slug="openrouter",
            canonical_hint="qwen3-vl-32b-instruct",
            fields={
                "display_name": "Qwen: Qwen3 VL 32B Instruct",
                "modes": ["chat"],
                "pricing": {"currency": "USD", "input_per_mtok": 0.104, "output_per_mtok": 0.416},
            },
            confidence="low",
            evidence_ref="openrouter_models.json",
        )
    ]

    registry, report = resolve_registry(evidence, curated={})

    assert registry["models"]["qwen3-vl-32b-instruct"]["aliases"] == [
        "qwen/qwen3-vl-32b-instruct",
        "openrouter/qwen/qwen3-vl-32b-instruct",
    ]
    assert registry["provider_models"]["openrouter/qwen/qwen3-vl-32b-instruct"] == {
        "enabled": True,
        "model_ref": "qwen3-vl-32b-instruct",
        "pricing": {"currency": "USD", "input_per_mtok": 0.104, "output_per_mtok": 0.416},
    }
    assert not report["quarantine"]


def test_resolve_merges_pricing_components_across_authorities() -> None:
    evidence = [
        SourceEvidence(
            source_name="portkey",
            source_model_id="gpt-image-1",
            provider_slug="openai",
            canonical_hint="gpt-image-1",
            fields={
                "display_name": "GPT Image 1",
                "modes": ["image_generation"],
                "pricing": {
                    "currency": "USD",
                    "input_image_per_mtok": 10.0,
                    "output_image_per_mtok": 40.0,
                    "image_generation_prices": [
                        {"quality": "high", "size": "1024x1024", "price": 0.167}
                    ],
                },
            },
            confidence="low",
            evidence_ref="portkey/openai.json",
        ),
        SourceEvidence(
            source_name="litellm",
            source_model_id="gpt-image-1",
            provider_slug="openai",
            canonical_hint="gpt-image-1",
            fields={
                "pricing": {
                    "currency": "USD",
                    "input_per_mtok": 5.0,
                    "output_per_mtok": 10.0,
                }
            },
            confidence="low",
            evidence_ref="litellm_model_prices.json",
        ),
    ]

    registry, _ = resolve_registry(evidence, curated={})

    assert registry["models"]["gpt-image-1"]["pricing"] == {
        "currency": "USD",
        "input_image_per_mtok": 10.0,
        "output_image_per_mtok": 40.0,
        "image_generation_prices": [
            {"quality": "high", "size": "1024x1024", "price": 0.167}
        ],
        "input_per_mtok": 5.0,
        "output_per_mtok": 10.0,
    }
