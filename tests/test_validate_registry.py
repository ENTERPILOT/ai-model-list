import json
from pathlib import Path

from pipeline.loaders import load_curated_config
from scripts import build_registry as build_registry_module
from scripts import fetch_sources as fetch_sources_module
from scripts.build_registry import build_registry
from scripts.fetch_sources import GITHUB_SNAPSHOT_BASE_URL, SOURCE_DESCRIPTORS, snapshot_path_for_run
from scripts.validate import check_registry_quality


def _write_minimal_curated_config(curated_dir: Path) -> None:
    curated_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in {
        "providers.json": {},
        "source_policies.json": {},
        "canonical_aliases.json": {},
        "rejections.json": {},
    }.items():
        (curated_dir / name).write_text(json.dumps(payload), encoding="utf-8")


def test_snapshot_path_for_run_nests_under_source_snapshots(tmp_path: Path) -> None:
    assert snapshot_path_for_run(tmp_path, "2026-03-23T06-15-00Z") == tmp_path / "source_snapshots" / "2026-03-23T06-15-00Z"


def test_fetch_sources_declares_expected_aggregator_sources() -> None:
    assert {descriptor.slug for descriptor in SOURCE_DESCRIPTORS} >= {"litellm", "openrouter", "llm_prices"}
    assert {descriptor.filename for descriptor in SOURCE_DESCRIPTORS} >= {
        "fetch_metadata.json",
        "litellm_model_prices.json",
        "openrouter_models.json",
        "llm_prices_current.json",
        "portkey/openai.json",
    }
    assert all(descriptor.url.startswith(GITHUB_SNAPSHOT_BASE_URL) for descriptor in SOURCE_DESCRIPTORS)


def test_build_registry_returns_top_level_sections(tmp_path: Path) -> None:
    _write_minimal_curated_config(tmp_path)
    result = build_registry(snapshot_dir=tmp_path, curated_dir=tmp_path)
    assert set(result.keys()) == {"version", "updated_at", "providers", "models", "provider_models"}


def test_build_registry_cli_writes_deterministic_outputs(tmp_path: Path, monkeypatch) -> None:
    _write_minimal_curated_config(tmp_path / "registry" / "curated")

    def fake_fetch_sources(snapshot_dir: Path) -> Path:
        return snapshot_dir

    def fake_build_registry_artifacts(snapshot_dir: Path, curated_dir: Path) -> tuple[dict, dict, list[dict]]:
        return (
            {
                "version": 1,
                "updated_at": "1970-01-01T00:00:00Z",
                "providers": {},
                "models": {},
                "provider_models": {},
            },
            {
                "summary": {"duplicate_clusters": 0, "quarantine_count": 0},
                "duplicate_clusters": [],
                "resolved_duplicates": [],
                "quarantine": [],
                "new_models": [],
            },
            [],
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build_registry_module, "fetch_sources_to", fake_fetch_sources)
    monkeypatch.setattr(build_registry_module, "build_registry_artifacts", fake_build_registry_artifacts)

    exit_code = build_registry_module.main(
        [
            "--report-md",
            str(tmp_path / "tmp" / "build" / "report.md"),
        ]
    )

    assert exit_code == 0

    models_path = tmp_path / "models.json"
    models_min_path = tmp_path / "models.min.json"
    report_json_path = tmp_path / "tmp" / "build" / "report.json"
    report_md_path = tmp_path / "tmp" / "build" / "report.md"

    expected_registry = {
        "version": 1,
        "updated_at": "1970-01-01T00:00:00Z",
        "providers": {},
        "models": {},
        "provider_models": {},
    }
    expected_report = {
        "summary": {"duplicate_clusters": 0, "quarantine_count": 0},
        "duplicate_clusters": [],
        "resolved_duplicates": [],
        "quarantine": [],
        "new_models": [],
    }

    assert json.loads(models_path.read_text(encoding="utf-8")) == expected_registry
    assert models_min_path.read_text(encoding="utf-8").strip() == json.dumps(expected_registry, ensure_ascii=False, separators=(",", ":"))
    assert json.loads(report_json_path.read_text(encoding="utf-8")) == expected_report
    assert report_md_path.read_text(encoding="utf-8").startswith("# Registry Audit Report")


def test_build_registry_cli_always_fetches_latest_snapshot_dir(tmp_path: Path, monkeypatch) -> None:
    _write_minimal_curated_config(tmp_path / "registry" / "curated")

    captured: dict[str, Path] = {}

    def fake_fetch_sources(snapshot_dir: Path) -> Path:
        captured["fetched_snapshot_dir"] = snapshot_dir
        return snapshot_dir

    def fake_build_registry_artifacts(snapshot_dir: Path, curated_dir: Path) -> tuple[dict, dict, list[dict]]:
        captured["build_snapshot_dir"] = snapshot_dir
        return (
            {
                "version": 1,
                "updated_at": "1970-01-01T00:00:00Z",
                "providers": {},
                "models": {},
                "provider_models": {},
            },
            {
                "summary": {"duplicate_clusters": 0, "quarantine_count": 0},
                "duplicate_clusters": [],
                "resolved_duplicates": [],
                "quarantine": [],
                "new_models": [],
            },
            [],
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build_registry_module, "fetch_sources_to", fake_fetch_sources)
    monkeypatch.setattr(build_registry_module, "build_registry_artifacts", fake_build_registry_artifacts)

    expected_snapshot_dir = tmp_path / "tmp" / "source_snapshots" / "latest"
    exit_code = build_registry_module.main([])

    assert exit_code == 0
    assert captured["fetched_snapshot_dir"] == expected_snapshot_dir
    assert captured["build_snapshot_dir"] == expected_snapshot_dir


def test_fetch_bytes_retries_with_timeout(monkeypatch) -> None:
    attempts: list[float] = []

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def read(self) -> bytes:
            return b"payload"

    def flaky_urlopen(request, timeout: float):
        attempts.append(timeout)
        if len(attempts) < 3:
            raise TimeoutError("slow upstream")
        return FakeResponse()

    monkeypatch.setattr(fetch_sources_module, "urlopen", flaky_urlopen)
    monkeypatch.setattr(fetch_sources_module.time, "sleep", lambda _delay: None)

    payload = fetch_sources_module._fetch_bytes(
        "https://example.com/models.json",
        timeout=12.5,
        retries=3,
        retry_delay=0.01,
    )

    assert payload == b"payload"
    assert attempts == [12.5, 12.5, 12.5]


def test_check_registry_quality_rejects_provider_specific_model_ids() -> None:
    data = {
        "version": 1,
        "updated_at": "2026-03-23T00:00:00Z",
        "providers": {"bedrock": {"display_name": "AWS Bedrock"}},
        "models": {
            "anthropic.claude-opus-4-6-v1:0": {
                "display_name": "Bad",
                "modes": ["chat"],
            }
        },
        "provider_models": {},
    }

    errors = check_registry_quality(data)

    assert any("provider-specific model key" in error for error in errors)


def test_check_registry_quality_rejects_unknown_owned_by() -> None:
    data = {
        "version": 1,
        "updated_at": "2026-03-23T00:00:00Z",
        "providers": {"openai": {"display_name": "OpenAI"}},
        "models": {
            "gpt-4o": {
                "display_name": "GPT-4o",
                "modes": ["chat"],
                "owned_by": "unknown",
            }
        },
        "provider_models": {},
    }

    errors = check_registry_quality(data)

    assert any("owned_by" in error for error in errors)


def test_check_registry_quality_rejects_family_owned_by_mismatch() -> None:
    data = {
        "version": 1,
        "updated_at": "2026-03-23T00:00:00Z",
        "providers": {
            "anthropic": {"display_name": "Anthropic"},
            "gemini": {"display_name": "Gemini"},
        },
        "models": {
            "claude-3-5-haiku": {
                "display_name": "Claude 3.5 Haiku",
                "modes": ["chat"],
                "owned_by": "gemini",
            }
        },
        "provider_models": {
            "anthropic/claude-3-5-haiku": {"model_ref": "claude-3-5-haiku", "enabled": True}
        },
    }

    errors = check_registry_quality(data)

    assert any("inferred family owner 'anthropic'" in error for error in errors)


def test_check_registry_quality_rejects_provider_without_provider_models() -> None:
    data = {
        "version": 1,
        "updated_at": "2026-03-23T00:00:00Z",
        "providers": {
            "openai": {"display_name": "OpenAI"},
            "runway": {"display_name": "Runway"},
        },
        "models": {
            "gpt-4o": {
                "display_name": "GPT-4o",
                "modes": ["chat"],
                "owned_by": "openai",
            }
        },
        "provider_models": {
            "openai/gpt-4o": {"model_ref": "gpt-4o", "enabled": True}
        },
    }

    errors = check_registry_quality(data)

    assert any("runway: provider has zero provider_models" in error for error in errors)


def test_load_curated_config_reads_authority_files(tmp_path: Path) -> None:
    expected = {
        "providers": {"openai": {"display_name": "OpenAI"}},
        "source_policies": {
            "official_sources": ["openai", "anthropic", "gemini", "xai", "mistral", "cohere", "deepseek"],
            "aggregator_sources": ["litellm", "openrouter", "portkey", "llm_prices"],
            "field_authority": {
                "owned_by": ["official"],
                "display_name": ["official"],
                "pricing": ["official", "portkey", "llm_prices", "openrouter", "litellm"],
            },
        },
        "canonical_aliases": {"gpt-4o-mini": "gpt-4o-mini"},
        "rejections": {
            "exact_model_ids": ["sample_spec", "search_api", "model_router"],
            "prefixes": ["1024-x-1024/", "512-x-512/", "256-x-256/"],
        },
    }

    for name, payload in expected.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")

    config = load_curated_config(tmp_path)
    assert config == expected


def test_build_registry_artifacts_promotes_grok_from_official_xai_catalog(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshot"
    curated_dir = tmp_path / "curated"
    snapshot_dir.mkdir()
    curated_dir.mkdir()

    (curated_dir / "providers.json").write_text(
        json.dumps({"xai": {"display_name": "xAI"}}),
        encoding="utf-8",
    )
    (curated_dir / "source_policies.json").write_text(
        json.dumps(
            {
                "official_sources": ["xai"],
                "aggregator_sources": ["llm_prices"],
                "field_authority": {
                    "owned_by": ["official"],
                    "display_name": ["official"],
                    "pricing": ["official", "llm_prices"],
                },
            }
        ),
        encoding="utf-8",
    )
    (curated_dir / "canonical_aliases.json").write_text(json.dumps({}), encoding="utf-8")
    (curated_dir / "rejections.json").write_text(json.dumps({}), encoding="utf-8")

    (snapshot_dir / "fetch_metadata.json").write_text(
        json.dumps({"fetched_at": "2026-03-23T19:00:35Z", "sources": {}}),
        encoding="utf-8",
    )
    (snapshot_dir / "llm_prices_current.json").write_text(
        json.dumps(
            {
                "updated_at": "2026-03-23T19:00:35Z",
                "prices": [
                    {
                        "id": "grok-4",
                        "vendor": "xai",
                        "name": "Grok 4 ≤128k",
                        "input": 3,
                        "output": 15,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (snapshot_dir / "pydantic_genai_prices.json").write_text(
        json.dumps(
            [
                {
                    "id": "x-ai",
                    "pricing_urls": ["https://docs.x.ai/docs/models"],
                    "models": [
                        {
                            "id": "grok-4-0709",
                            "name": "Grok 4",
                            "description": "Flagship model.",
                            "context_window": 256000,
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
        ),
        encoding="utf-8",
    )

    registry, report, quarantine = build_registry_module.build_registry_artifacts(
        snapshot_dir=snapshot_dir,
        curated_dir=curated_dir,
    )

    assert registry["updated_at"] == "2026-03-23T19:00:35Z"
    assert registry["models"]["grok-4"] == {
        "aliases": [
            "grok-4-0709",
            "grok-4-latest",
            "xai/grok-4",
            "xai/grok-4-0709",
            "xai/grok-4-latest",
        ],
        "context_window": 256000,
        "description": "Flagship model.",
        "display_name": "Grok 4",
        "modes": ["chat"],
        "owned_by": "xai",
        "pricing": {
            "cached_input_per_mtok": 0.75,
            "currency": "USD",
            "input_per_mtok": 3.0,
            "output_per_mtok": 15.0,
        },
        "source_url": "https://docs.x.ai/docs/models",
    }
    assert registry["provider_models"]["xai/grok-4"] == {
        "context_window": 256000,
        "enabled": True,
        "model_ref": "grok-4",
        "pricing": {
            "cached_input_per_mtok": 0.75,
            "currency": "USD",
            "input_per_mtok": 3.0,
            "output_per_mtok": 15.0,
        },
    }
    assert registry["provider_models"]["xai/grok-4-0709"]["model_ref"] == "grok-4"
    assert registry["provider_models"]["xai/grok-4-latest"]["model_ref"] == "grok-4"
    assert not quarantine
    assert report["summary"]["quarantine_count"] == 0


def test_build_registry_artifacts_prefers_live_xai_docs_over_stale_pydantic_subset(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshot"
    curated_dir = tmp_path / "curated"
    snapshot_dir.mkdir()
    curated_dir.mkdir()

    (curated_dir / "providers.json").write_text(
        json.dumps({"xai": {"display_name": "xAI"}}),
        encoding="utf-8",
    )
    (curated_dir / "source_policies.json").write_text(
        json.dumps(
            {
                "official_sources": ["xai"],
                "aggregator_sources": [],
                "field_authority": {
                    "owned_by": ["official"],
                    "display_name": ["official"],
                    "pricing": ["official"],
                },
            }
        ),
        encoding="utf-8",
    )
    (curated_dir / "canonical_aliases.json").write_text(json.dumps({}), encoding="utf-8")
    (curated_dir / "rejections.json").write_text(json.dumps({}), encoding="utf-8")

    (snapshot_dir / "fetch_metadata.json").write_text(
        json.dumps({"fetched_at": "2026-03-24T09:00:00Z", "sources": {}}),
        encoding="utf-8",
    )
    (snapshot_dir / "pydantic_genai_prices.json").write_text(
        json.dumps(
            [
                {
                    "id": "x-ai",
                    "pricing_urls": ["https://docs.x.ai/docs/models"],
                    "models": [
                        {
                            "id": "grok-3",
                            "name": "Grok 3",
                            "match": {"or": [{"equals": "grok-3"}, {"equals": "grok-3-fast"}]},
                            "prices": {"input_mtok": 3, "output_mtok": 15},
                        },
                        {
                            "id": "grok-3-fast",
                            "name": "Grok 3 Fast",
                            "match": {"equals": "grok-3-fast"},
                            "prices": {"input_mtok": 5, "output_mtok": 25},
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    (snapshot_dir / "xai_models_official.json").write_text(
        json.dumps(
            {
                "source_url": "https://docs.x.ai/developers/models?cluster=us-east-1",
                "language_models": [
                    {
                        "name": "grok-3",
                        "aliases": ["grok-3-fast"],
                        "maxPromptLength": 131072,
                        "promptTextTokenPrice": "$n30000",
                        "completionTextTokenPrice": "$n150000",
                    },
                    {
                        "name": "grok-4.20-0309-reasoning",
                        "aliases": ["grok-4.20"],
                        "maxPromptLength": 2000000,
                        "promptTextTokenPrice": "$n20000",
                        "completionTextTokenPrice": "$n60000",
                    },
                ],
                "image_generation_models": [
                    {
                        "name": "grok-imagine-image",
                        "aliases": [],
                        "imagePrice": "$n200000000",
                        "pricePerInputImage": "$n20000000",
                    }
                ],
                "video_generation_models": [
                    {
                        "name": "grok-imagine-video",
                        "aliases": [],
                        "resolutionPricing": [{"pricePerSecond": "$n500000000"}],
                        "pricePerInputImage": "$n20000000",
                        "pricePerInputVideoSecond": "$n100000000",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    registry, report, quarantine = build_registry_module.build_registry_artifacts(
        snapshot_dir=snapshot_dir,
        curated_dir=curated_dir,
    )

    assert "grok-3" in registry["models"]
    assert "grok-3-fast" not in registry["models"]
    assert "grok-4.20" in registry["models"]
    assert registry["models"]["grok-imagine-image"]["modes"] == ["image_generation"]
    assert registry["models"]["grok-imagine-video"]["modes"] == ["video_generation"]
    assert not quarantine
    assert report["summary"]["quarantine_count"] == 0
