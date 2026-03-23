import json
from pathlib import Path

from pipeline.loaders import load_curated_config
from scripts.build_registry import build_registry


def test_build_registry_returns_top_level_sections(tmp_path: Path) -> None:
    result = build_registry(snapshot_dir=tmp_path, curated_dir=tmp_path)
    assert set(result.keys()) == {"version", "updated_at", "providers", "models", "provider_models"}


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
