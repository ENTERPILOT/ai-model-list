"""Pipeline loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline.rankings import (
    ARENA_CATALOG_DIRNAME,
    ARENA_CATALOG_METADATA_FILENAME,
    ARENA_LEADERBOARD_FILENAMES,
    ARTIFICIAL_ANALYSIS_DIRNAME,
    ARTIFICIAL_ANALYSIS_METADATA_FILENAME,
    ARTIFICIAL_ANALYSIS_MODELS_FILENAME,
    LIVEBENCH_CATEGORIES_FILENAME,
    LIVEBENCH_DIRNAME,
    LIVEBENCH_METADATA_FILENAME,
    LIVEBENCH_TABLE_FILENAME,
)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_curated_config(curated_dir: Path) -> dict[str, Any]:
    return {
        "providers": _read_json(curated_dir / "providers.json"),
        "source_policies": _read_json(curated_dir / "source_policies.json"),
        "canonical_aliases": _read_json(curated_dir / "canonical_aliases.json"),
        "rejections": _read_json(curated_dir / "rejections.json"),
    }


def load_snapshot_payloads(snapshot_dir: Path) -> dict[str, Any]:
    payloads: dict[str, Any] = {}

    litellm_path = snapshot_dir / "litellm_model_prices.json"
    if litellm_path.exists():
        payloads["litellm"] = _read_json(litellm_path)

    pydantic_genai_path = snapshot_dir / "pydantic_genai_prices.json"
    if pydantic_genai_path.exists():
        payloads["pydantic_genai"] = _read_json(pydantic_genai_path)

    xai_models_official_path = snapshot_dir / "xai_models_official.json"
    if xai_models_official_path.exists():
        payloads["xai_models_official"] = _read_json(xai_models_official_path)

    deepseek_models_official_path = snapshot_dir / "deepseek_models_official.json"
    if deepseek_models_official_path.exists():
        payloads["deepseek_official"] = _read_json(deepseek_models_official_path)

    runway_models_official_path = snapshot_dir / "runway_models_official.json"
    if runway_models_official_path.exists():
        payloads["runway_official"] = _read_json(runway_models_official_path)

    google_speech_models_official_path = snapshot_dir / "google_speech_models_official.json"
    if google_speech_models_official_path.exists():
        payloads["google_speech_official"] = _read_json(google_speech_models_official_path)

    openrouter_path = snapshot_dir / "openrouter_models.json"
    if openrouter_path.exists():
        openrouter_payload = _read_json(openrouter_path)
        payloads["openrouter"] = openrouter_payload.get("data", openrouter_payload)

    llm_prices_path = snapshot_dir / "llm_prices_current.json"
    if llm_prices_path.exists():
        llm_prices_payload = _read_json(llm_prices_path)
        payloads["llm_prices"] = llm_prices_payload.get("prices", llm_prices_payload)
        payloads["llm_prices_metadata"] = llm_prices_payload

    portkey_dir = snapshot_dir / "portkey"
    if portkey_dir.exists():
        payloads["portkey"] = {
            path.name: _read_json(path)
            for path in sorted(portkey_dir.glob("*.json"))
        }

    fetch_metadata_path = snapshot_dir / "fetch_metadata.json"
    if fetch_metadata_path.exists():
        payloads["fetch_metadata"] = _read_json(fetch_metadata_path)

    arena_catalog_dir = snapshot_dir / ARENA_CATALOG_DIRNAME
    if arena_catalog_dir.exists():
        leaderboards = {
            filename: _read_json(arena_catalog_dir / filename)
            for filename in ARENA_LEADERBOARD_FILENAMES
            if (arena_catalog_dir / filename).exists()
        }
        if leaderboards:
            payloads["arena_catalog"] = leaderboards

        arena_metadata_path = arena_catalog_dir / ARENA_CATALOG_METADATA_FILENAME
        if arena_metadata_path.exists():
            payloads["arena_catalog_metadata"] = _read_json(arena_metadata_path)

    artificial_analysis_dir = snapshot_dir / ARTIFICIAL_ANALYSIS_DIRNAME
    if artificial_analysis_dir.exists():
        artificial_analysis_models_path = artificial_analysis_dir / ARTIFICIAL_ANALYSIS_MODELS_FILENAME
        if artificial_analysis_models_path.exists():
            payloads["artificial_analysis"] = _read_json(artificial_analysis_models_path)

        artificial_analysis_metadata_path = artificial_analysis_dir / ARTIFICIAL_ANALYSIS_METADATA_FILENAME
        if artificial_analysis_metadata_path.exists():
            payloads["artificial_analysis_metadata"] = _read_json(artificial_analysis_metadata_path)

    livebench_dir = snapshot_dir / LIVEBENCH_DIRNAME
    if livebench_dir.exists():
        livebench_payload: dict[str, Any] = {}

        livebench_table_path = livebench_dir / LIVEBENCH_TABLE_FILENAME
        if livebench_table_path.exists():
            livebench_payload["table"] = _read_json(livebench_table_path)

        livebench_categories_path = livebench_dir / LIVEBENCH_CATEGORIES_FILENAME
        if livebench_categories_path.exists():
            livebench_payload["categories"] = _read_json(livebench_categories_path)

        if livebench_payload:
            payloads["livebench"] = livebench_payload

        livebench_metadata_path = livebench_dir / LIVEBENCH_METADATA_FILENAME
        if livebench_metadata_path.exists():
            payloads["livebench_metadata"] = _read_json(livebench_metadata_path)

    return payloads
