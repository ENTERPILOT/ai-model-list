"""Pipeline loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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

    return payloads
