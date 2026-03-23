"""Render registry data into sparse, deterministic output."""

from __future__ import annotations

from typing import Any


def render_registry(resolved: dict[str, Any], updated_at: str) -> dict[str, Any]:
    return {
        "version": 1,
        "updated_at": updated_at,
        "providers": _sorted_clean_mapping(resolved.get("providers", {})),
        "models": _sorted_clean_mapping(resolved.get("models", {})),
        "provider_models": _sorted_clean_mapping(resolved.get("provider_models", {})),
    }


def _sorted_clean_mapping(value: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key in sorted(value):
        stripped = _strip_nulls(value[key])
        if stripped is not None:
            cleaned[key] = stripped
    return cleaned


def _strip_nulls(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {
            key: cleaned
            for key, cleaned in ((key, _strip_nulls(item)) for key, item in value.items())
            if cleaned is not None
        }
        return cleaned or None
    if isinstance(value, list):
        cleaned = [cleaned for cleaned in (_strip_nulls(item) for item in value) if cleaned is not None]
        return cleaned or None
    if isinstance(value, tuple):
        cleaned = tuple(cleaned for cleaned in (_strip_nulls(item) for item in value) if cleaned is not None)
        return cleaned or None
    return value
