"""Render registry data into sparse, deterministic output."""

from __future__ import annotations

from typing import Any

INTEGER_VALUE_KEYS = {
    "context_window",
    "max_output_tokens",
    "max_images_per_request",
    "max_audio_length_seconds",
    "max_video_length_seconds",
    "max_videos_per_request",
    "max_audio_per_request",
    "output_vector_size",
    "rpm",
    "tpm",
}


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
        stripped = _strip_nulls(value[key], field_name=key)
        if stripped is not None:
            cleaned[key] = stripped
    return cleaned


def _strip_nulls(value: Any, *, field_name: str | None = None) -> Any:
    if isinstance(value, dict):
        cleaned = {
            key: cleaned
            for key, cleaned in ((key, _strip_nulls(item, field_name=key)) for key, item in value.items())
            if cleaned is not None
        }
        return cleaned or None
    if isinstance(value, list):
        cleaned = [cleaned for cleaned in (_strip_nulls(item) for item in value) if cleaned is not None]
        return cleaned or None
    if isinstance(value, tuple):
        cleaned = tuple(cleaned for cleaned in (_strip_nulls(item) for item in value) if cleaned is not None)
        return cleaned or None
    if field_name in INTEGER_VALUE_KEYS and isinstance(value, float) and value.is_integer():
        return int(value)
    return value
