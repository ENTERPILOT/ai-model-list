"""Helpers for extracting the official Kimi model catalog (static).

Moonshot docs are an SPA without public token pricing. We ship a static
catalog with metadata/context windows but no pricing — aggregator sources
fill pricing when available.
"""

from __future__ import annotations

from typing import Any


# ── static catalog ───────────────────────────────────────────────────

CHAT_MODELS: dict[str, dict[str, Any]] = {
    "kimi-for-coding": {
        "name": "Kimi for Coding",
        "mode": "chat",
        "context_window": 262_144,
        "description": "Kimi K2.7 Code. Supports reasoning, vision, and video input.",
    },
    "kimi-for-coding-highspeed": {
        "name": "Kimi for Coding Highspeed",
        "mode": "chat",
        "context_window": 262_144,
        "description": "Speed tier of Kimi for Coding.",
    },
    "k3": {
        "name": "Kimi K3",
        "mode": "chat",
        "context_window": 1_000_000,
    },
    "k3-256k": {
        "name": "Kimi K3 256K",
        "mode": "chat",
        "context_window": 262_144,
    },
}

EMBEDDING_MODELS: dict[str, dict[str, Any]] = {
    "bge_m3_embed": {
        "name": "BGE M3 Embed",
        "mode": "embedding",
        "context_window": 8_192,
    },
}


def build_kimi_models_snapshot(
    pricing_markdown: str | None,
    pricing_source_url: str,
    *,
    model_source_url: str | None = None,
) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []

    for catalog in (CHAT_MODELS, EMBEDDING_MODELS):
        for model_id, metadata in catalog.items():
            model: dict[str, Any] = {
                "id": model_id,
                "name": metadata.get("name", _display_name_from_model_id(model_id)),
                "mode": metadata.get("mode", "chat"),
            }
            for key in ("context_window", "description", "deprecation_date"):
                if key in metadata:
                    model[key] = metadata[key]
            # No pricing — Moonshot docs are an SPA with no public token pricing.
            models.append(model)

    if not models:
        raise ValueError("unable to locate Kimi model catalog")

    return [
        {
            "id": "kimicode",
            "pricing_urls": [pricing_source_url],
            "model_urls": [model_source_url] if model_source_url else [],
            "models": models,
        }
    ]


def _display_name_from_model_id(model_id: str) -> str:
    return model_id
