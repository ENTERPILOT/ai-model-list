"""Helpers for extracting the official MiniMax model catalog from docs Markdown."""

from __future__ import annotations

import re
from typing import Any


PRICE_PATTERN = re.compile(r"\$(\d+(?:\.\d+)?)")
CTX_PATTERN = re.compile(r"(\d[\d,]*)\s*(?:tokens?)", re.IGNORECASE)


def _parse_price(value: str) -> float | None:
    match = PRICE_PATTERN.search(value)
    if match is None:
        return None
    return float(match.group(1))


# ── static fallback catalog ──────────────────────────────────────────

CHAT_MODELS: dict[str, dict[str, Any]] = {
    "MiniMax-M3": {
        "name": "MiniMax-M3",
        "mode": "chat",
        "context_window": 1_000_000,
        "max_output_tokens": 524_288,
        "prices": {
            "input_per_mtok": 0.30,
            "output_per_mtok": 1.20,
            "cached_input_per_mtok": 0.06,
        },
        "description": "MiniMax-M3 supports multimodal input.",
    },
    "MiniMax-M2.7": {
        "name": "MiniMax-M2.7",
        "mode": "chat",
        "context_window": 204_800,
        "max_output_tokens": 204_800,
        "prices": {
            "input_per_mtok": 0.30,
            "output_per_mtok": 1.20,
            "cached_input_per_mtok": 0.06,
            "cache_write_per_mtok": 0.375,
        },
    },
    "MiniMax-M2.7-highspeed": {
        "name": "MiniMax-M2.7 Highspeed",
        "mode": "chat",
        "context_window": 204_800,
        "max_output_tokens": 204_800,
        "prices": {
            "input_per_mtok": 0.60,
            "output_per_mtok": 2.40,
            "cached_input_per_mtok": 0.06,
            "cache_write_per_mtok": 0.375,
        },
    },
    "MiniMax-M2.5": {
        "name": "MiniMax-M2.5",
        "mode": "chat",
        "context_window": 204_800,
        "max_output_tokens": 204_800,
        "prices": {
            "input_per_mtok": 0.30,
            "output_per_mtok": 1.20,
            "cached_input_per_mtok": 0.03,
            "cache_write_per_mtok": 0.375,
        },
    },
    "MiniMax-M2.5-highspeed": {
        "name": "MiniMax-M2.5 Highspeed",
        "mode": "chat",
        "context_window": 204_800,
        "max_output_tokens": 204_800,
        "prices": {
            "input_per_mtok": 0.60,
            "output_per_mtok": 2.40,
            "cached_input_per_mtok": 0.03,
            "cache_write_per_mtok": 0.375,
        },
    },
    "MiniMax-M2.1": {
        "name": "MiniMax-M2.1",
        "mode": "chat",
        "context_window": 204_800,
        "max_output_tokens": 204_800,
        "prices": {
            "input_per_mtok": 0.30,
            "output_per_mtok": 1.20,
            "cached_input_per_mtok": 0.03,
            "cache_write_per_mtok": 0.375,
        },
    },
    "MiniMax-M2.1-highspeed": {
        "name": "MiniMax-M2.1 Highspeed",
        "mode": "chat",
        "context_window": 204_800,
        "max_output_tokens": 204_800,
        "prices": {
            "input_per_mtok": 0.60,
            "output_per_mtok": 2.40,
            "cached_input_per_mtok": 0.03,
            "cache_write_per_mtok": 0.375,
        },
    },
    "MiniMax-M2": {
        "name": "MiniMax-M2",
        "mode": "chat",
        "context_window": 204_800,
        "max_output_tokens": 131_072,
        "prices": {
            "input_per_mtok": 0.30,
            "output_per_mtok": 1.20,
            "cached_input_per_mtok": 0.03,
            "cache_write_per_mtok": 0.375,
        },
    },
    "M2-her": {
        "name": "M2-Her",
        "mode": "chat",
        "context_window": 65_536,
    },
}

SPEECH_MODELS: dict[str, dict[str, Any]] = {
    "speech-2.8-hd": {
        "name": "Speech-2.8 HD",
        "mode": "audio_speech",
        "prices": {"per_character_input": 0.0001},
    },
    "speech-2.8-turbo": {
        "name": "Speech-2.8 Turbo",
        "mode": "audio_speech",
        "prices": {"per_character_input": 6e-05},
    },
    "speech-2.6-hd": {
        "name": "Speech-2.6 HD",
        "mode": "audio_speech",
        "prices": {"per_character_input": 0.0001},
    },
    "speech-2.6-turbo": {
        "name": "Speech-2.6 Turbo",
        "mode": "audio_speech",
        "prices": {"per_character_input": 6e-05},
    },
    "speech-02-hd": {
        "name": "Speech-02 HD",
        "mode": "audio_speech",
        "prices": {"per_character_input": 0.0001},
    },
    "speech-02-turbo": {
        "name": "Speech-02 Turbo",
        "mode": "audio_speech",
        "prices": {"per_character_input": 6e-05},
    },
    "speech-01-hd": {
        "name": "Speech-01 HD",
        "mode": "audio_speech",
    },
    "speech-01-turbo": {
        "name": "Speech-01 Turbo",
        "mode": "audio_speech",
    },
}

VIDEO_MODELS: dict[str, dict[str, Any]] = {
    "MiniMax-H3": {
        "name": "MiniMax H3",
        "mode": "video_generation",
        "prices": {"per_second_output": 0.08},
    },
    "MiniMax-H3-Max": {
        "name": "MiniMax H3 Max",
        "mode": "video_generation",
        "prices": {"per_second_output": 0.08},
    },
    "MiniMax-Hailuo-2.3": {
        "name": "MiniMax Hailuo 2.3",
        "mode": "video_generation",
        "prices": {"per_image": 0.28},
    },
    "MiniMax-Hailuo-2.3-Fast": {
        "name": "MiniMax Hailuo 2.3 Fast",
        "mode": "video_generation",
        "prices": {"per_image": 0.19},
    },
    "MiniMax-Hailuo-02": {
        "name": "MiniMax Hailuo 02",
        "mode": "video_generation",
        "prices": {"per_image": 0.10},
    },
    "T2V-01-Director": {
        "name": "T2V-01 Director",
        "mode": "video_generation",
    },
    "T2V-01": {
        "name": "T2V-01",
        "mode": "video_generation",
    },
}

IMAGE_MODELS: dict[str, dict[str, Any]] = {
    "image-01": {
        "name": "MiniMax Image 01",
        "mode": "image_generation",
        "prices": {"per_image": 0.0035},
    },
}

MUSIC_MODELS: dict[str, dict[str, Any]] = {
    "music-3.0": {
        "name": "MiniMax Music 3.0",
        "mode": "music_generation",
        "deprecation_date": "2026-08-20",
        "prices": {"per_request": 0.15},
    },
    "music-2.6": {
        "name": "MiniMax Music 2.6",
        "mode": "music_generation",
        "deprecation_date": "2026-08-20",
        "prices": {"per_request": 0.15},
    },
    "music-cover": {
        "name": "MiniMax Music Cover",
        "mode": "music_generation",
        "deprecation_date": "2026-08-20",
        "prices": {"per_request": 0.15},
    },
}


def build_minimax_models_snapshot(
    pricing_markdown: str,
    pricing_source_url: str,
    *,
    model_source_url: str | None = None,
) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []

    # Merge all categories
    for catalog in (CHAT_MODELS, SPEECH_MODELS, VIDEO_MODELS, IMAGE_MODELS, MUSIC_MODELS):
        for model_id, metadata in catalog.items():
            model: dict[str, Any] = {
                "id": model_id,
                "name": metadata.get("name", _display_name_from_model_id(model_id)),
                "mode": metadata.get("mode", "chat"),
            }
            for key in ("context_window", "max_output_tokens", "description", "deprecation_date"):
                if key in metadata:
                    model[key] = metadata[key]
            if "prices" in metadata:
                model["prices"] = metadata["prices"]
            models.append(model)

    if not models:
        raise ValueError("unable to locate MiniMax model catalog")

    return [
        {
            "id": "minimax",
            "pricing_urls": [pricing_source_url],
            "model_urls": [model_source_url] if model_source_url else [],
            "models": models,
        }
    ]


def _display_name_from_model_id(model_id: str) -> str:
    return model_id