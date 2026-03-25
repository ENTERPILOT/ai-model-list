"""Helpers for extracting the official Runway pricing catalog from docs HTML."""

from __future__ import annotations

import re
from typing import Any


USD_PER_CREDIT = 0.01
SECTION_PATTERN = re.compile(
    r"<h2 id=\"(?P<section>[^\"]+)\">.*?</h2>.*?<table><thead><tr><th>Model</th><th>Pricing</th></tr></thead><tbody>(?P<body>.*?)</tbody></table>",
    re.S,
)
ROW_PATTERN = re.compile(r"<tr><td><code dir=\"auto\">(?P<model>[^<]+)</code></td><td>(?P<pricing>.*?)</td></tr>", re.S)


def build_runway_models_snapshot(html_text: str, source_url: str) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for section_match in SECTION_PATTERN.finditer(html_text):
        section = section_match.group("section")
        for row_match in ROW_PATTERN.finditer(section_match.group("body")):
            model_id = row_match.group("model").strip()
            pricing_text = row_match.group("pricing").strip()
            model = _build_model(section, model_id, pricing_text)
            if model is not None:
                models.append(model)

    return [
        {
            "id": "runway",
            "pricing_urls": [source_url],
            "models": models,
        }
    ]


def _build_model(section: str, model_id: str, pricing_text: str) -> dict[str, Any] | None:
    if "(" in model_id:
        return None
    if not model_id.startswith(("gen", "act_", "gwm")):
        return None

    mode_by_section = {
        "video-generation-pricing": "video_generation",
        "image-generation-pricing": "image_generation",
        "audio-generation-pricing": "audio_speech",
        "real-time-pricing": "realtime",
    }
    mode = mode_by_section.get(section)
    if mode is None:
        return None

    prices: dict[str, float] = {}
    if match := re.search(r"([0-9.]+) credits per second", pricing_text):
        numeric_value = float(match.group(1)) * USD_PER_CREDIT
        if mode in {"video_generation", "audio_speech", "realtime"}:
            prices["per_second_output"] = numeric_value
    elif match := re.search(r"([0-9.]+) credits per 50 characters", pricing_text):
        prices["per_character_input"] = (float(match.group(1)) * USD_PER_CREDIT) / 50
    elif match := re.search(r"([0-9.]+) credits per 720p image, or ([0-9.]+) credits per 1080p image", pricing_text):
        prices["image_generation_prices"] = [
            {"size": "720p", "price": float(match.group(1)) * USD_PER_CREDIT},
            {"size": "1080p", "price": float(match.group(2)) * USD_PER_CREDIT},
        ]
        prices["per_image"] = float(match.group(1)) * USD_PER_CREDIT
    elif match := re.search(r"([0-9.]+) credits per image", pricing_text):
        prices["per_image"] = float(match.group(1)) * USD_PER_CREDIT
    else:
        return None

    return {
        "id": model_id,
        "name": _display_name(model_id),
        "mode": mode,
        "match": {"equals": model_id},
        "prices": prices,
    }


def _display_name(model_id: str) -> str:
    parts = re.split(r"[_-]+", model_id)
    display_parts: list[str] = []
    for part in parts:
        if any(char.isdigit() for char in part):
            display_parts.append(part.replace("gen", "Gen-", 1) if part.startswith("gen") else part)
        else:
            display_parts.append(part.upper() if part.isupper() else part.title())
    return " ".join(display_parts)
