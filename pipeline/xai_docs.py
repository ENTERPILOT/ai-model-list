"""Helpers for extracting the official xAI model catalog from docs HTML."""

from __future__ import annotations

import json
from typing import Any


MODEL_TYPE_BY_BUCKET = {
    "language_models": "auth_mgmt.LanguageModel",
    "image_generation_models": "auth_mgmt.ImageGenerationModel",
    "video_generation_models": "auth_mgmt.VideoGenerationModel",
    "audio_models": "auth_mgmt.AudioModel",
}


def build_xai_models_snapshot(html: str, source_url: str) -> dict[str, Any]:
    return {
        "source_url": source_url,
        **{
            bucket_name: _extract_embedded_objects(html, type_name)
            for bucket_name, type_name in MODEL_TYPE_BY_BUCKET.items()
        },
    }


def _extract_embedded_objects(html: str, type_name: str) -> list[dict[str, Any]]:
    marker = _escaped_type_marker(type_name)
    objects: list[dict[str, Any]] = []
    offset = 0

    while True:
        start = html.find(marker, offset)
        if start == -1:
            return objects

        raw_object, offset = _extract_balanced_object(html, start)
        objects.append(json.loads(raw_object.encode("utf-8").decode("unicode_escape")))


def _escaped_type_marker(type_name: str) -> str:
    return '{\\"$typeName\\":\\"' + type_name + '\\"'


def _extract_balanced_object(text: str, start: int) -> tuple[str, int]:
    depth = 0
    for index, char in enumerate(text[start:], start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1], index + 1
    raise ValueError("unbalanced object while parsing xAI docs payload")
