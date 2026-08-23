"""Helpers for extracting the official xAI model catalog from docs HTML."""

from __future__ import annotations

import json
from typing import Any


# Current docs embed the catalog as a plain JSON assignment; every model list
# is grouped per serving cluster under ``clusterConfigs``.
PUBLIC_MODELS_MARKER = "globalThis.__XAI_PUBLIC_MODELS__="

# Snapshot bucket -> key inside each ``clusterConfigs`` entry.
PUBLIC_MODELS_BUCKETS = {
    "language_models": "languageModels",
    "image_generation_models": "imageGenerationModels",
    "video_generation_models": "videoGenerationModels",
    "audio_models": "audioModels",
}

# Legacy docs streamed one escaped object per model, tagged by protobuf type.
MODEL_TYPE_BY_BUCKET = {
    "language_models": "auth_mgmt.LanguageModel",
    "image_generation_models": "auth_mgmt.ImageGenerationModel",
    "video_generation_models": "auth_mgmt.VideoGenerationModel",
    "audio_models": "auth_mgmt.AudioModel",
}


def build_xai_models_snapshot(html: str, source_url: str) -> dict[str, Any]:
    buckets = _extract_public_models(html)
    if buckets is None:
        buckets = {
            bucket_name: _extract_embedded_objects(html, type_name)
            for bucket_name, type_name in MODEL_TYPE_BY_BUCKET.items()
        }
    return {"source_url": source_url, **buckets}


def _extract_public_models(html: str) -> dict[str, list[dict[str, Any]]] | None:
    start = html.find(PUBLIC_MODELS_MARKER)
    if start == -1:
        return None

    raw_object, _ = _extract_balanced_object(html, start + len(PUBLIC_MODELS_MARKER))
    payload = json.loads(raw_object)
    cluster_configs = payload.get("clusterConfigs") if isinstance(payload, dict) else None
    if not isinstance(cluster_configs, list):
        return None

    # The same model is listed once per cluster it is served from; keep the
    # first occurrence so prices and aliases stay stable across clusters.
    buckets: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in PUBLIC_MODELS_BUCKETS}
    seen: dict[str, set[str]] = {bucket: set() for bucket in PUBLIC_MODELS_BUCKETS}
    for config in cluster_configs:
        if not isinstance(config, dict):
            continue
        for bucket_name, payload_key in PUBLIC_MODELS_BUCKETS.items():
            for model in config.get(payload_key) or []:
                if not isinstance(model, dict):
                    continue
                name = model.get("name")
                if not isinstance(name, str) or name in seen[bucket_name]:
                    continue
                seen[bucket_name].add(name)
                buckets[bucket_name].append(model)
    return buckets


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
