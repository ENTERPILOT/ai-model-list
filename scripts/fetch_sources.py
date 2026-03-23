#!/usr/bin/env python3
"""Fetch model source snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class SourceDescriptor:
    slug: str
    url: str
    filename: str


SOURCE_DESCRIPTORS: tuple[SourceDescriptor, ...] = (
    SourceDescriptor(
        slug="litellm",
        url="https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
        filename="litellm_model_prices.json",
    ),
    SourceDescriptor(
        slug="openrouter",
        url="https://openrouter.ai/api/v1/models",
        filename="openrouter_models.json",
    ),
    SourceDescriptor(
        slug="llm_prices",
        url="https://www.llm-prices.com/current-v1.json",
        filename="llm_prices_current-v1.json",
    ),
)

SOURCE_URLS = {descriptor.slug: descriptor.url for descriptor in SOURCE_DESCRIPTORS}


def snapshot_path_for_run(base_dir: Path, run_id: str) -> Path:
    return base_dir / "source_snapshots" / run_id


def _fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "ai-model-list/1.0"})
    with urlopen(request) as response:
        return response.read()


def fetch_sources(base_dir: Path, run_id: str | None = None, descriptors: Iterable[SourceDescriptor] = SOURCE_DESCRIPTORS) -> Path:
    run_token = run_id or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    snapshot_dir = snapshot_path_for_run(base_dir, run_token)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    for descriptor in descriptors:
        (snapshot_dir / descriptor.filename).write_bytes(_fetch_bytes(descriptor.url))

    return snapshot_dir

