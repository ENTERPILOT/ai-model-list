#!/usr/bin/env python3
"""Fetch model source snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time
from typing import Iterable
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class SourceDescriptor:
    slug: str
    url: str
    filename: str


GITHUB_SNAPSHOT_BASE_URL = "https://raw.githubusercontent.com/ENTERPILOT/ai-model-price-list/main/sources"
TOP_LEVEL_SOURCE_FILES: tuple[tuple[str, str], ...] = (
    ("fetch-metadata", "fetch_metadata.json"),
    ("litellm", "litellm_model_prices.json"),
    ("llm_prices", "llm_prices_current.json"),
    ("openrouter", "openrouter_models.json"),
    ("pydantic-genai-prices", "pydantic_genai_prices.json"),
)
PORTKEY_SOURCE_FILES: tuple[str, ...] = (
    "anthropic.json",
    "azure-openai.json",
    "bedrock.json",
    "cohere.json",
    "deepseek.json",
    "fireworks-ai.json",
    "google.json",
    "groq.json",
    "mistral-ai.json",
    "openai.json",
    "together-ai.json",
)


def _github_source_url(relative_path: str) -> str:
    return f"{GITHUB_SNAPSHOT_BASE_URL}/{relative_path}"


SOURCE_DESCRIPTORS: tuple[SourceDescriptor, ...] = (
    tuple(
        SourceDescriptor(
            slug=slug,
            url=_github_source_url(filename),
            filename=filename,
        )
        for slug, filename in TOP_LEVEL_SOURCE_FILES
    )
    + tuple(
        SourceDescriptor(
            slug=f"portkey-{filename.removesuffix('.json')}",
            url=_github_source_url(f"portkey/{filename}"),
            filename=f"portkey/{filename}",
        )
        for filename in PORTKEY_SOURCE_FILES
    )
)

SOURCE_URLS = {descriptor.slug: descriptor.url for descriptor in SOURCE_DESCRIPTORS}
DEFAULT_FETCH_TIMEOUT_SECONDS = 30.0
DEFAULT_FETCH_RETRIES = 3
DEFAULT_RETRY_DELAY_SECONDS = 2.0


def snapshot_path_for_run(base_dir: Path, run_id: str) -> Path:
    return base_dir / "source_snapshots" / run_id


def _fetch_bytes(
    url: str,
    *,
    timeout: float = DEFAULT_FETCH_TIMEOUT_SECONDS,
    retries: int = DEFAULT_FETCH_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY_SECONDS,
) -> bytes:
    request = Request(url, headers={"User-Agent": "ai-model-list/1.0"})
    for attempt in range(1, retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except (OSError, TimeoutError):
            if attempt == retries:
                raise
            time.sleep(retry_delay)

    raise RuntimeError("exhausted fetch retries without raising")


def fetch_sources_to(
    snapshot_dir: Path,
    descriptors: Iterable[SourceDescriptor] = SOURCE_DESCRIPTORS,
) -> Path:
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    for descriptor in descriptors:
        output_path = snapshot_dir / descriptor.filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(_fetch_bytes(descriptor.url))

    return snapshot_dir


def fetch_sources(
    base_dir: Path,
    run_id: str | None = None,
    descriptors: Iterable[SourceDescriptor] = SOURCE_DESCRIPTORS,
) -> Path:
    run_token = run_id or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    snapshot_dir = snapshot_path_for_run(base_dir, run_token)
    return fetch_sources_to(snapshot_dir, descriptors=descriptors)
