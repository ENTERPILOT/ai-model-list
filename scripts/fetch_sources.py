#!/usr/bin/env python3
"""Fetch model source snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Iterable
from urllib.request import Request, urlopen

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.deepseek_docs import build_deepseek_models_snapshot
from pipeline.google_speech_docs import build_google_speech_models_snapshot
from pipeline.rankings import (
    ARENA_CATALOG_DIRNAME,
    ARENA_CATALOG_METADATA_FILENAME,
    ARENA_LEADERBOARD_FILENAMES,
    ARENA_SOURCE_URLS,
)
from pipeline.runway_docs import build_runway_models_snapshot
from pipeline.xai_docs import build_xai_models_snapshot


@dataclass(frozen=True)
class SourceDescriptor:
    slug: str
    url: str
    filename: str


GITHUB_SNAPSHOT_BASE_URL = "https://raw.githubusercontent.com/ENTERPILOT/ai-model-price-list/main/sources"
XAI_MODELS_SOURCE_URL = "https://docs.x.ai/developers/models?cluster=us-east-1"
XAI_MODELS_SOURCE_FILENAME = "xai_models_official.json"
DEEPSEEK_MODELS_SOURCE_URL = "https://api-docs.deepseek.com/quick_start/pricing"
DEEPSEEK_MODELS_SOURCE_FILENAME = "deepseek_models_official.json"
RUNWAY_MODELS_SOURCE_URL = "https://docs.dev.runwayml.com/guides/pricing/"
RUNWAY_MODELS_SOURCE_FILENAME = "runway_models_official.json"
GOOGLE_SPEECH_SOURCE_URL = "https://cloud.google.com/speech-to-text/pricing"
GOOGLE_SPEECH_SOURCE_FILENAME = "google_speech_models_official.json"
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

    xai_models_html = _fetch_bytes(XAI_MODELS_SOURCE_URL).decode("utf-8")
    xai_models_payload = build_xai_models_snapshot(xai_models_html, XAI_MODELS_SOURCE_URL)
    (snapshot_dir / XAI_MODELS_SOURCE_FILENAME).write_text(
        json.dumps(xai_models_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    deepseek_models_html = _fetch_bytes(DEEPSEEK_MODELS_SOURCE_URL).decode("utf-8")
    deepseek_models_payload = build_deepseek_models_snapshot(deepseek_models_html, DEEPSEEK_MODELS_SOURCE_URL)
    (snapshot_dir / DEEPSEEK_MODELS_SOURCE_FILENAME).write_text(
        json.dumps(deepseek_models_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    runway_models_html = _fetch_bytes(RUNWAY_MODELS_SOURCE_URL).decode("utf-8")
    runway_models_payload = build_runway_models_snapshot(runway_models_html, RUNWAY_MODELS_SOURCE_URL)
    (snapshot_dir / RUNWAY_MODELS_SOURCE_FILENAME).write_text(
        json.dumps(runway_models_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    google_speech_html = _fetch_bytes(GOOGLE_SPEECH_SOURCE_URL).decode("utf-8")
    google_speech_payload = build_google_speech_models_snapshot(google_speech_html, GOOGLE_SPEECH_SOURCE_URL)
    (snapshot_dir / GOOGLE_SPEECH_SOURCE_FILENAME).write_text(
        json.dumps(google_speech_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    arena_catalog_dir = snapshot_dir / ARENA_CATALOG_DIRNAME
    arena_catalog_dir.mkdir(parents=True, exist_ok=True)
    arena_fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for filename in ARENA_LEADERBOARD_FILENAMES:
        (arena_catalog_dir / filename).write_bytes(_fetch_bytes(ARENA_SOURCE_URLS[filename]))
    (arena_catalog_dir / ARENA_CATALOG_METADATA_FILENAME).write_text(
        json.dumps(
            {
                "fetched_at": arena_fetched_at,
                "sources": ARENA_SOURCE_URLS,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return snapshot_dir


def fetch_sources(
    base_dir: Path,
    run_id: str | None = None,
    descriptors: Iterable[SourceDescriptor] = SOURCE_DESCRIPTORS,
) -> Path:
    run_token = run_id or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    snapshot_dir = snapshot_path_for_run(base_dir, run_token)
    return fetch_sources_to(snapshot_dir, descriptors=descriptors)
