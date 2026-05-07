"""Helpers for extracting the Ollama Cloud catalog from the search HTML.

Ollama does not publish per-token pricing yet (their /pricing page advertises
subscription plans only), so this parser captures the cloud-eligible model list
from https://ollama.com/search?c=cloud and emits a snapshot in the shape the
``normalize_pydantic_genai_rows`` shim consumes. ``prices`` is intentionally
omitted: the resolver will record provider entries with ``pricing=null`` until
Ollama publishes per-token rates.
"""

from __future__ import annotations

import re
from typing import Any


MODEL_LI_PATTERN = re.compile(
    r"<li[^>]*x-test-model[^>]*>(?P<body>.*?)</li>",
    re.S,
)
MODEL_HREF_PATTERN = re.compile(r'href="/library/(?P<id>[A-Za-z0-9._-]+)"')
TITLE_PATTERN = re.compile(
    r"<span[^>]*x-test-search-response-title[^>]*>\s*(?P<title>[^<]+?)\s*</span>",
)
DESCRIPTION_PATTERN = re.compile(
    r'<p[^>]*class="[^"]*break-words[^"]*"[^>]*>\s*(?P<description>.*?)\s*</p>',
    re.S,
)
CAPABILITY_PATTERN = re.compile(
    r"<span[^>]*x-test-capability[^>]*>\s*(?P<capability>[^<]+?)\s*</span>",
)
CLOUD_BADGE_PATTERN = re.compile(
    r"<span[^>]*bg-cyan-50[^>]*>\s*cloud\s*</span>",
)


def build_ollama_cloud_models_snapshot(html: str, source_url: str) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for li_match in MODEL_LI_PATTERN.finditer(html):
        body = li_match.group("body")
        if CLOUD_BADGE_PATTERN.search(body) is None:
            continue

        href_match = MODEL_HREF_PATTERN.search(body)
        title_match = TITLE_PATTERN.search(body)
        if href_match is None or title_match is None:
            continue

        model_id = href_match.group("id")
        if model_id in seen_ids:
            continue
        seen_ids.add(model_id)

        title = title_match.group("title").strip()
        capabilities = sorted({
            cap_match.group("capability").strip()
            for cap_match in CAPABILITY_PATTERN.finditer(body)
            if cap_match.group("capability").strip()
        })

        model: dict[str, Any] = {
            "id": model_id,
            "name": title,
            "mode": "embedding" if "embedding" in capabilities else "chat",
        }

        description_match = DESCRIPTION_PATTERN.search(body)
        if description_match is not None:
            description = re.sub(r"\s+", " ", description_match.group("description")).strip()
            if description:
                model["description"] = description

        models.append(model)

    return [
        {
            "id": "ollama_cloud",
            "pricing_urls": [source_url],
            "models": models,
        }
    ]
