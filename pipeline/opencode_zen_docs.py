"""Helpers for extracting the OpenCode Zen model catalog from docs HTML.

The docs page at https://opencode.ai/docs/zen renders two tables we care about:
- a "Models" table with columns ``Model | Model ID | Endpoint | AI SDK Package``
- a "Pricing" table with columns ``Model | Input | Output | Cached Read | Cached Write``

The /zen/v1/models JSON endpoint only returns IDs, so the docs page is the
authoritative pricing source. We join the two tables on the display name and
emit a snapshot in the same shape consumed by ``normalize_pydantic_genai_rows``.
"""

from __future__ import annotations

import re
from typing import Any


MODELS_TABLE_PATTERN = re.compile(
    r"<table>\s*<thead>\s*<tr>\s*<th>Model</th>\s*<th>Model ID</th>"
    r"\s*<th>Endpoint</th>\s*<th>AI SDK Package</th>\s*</tr>\s*</thead>"
    r"\s*<tbody>(?P<body>.*?)</tbody>\s*</table>",
    re.S,
)
PRICING_TABLE_PATTERN = re.compile(
    r"<table>\s*<thead>\s*<tr>\s*<th>Model</th>\s*<th>Input</th>"
    r"\s*<th>Output</th>\s*<th>Cached Read</th>\s*<th>Cached Write</th>"
    r"\s*</tr>\s*</thead>\s*<tbody>(?P<body>.*?)</tbody>\s*</table>",
    re.S,
)
MODELS_ROW_PATTERN = re.compile(
    r"<tr>\s*<td>(?P<name>[^<]+)</td>"
    r"\s*<td>(?P<id>[^<]+)</td>"
    r"\s*<td>(?P<endpoint>.*?)</td>"
    r"\s*<td>(?P<package>.*?)</td>\s*</tr>",
    re.S,
)
PRICING_ROW_PATTERN = re.compile(
    r"<tr>\s*<td>(?P<name>[^<]+)</td>"
    r"\s*<td>(?P<input>[^<]+)</td>"
    r"\s*<td>(?P<output>[^<]+)</td>"
    r"\s*<td>(?P<cached_read>[^<]+)</td>"
    r"\s*<td>(?P<cached_write>[^<]+)</td>\s*</tr>",
    re.S,
)
TIER_SUFFIX_PATTERN = re.compile(r"\s*\([^)]*tokens\)\s*$")
ENDPOINT_TEXT_PATTERN = re.compile(r"<code[^>]*>([^<]+)</code>")


def build_opencode_zen_models_snapshot(html: str, source_url: str) -> list[dict[str, Any]]:
    pricing_by_name = _parse_pricing_table(html)
    models: list[dict[str, Any]] = []
    for row in _iter_model_rows(html):
        prices = pricing_by_name.get(row["name"])
        model: dict[str, Any] = {
            "id": row["id"],
            "name": row["name"],
            "mode": _mode_from_endpoint(row["endpoint"]),
        }
        if prices is not None:
            model["prices"] = prices
        models.append(model)

    return [
        {
            "id": "opencode_zen",
            "pricing_urls": [source_url],
            "models": models,
        }
    ]


def _iter_model_rows(html: str):
    match = MODELS_TABLE_PATTERN.search(html)
    if match is None:
        return
    for row_match in MODELS_ROW_PATTERN.finditer(match.group("body")):
        yield {
            "name": row_match.group("name").strip(),
            "id": row_match.group("id").strip(),
            "endpoint": row_match.group("endpoint"),
        }


def _parse_pricing_table(html: str) -> dict[str, dict[str, float]]:
    match = PRICING_TABLE_PATTERN.search(html)
    if match is None:
        return {}

    pricing_by_name: dict[str, dict[str, float]] = {}
    for row_match in PRICING_ROW_PATTERN.finditer(match.group("body")):
        raw_name = row_match.group("name").strip()
        display_name = TIER_SUFFIX_PATTERN.sub("", raw_name).strip()
        if display_name in pricing_by_name:
            continue
        prices = {
            "input_mtok": _parse_dollar(row_match.group("input")),
            "output_mtok": _parse_dollar(row_match.group("output")),
            "cache_read_mtok": _parse_dollar(row_match.group("cached_read")),
            "cache_write_mtok": _parse_dollar(row_match.group("cached_write")),
        }
        pricing_by_name[display_name] = {key: value for key, value in prices.items() if value is not None}
    return pricing_by_name


def _parse_dollar(value: str) -> float | None:
    cleaned = value.strip()
    if not cleaned or cleaned in {"-", "—"}:
        return None
    if cleaned.lower() == "free":
        return 0.0
    if cleaned.startswith("$"):
        cleaned = cleaned[1:]
    try:
        return float(cleaned)
    except ValueError:
        return None


def _mode_from_endpoint(endpoint_html: str) -> str:
    match = ENDPOINT_TEXT_PATTERN.search(endpoint_html)
    if match is None:
        return "chat"
    url = match.group(1)
    if "/responses" in url:
        return "chat"
    if "/messages" in url:
        return "chat"
    if "/chat/completions" in url:
        return "chat"
    if "/models/gemini-" in url:
        return "chat"
    return "chat"
