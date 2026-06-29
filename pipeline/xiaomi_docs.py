"""Helpers for extracting the official Xiaomi MiMo model catalog from docs Markdown."""

from __future__ import annotations

import html
import re
from typing import Any


TABLE_PATTERN = re.compile(r"<table\b[^>]*>(?P<body>.*?)</table>", re.IGNORECASE | re.S)
ROW_PATTERN = re.compile(r"<tr\b[^>]*>(?P<body>.*?)</tr>", re.IGNORECASE | re.S)
CELL_PATTERN = re.compile(r"<t[dh]\b[^>]*>(?P<body>.*?)</t[dh]>", re.IGNORECASE | re.S)
MODEL_ID_PATTERN = re.compile(r"`([^`]+)`")
PRICE_PATTERN = re.compile(r"\$(\d+(?:\.\d+)?)")

DEFAULT_TEXT_MODEL_METADATA: dict[str, dict[str, Any]] = {
    "mimo-v2.5-pro": {
        "name": "Xiaomi MiMo V2.5 Pro",
        "context_window": 1_000_000,
        "max_output_tokens": 128_000,
    },
    "mimo-v2.5": {
        "name": "Xiaomi MiMo V2.5",
        "context_window": 1_000_000,
        "max_output_tokens": 128_000,
    },
    "mimo-v2-pro": {
        "name": "Xiaomi MiMo V2 Pro",
        "context_window": 1_000_000,
        "max_output_tokens": 128_000,
        "deprecation_date": "2026-06-30",
    },
    "mimo-v2-omni": {
        "name": "Xiaomi MiMo V2 Omni",
        "context_window": 256_000,
        "max_output_tokens": 128_000,
        "deprecation_date": "2026-06-30",
    },
    "mimo-v2-flash": {
        "name": "Xiaomi MiMo V2 Flash",
        "context_window": 256_000,
        "max_output_tokens": 64_000,
        "deprecation_date": "2026-06-30",
    },
}


def build_xiaomi_models_snapshot(
    pricing_markdown: str,
    pricing_source_url: str,
    *,
    model_source_url: str | None = None,
) -> list[dict[str, Any]]:
    rows = _extract_overseas_text_model_pricing_rows(pricing_markdown)
    if not rows:
        raise ValueError("unable to locate Xiaomi MiMo overseas text model pricing")

    return [
        {
            "id": "xiaomi",
            "pricing_urls": [pricing_source_url],
            "model_urls": [model_source_url] if model_source_url else [],
            "models": [
                _build_text_model(
                    model_id=model_id,
                    cache_price=cache_price,
                    input_price=input_price,
                    output_price=output_price,
                )
                for model_id, cache_price, input_price, output_price in rows
            ],
        }
    ]


def _extract_overseas_text_model_pricing_rows(markdown: str) -> list[tuple[str, float, float, float]]:
    _, separator, overseas_section = markdown.partition("### Overseas Pricing")
    if not separator:
        raise ValueError("unable to locate Xiaomi MiMo overseas pricing section")

    rows: list[tuple[str, float, float, float]] = []
    for table_match in TABLE_PATTERN.finditer(overseas_section):
        table_rows = _extract_rows(table_match.group("body"))
        if not table_rows or not _is_text_model_pricing_table(table_rows):
            continue
        for row in table_rows:
            if not row or not row[0].startswith("`mimo-"):
                continue
            model_id_match = MODEL_ID_PATTERN.search(row[0])
            prices = [_parse_price(cell) for cell in row[1:]]
            prices = [price for price in prices if price is not None]
            if model_id_match is not None and len(prices) >= 3:
                rows.append((model_id_match.group(1), prices[0], prices[1], prices[2]))
        if rows:
            return rows

    return rows


def _extract_rows(table_html: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for row_match in ROW_PATTERN.finditer(table_html):
        cells = [
            _normalize_cell_text(cell_match.group("body"))
            for cell_match in CELL_PATTERN.finditer(row_match.group("body"))
        ]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append(cells)
    return rows


def _normalize_cell_text(value: str) -> str:
    normalized = re.sub(r"(?i)<br\s*/?>", "\n", value)
    normalized = re.sub(r"<[^>]+>", " ", normalized)
    normalized = html.unescape(normalized).replace("\xa0", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _is_text_model_pricing_table(rows: list[list[str]]) -> bool:
    header_text = " ".join(cell.upper() for row in rows[:2] for cell in row)
    return (
        "INPUT (CACHE HIT)" in header_text
        and "INPUT (CACHE MISS)" in header_text
        and "OUTPUT" in header_text
    )


def _parse_price(value: str) -> float | None:
    match = PRICE_PATTERN.search(value)
    if match is None:
        return None
    return float(match.group(1))


def _build_text_model(
    *,
    model_id: str,
    cache_price: float,
    input_price: float,
    output_price: float,
) -> dict[str, Any]:
    metadata = DEFAULT_TEXT_MODEL_METADATA.get(model_id, {})
    model: dict[str, Any] = {
        "id": model_id,
        "name": metadata.get("name", _display_name_from_model_id(model_id)),
        "mode": "chat",
        "prices": {
            "input_mtok": input_price,
            "cache_read_mtok": cache_price,
            "output_mtok": output_price,
        },
    }
    for key in ("context_window", "max_output_tokens", "deprecation_date"):
        if key in metadata:
            model[key] = metadata[key]
    return model


def _display_name_from_model_id(model_id: str) -> str:
    return "Xiaomi " + " ".join(token.upper() if token.startswith("v") else token.title() for token in model_id.split("-"))
