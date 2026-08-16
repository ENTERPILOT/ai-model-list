"""Helpers for extracting the official Meta Model API catalog from docs Markdown."""

from __future__ import annotations

import re
from typing import Any


MODEL_ID_PATTERN = re.compile(r"`([^`]+)`")
PRICE_PATTERN = re.compile(r"\$(\d+(?:\.\d+)?)")
TOKEN_COUNT_PATTERN = re.compile(r"(\d[\d,]*)\s*tokens", re.IGNORECASE)

PRICE_LABEL_TO_KEY = {
    "input": "input_mtok",
    "cached input": "cache_read_mtok",
    "output": "output_mtok",
}


def build_meta_models_snapshot(
    models_markdown: str,
    pricing_markdown: str | None,
    *,
    models_source_url: str,
    pricing_source_url: str,
) -> list[dict[str, Any]]:
    models = _extract_models(models_markdown)
    if not models:
        raise ValueError("unable to locate Meta Model API available-models table")

    # A missing pricing document (Meta's docs site intermittently stops serving
    # the Markdown variant) leaves pricing to the aggregator sources instead of
    # failing the whole build. A document that is present but no longer parses
    # still raises, so a real table change is not silently dropped.
    if pricing_markdown is not None:
        prices = _extract_prices(pricing_markdown)
        if "input_mtok" not in prices or "output_mtok" not in prices:
            raise ValueError("unable to locate Meta Model API per-token pricing table")

        # Pricing is catalog-wide (one pay-as-you-go table), so every model gets
        # the same prices. Revisit if Meta introduces per-model pricing rows.
        for model in models:
            model["prices"] = dict(prices)

    return [
        {
            "id": "meta",
            "pricing_urls": [pricing_source_url],
            "model_urls": [models_source_url],
            "models": models,
        }
    ]


def _extract_models(markdown: str) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for table in _extract_pipe_tables(markdown):
        header = [cell.lower() for cell in table[0]]
        if "model id" not in header or not any("context window" in cell for cell in header):
            continue
        context_index = next(i for i, cell in enumerate(header) if "context window" in cell)
        for row in table[1:]:
            model_id_match = MODEL_ID_PATTERN.search(row[0])
            if model_id_match is None:
                continue
            model_id = model_id_match.group(1)
            model: dict[str, Any] = {
                "id": model_id,
                "name": _display_name_from_model_id(model_id),
                "mode": "chat",
            }
            context_window = _parse_token_count(row[context_index]) if context_index < len(row) else None
            if context_window is not None:
                model["context_window"] = context_window
            models.append(model)
        if models:
            return models
    return models


def _extract_prices(markdown: str) -> dict[str, float]:
    for table in _extract_pipe_tables(markdown):
        header = " ".join(cell.lower() for cell in table[0])
        if "usage" not in header or "per 1m tokens" not in header:
            continue
        prices: dict[str, float] = {}
        for row in table[1:]:
            if len(row) < 2:
                continue
            key = PRICE_LABEL_TO_KEY.get(row[0].strip().lower())
            price_match = PRICE_PATTERN.search(row[1])
            if key is not None and price_match is not None:
                prices[key] = float(price_match.group(1))
        if prices:
            return prices
    return {}


def _extract_pipe_tables(markdown: str) -> list[list[list[str]]]:
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells):
                continue  # separator row
            current.append(cells)
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


def _parse_token_count(value: str) -> int | None:
    match = TOKEN_COUNT_PATTERN.search(value)
    if match is None:
        return None
    return int(match.group(1).replace(",", ""))


def _display_name_from_model_id(model_id: str) -> str:
    return " ".join(token.title() for token in model_id.split("-"))
