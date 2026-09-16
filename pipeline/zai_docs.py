"""Helpers for extracting the official Z.ai model catalog from docs Markdown."""

from __future__ import annotations

import re
from typing import Any


PRICE_PATTERN = re.compile(r"\$(\d+(?:\.\d+)?)")
CTX_PATTERN = re.compile(r"(\d[\d,]*)\s*(?:tokens?)", re.IGNORECASE)
MAXOUT_PATTERN = re.compile(r"max\.?\s*output\s*(?:tokens?)?\s*[:\s]*(\d[\d,]*)", re.IGNORECASE)
FREE_PATTERN = re.compile(r"(?:free|0\s*/\s*0\s*/\s*0)", re.IGNORECASE)


def _parse_price(value: str) -> float | None:
    match = PRICE_PATTERN.search(value)
    if match is None:
        return None
    return float(match.group(1))


def _parse_token_count(value: str) -> int | None:
    if not value:
        return None
    return int(value.replace(",", ""))


# ── static fallback catalog ──────────────────────────────────────────

DEFAULT_TEXT_MODEL_METADATA: dict[str, dict[str, Any]] = {
    "glm-5.3": {
        "name": "GLM-5.3",
        "mode": "chat",
        "context_window": 1_048_576,
        "max_output_tokens": 128_000,
    },
    "glm-5.3-flash": {
        "name": "GLM-5.3 Flash",
        "mode": "chat",
        "context_window": 1_048_576,
        "max_output_tokens": 128_000,
    },
    "glm-5.2": {
        "name": "GLM-5.2",
        "mode": "chat",
        "context_window": 1_048_576,
        "max_output_tokens": 128_000,
    },
    "glm-5.1": {
        "name": "GLM-5.1",
        "mode": "chat",
        "context_window": 200_000,
        "max_output_tokens": 128_000,
    },
    "glm-5": {
        "name": "GLM-5",
        "mode": "chat",
        "context_window": 200_000,
        "max_output_tokens": 128_000,
    },
    "glm-4.7": {
        "name": "GLM-4.7",
        "mode": "chat",
        "context_window": 200_000,
        "max_output_tokens": 128_000,
    },
    "glm-4.7-flash": {
        "name": "GLM-4.7 Flash",
        "mode": "chat",
        "context_window": 200_000,
        "max_output_tokens": 128_000,
    },
    "glm-4.7-flashx": {
        "name": "GLM-4.7 FlashX",
        "mode": "chat",
        "context_window": 200_000,
        "max_output_tokens": 128_000,
        "prices": {"input_mtok": 0.07, "output_mtok": 0.40, "cache_read_mtok": 0.01},
    },
    "glm-4.6": {
        "name": "GLM-4.6",
        "mode": "chat",
        "context_window": 200_000,
        "max_output_tokens": 128_000,
    },
    "glm-4.5": {
        "name": "GLM-4.5",
        "mode": "chat",
        "context_window": 128_000,
        "max_output_tokens": 96_000,
    },
    "glm-4.5-x": {
        "name": "GLM-4.5 X",
        "mode": "chat",
        "context_window": 128_000,
        "max_output_tokens": 96_000,
    },
    "glm-4.5-air": {
        "name": "GLM-4.5 Air",
        "mode": "chat",
        "context_window": 128_000,
        "max_output_tokens": 96_000,
    },
    "glm-4.5-airx": {
        "name": "GLM-4.5 AirX",
        "mode": "chat",
        "context_window": 128_000,
        "max_output_tokens": 96_000,
    },
    "glm-4.5-flash": {
        "name": "GLM-4.5 Flash",
        "mode": "chat",
        "context_window": 200_000,
        "max_output_tokens": 96_000,
    },
    "glm-4-32b-0414-128k": {
        "name": "GLM-4-32B-0414-128K",
        "mode": "chat",
        "context_window": 128_000,
        "max_output_tokens": 16_000,
    },
    "glm-4.6v": {
        "name": "GLM-4.6V",
        "mode": "vision",
        "context_window": 128_000,
        "max_output_tokens": 32_000,
    },
    "glm-4.6v-flash": {
        "name": "GLM-4.6V Flash",
        "mode": "vision",
        "context_window": 128_000,
        "max_output_tokens": 32_000,
    },
    "glm-4.6v-flashx": {
        "name": "GLM-4.6V FlashX",
        "mode": "vision",
        "context_window": 128_000,
        "max_output_tokens": 32_000,
        "prices": {"input_mtok": 0.04, "output_mtok": 0.40},
    },
    "glm-4.5v": {
        "name": "GLM-4.5V",
        "mode": "vision",
        "context_window": 64_000,
        "max_output_tokens": 16_000,
    },
    "glm-ocr": {
        "name": "GLM-OCR",
        "mode": "ocr",
        "prices": {"input_mtok": 0.03, "output_mtok": 0.03},
    },
    "glm-image": {
        "name": "GLM-Image",
        "mode": "image_generation",
        "prices": {"per_image": 0.015},
    },
    "cogview-4": {
        "name": "CogView-4",
        "mode": "image_generation",
        "prices": {"per_image": 0.01},
    },
    "cogvideox-3": {
        "name": "CogVideoX-3",
        "mode": "video_generation",
        "prices": {"per_request": 0.20},
    },
    "glm-asr-2512": {
        "name": "GLM-ASR-2512",
        "mode": "audio_transcription",
        "prices": {"input_audio_mtok": 0.03},
    },
}


def build_zai_models_snapshot(
    pricing_markdown: str,
    pricing_source_url: str,
    *,
    model_source_url: str | None = None,
) -> list[dict[str, Any]]:
    rows = _extract_pricing_rows(pricing_markdown)
    models: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for model_id, price_data in rows:
        # Drop rows where all prices are $0 — treated as "free", not priced.
        if price_data and all(v == 0 for v in price_data.values()):
            price_data = None
        metadata = DEFAULT_TEXT_MODEL_METADATA.get(model_id, {})
        model: dict[str, Any] = {
            "id": model_id,
            "name": metadata.get("name", _display_name_from_model_id(model_id)),
            "mode": metadata.get("mode", "chat"),
        }
        if "context_window" in metadata:
            model["context_window"] = metadata["context_window"]
        if "max_output_tokens" in metadata:
            model["max_output_tokens"] = metadata["max_output_tokens"]
        # Scraped prices win; the static catalog fills in when the page's
        # table has no parseable price for a model (per-image, per-video,
        # and audio rows fall outside the input/output column pattern).
        # An empty scraped dict counts as "not priced".
        if not price_data:
            price_data = metadata.get("prices")
        if price_data is not None:
            model["prices"] = price_data
        models.append(model)
        seen_ids.add(model_id)

    # Add any static-catalog models the parser didn't emit (e.g. image gen
    # with per-image pricing that falls outside the input/output column pattern).
    for model_id, metadata in DEFAULT_TEXT_MODEL_METADATA.items():
        if model_id in seen_ids:
            continue
        model: dict[str, Any] = {
            "id": model_id,
            "name": metadata.get("name", _display_name_from_model_id(model_id)),
            "mode": metadata.get("mode", "chat"),
        }
        if "context_window" in metadata:
            model["context_window"] = metadata["context_window"]
        if "max_output_tokens" in metadata:
            model["max_output_tokens"] = metadata["max_output_tokens"]
        if "prices" in metadata:
            model["prices"] = metadata["prices"]
        models.append(model)

    if not models:
        raise ValueError("unable to locate Z.ai model pricing")

    return [
        {
            "id": "zai",
            "pricing_urls": [pricing_source_url],
            "model_urls": [model_source_url] if model_source_url else [],
            "models": models,
        }
    ]


def _extract_pricing_rows(markdown: str) -> list[tuple[str, dict[str, float] | None]]:
    """Parse pricing tables from Z.ai Markdown docs.

    Handles both pipe-table and HTML-table formats. Returns
    [(model_id, prices_dict_or_None)].
    """
    sections = markdown.split("\n\n")
    rows: list[tuple[str, dict[str, float] | None]] = []

    for section in sections:
        stripped = section.strip()
        if not stripped:
            continue
        # pipe tables
        if stripped.startswith("|"):
            pipe_rows = _parse_pipe_table(section)
            if pipe_rows:
                rows.extend(pipe_rows)
        # HTML tables
        if "<table" in stripped.lower():
            rows.extend(_parse_html_table(section))

    # Deduplicate: normalize casing (docs mix "GLM-5.3-Flash" display names
    # with "glm-5.3-flash" IDs), drop separator junk, and prefer the row
    # carrying the most price fields when a model appears twice.
    seen: dict[str, dict[str, float] | None] = {}
    for raw_id, prices in rows:
        model_id = _normalize_model_id(raw_id)
        if model_id is None:
            continue
        existing = seen.get(model_id)
        if existing is None or (prices is not None and len(prices) > len(existing)):
            seen[model_id] = prices
    return list(seen.items())


def _normalize_model_id(raw_id: str) -> str | None:
    """Canonicalize a scraped model cell into a registry ID, or None for junk.

    The docs mix display-name casing ("GLM-5.3-Flash") with real IDs
    ("glm-5.3-flash") and markdown separator rows slip through as cells
    (":------"). Both collapse here.
    """
    model_id = raw_id.strip().strip("`").strip()
    if not model_id:
        return None
    if all(ch in ":- " for ch in model_id):
        return None
    return model_id.lower()


def _parse_pipe_table(text: str) -> list[tuple[str, dict[str, float] | None]]:
    lines = [l.strip() for l in text.splitlines() if l.strip().startswith("|")]
    if len(lines) < 2:
        return []

    # Parse header
    header_cells = [c.strip().strip("*").strip("|").lower() for c in lines[0].strip("|").split("|")]
    header_cells = [c for c in header_cells if c]

    data_rows: list[list[str]] = []
    for line in lines[1:]:
        cells = [c.strip().strip("*").strip("`").strip("|") for c in line.strip("|").split("|")]
        cells = [c for c in cells if c]
        if cells:
            data_rows.append(cells)

    if not data_rows:
        return []

    # Check for chat pricing table pattern
    header_text = " ".join(header_cells).lower()
    results: list[tuple[str, dict[str, float] | None]] = []

    # Pattern 1: standard pricing table with Model, Input, Output, Cached Input
    model_col = None
    input_col = None
    output_col = None
    cache_col = None

    for i, cell in enumerate(header_cells):
        if cell in ("model", "model id", "model name", "model-id", "name"):
            model_col = i
        if "input" in cell and ("cache" not in cell or "hit" in cell or "miss" in cell):
            if input_col is None:
                input_col = i
        if cell.startswith("output"):
            output_col = i
        if "cache" in cell and ("input" in cell or "read" in cell or "hit" in cell):
            cache_col = i

    if model_col is not None and input_col is not None:
        for row in data_rows:
            if model_col >= len(row):
                continue
            model_id = _extract_model_id(row[model_col])
            if not model_id:
                continue
            prices = _extract_prices_from_row(row, input_col, output_col, cache_col)
            results.append((model_id, prices))

    # Pattern 2: vision tables (fewer columns, no cache)
    if not results and model_col is not None:
        # Check if we have input and output columns even without cache
        if input_col is not None and output_col is not None:
            for row in data_rows:
                if model_col >= len(row):
                    continue
                model_id = _extract_model_id(row[model_col])
                if not model_id:
                    continue
                prices = _extract_prices_from_row(row, input_col, output_col, cache_col)
                results.append((model_id, prices))

    # Pattern 3: simple two-column pricing (input, output only)
    if not results:
        for i, cell in enumerate(header_cells):
            if "input" in cell and "cache" not in cell and "output" not in cell:
                input_col = i
            if cell.startswith("output"):
                output_col = i

        # Find model column by position (first col) or by content
        for i, cell in enumerate(header_cells):
            if i not in (input_col, output_col) and i < (input_col if input_col is not None else 0):
                model_col = i
                break

        if model_col is not None and input_col is not None and output_col is not None:
            for row in data_rows:
                if model_col >= len(row) or input_col >= len(row) or output_col >= len(row):
                    continue
                model_id = _extract_model_id(row[model_col])
                if not model_id:
                    continue
                prices = _extract_prices_from_row(row, input_col, output_col, cache_col)
                results.append((model_id, prices))

    # Pattern 4: model name in first column, prices follow
    if not results and len(header_cells) >= 2:
        model_col = 0
        # Find price columns
        price_cols: list[int] = []
        for i, cell in enumerate(header_cells[1:], 1):
            if _looks_like_price_column(cell):
                price_cols.append(i)

        if len(price_cols) >= 2:
            for row in data_rows:
                if len(row) <= model_col:
                    continue
                model_id = _extract_model_id(row[model_col])
                if not model_id:
                    continue
                prices = _extract_prices_from_col_indices(row, price_cols)
                results.append((model_id, prices))

    return results


def _looks_like_price_column(cell: str) -> bool:
    """Check if a header cell looks like it contains prices."""
    lowered = cell.lower()
    price_indicators = ("$price", "$", "per 1m", "price", "input", "output", "cached", "cache")
    return any(ind in lowered for ind in price_indicators)


def _parse_html_table(text: str) -> list[tuple[str, dict[str, float] | None]]:
    import re as _re
    table_pattern = _re.compile(r"<table\b[^>]*>(.*?)</table>", _re.S | _re.I)
    row_pattern = _re.compile(r"<tr\b[^>]*>(.*?)</tr>", _re.S | _re.I)
    cell_pattern = _re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", _re.S | _re.I)

    results: list[tuple[str, dict[str, float] | None]] = []
    for table_match in table_pattern.finditer(text):
        table_body = table_match.group(1)
        tr_matches = row_pattern.findall(table_body)
        if not tr_matches:
            continue

        all_rows: list[list[str]] = []
        for tr in tr_matches:
            cells = []
            for cm in cell_pattern.finditer(tr):
                cell_text = _clean_html_cell(cm.group(1))
                cells.append(cell_text)
            cells = [c for c in cells if c]
            if cells:
                all_rows.append(cells)

        if not all_rows:
            continue

        # Check if this is a pricing table
        header_text = " ".join(all_rows[0]).lower()
        if "input" not in header_text and "output" not in header_text and "$" not in header_text:
            continue

        model_col = None
        input_col = None
        output_col = None
        cache_col = None

        for i, cell in enumerate(all_rows[0]):
            cl = cell.lower()
            if "model" in cl and "id" in cl:
                model_col = i
            elif "model" in cl or cl in ("name", "model name"):
                if model_col is None:
                    model_col = i
            if cl.startswith("output"):
                output_col = i
            if "input" in cl and ("cache" in cl or "hit" in cl or "miss" in cl):
                if "cache" not in cl:
                    input_col = i
                else:
                    cache_col = i
            elif "input" in cl and "cache" not in cl and "hit" not in cl and "miss" not in cl:
                if input_col is None:
                    input_col = i

        if model_col is None:
            # Try first column as model
            model_col = 0

        if input_col is None:
            for i in range(len(all_rows[0])):
                if i != model_col and i != output_col and i != cache_col:
                    if _looks_like_price_column(all_rows[0][i]):
                        input_col = i
                        break

        if output_col is None and input_col is not None:
            for i in range(len(all_rows[0])):
                if i != model_col and i != input_col and i != cache_col:
                    if _looks_like_price_column(all_rows[0][i]):
                        output_col = i
                        break

        for row in all_rows[1:]:
            if model_col >= len(row):
                continue
            model_id = _extract_model_id(row[model_col])
            if not model_id:
                continue
            prices = _extract_prices_from_row(row, input_col, output_col, cache_col)
            results.append((model_id, prices))

    return results


def _clean_html_cell(text: str) -> str:
    import html as _html
    import re as _re
    text = _re.sub(r"(?i)<br\s*/?>", " ", text)
    text = _re.sub(r"<[^>]+>", " ", text)
    text = _html.unescape(text).replace("\xa0", " ")
    text = _re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_model_id(cell: str) -> str | None:
    """Extract model ID from a table cell, stripping backticks."""
    cell = cell.strip()
    cell = cell.strip("`")
    cell = cell.strip()
    if not cell:
        return None
    return cell


def _extract_prices_from_row(
    row: list[str],
    input_col: int | None,
    output_col: int | None,
    cache_col: int | None,
) -> dict[str, float] | None:
    prices: dict[str, float] = {}
    if input_col is not None and input_col < len(row):
        price = _parse_price(row[input_col])
        if price is not None:
            prices["input_mtok"] = price
    if output_col is not None and output_col < len(row):
        price = _parse_price(row[output_col])
        if price is not None:
            prices["output_mtok"] = price
    if cache_col is not None and cache_col < len(row):
        price = _parse_price(row[cache_col])
        if price is not None:
            prices["cache_read_mtok"] = price

    if len(prices) < 2:
        return None
    return prices


def _extract_prices_from_col_indices(
    row: list[str],
    cols: list[int],
) -> dict[str, float] | None:
    prices: dict[str, float] = {}
    for i, col in enumerate(cols):
        if col >= len(row):
            continue
        price = _parse_price(row[col])
        if price is None:
            continue
        key = ["input_mtok", "output_mtok", "cache_read_mtok"][i] if i < 3 else f"_col_{i}"
        if key not in prices:
            prices[key] = price
    return prices if len(prices) >= 2 else None


def _display_name_from_model_id(model_id: str) -> str:
    return model_id
