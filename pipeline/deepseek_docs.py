"""Helpers for extracting the official DeepSeek pricing catalog from docs HTML."""

from __future__ import annotations

import html
import re
from typing import Any


TABLE_PATTERN = re.compile(r"<table\b[^>]*>(?P<body>.*?)</table>", re.IGNORECASE | re.S)
ROW_PATTERN = re.compile(r"<tr\b[^>]*>(?P<body>.*?)</tr>", re.IGNORECASE | re.S)
CELL_PATTERN = re.compile(r"<t[dh]\b[^>]*>(?P<body>.*?)</t[dh]>", re.IGNORECASE | re.S)
PRICE_PATTERN = re.compile(r"\$(\d+(?:\.\d+)?)")
TOKEN_COUNT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*([kKmM])")

MODEL_LABEL = "MODEL"
MODEL_VERSION_LABEL = "MODEL VERSION"
CONTEXT_LENGTH_LABEL = "CONTEXT LENGTH"
MAX_OUTPUT_LABEL = "MAX OUTPUT"
CACHE_HIT_LABEL = "1M INPUT TOKENS (CACHE HIT)"
CACHE_MISS_LABEL = "1M INPUT TOKENS (CACHE MISS)"
OUTPUT_LABEL = "1M OUTPUT TOKENS"

DEPRECATED_EXACT_ALIASES = {
    "deepseek-v4-flash": ("deepseek-chat", "deepseek-reasoner"),
}


def build_deepseek_models_snapshot(html_text: str, source_url: str) -> list[dict[str, Any]]:
    rows = _extract_pricing_table_rows(html_text)
    model_ids = [_normalize_model_id(value) for value in _row_values(rows, MODEL_LABEL)]
    if not model_ids:
        raise ValueError("unable to locate DeepSeek model identifiers")

    model_versions = _expand_shared_values(_row_values(rows, MODEL_VERSION_LABEL), len(model_ids))
    context_window = _parse_token_count(_shared_row_value(rows, CONTEXT_LENGTH_LABEL))
    max_outputs = [
        _parse_max_output_tokens(value)
        for value in _expand_shared_values(_row_values(rows, MAX_OUTPUT_LABEL), len(model_ids))
    ]
    cache_prices = _parse_price_values(rows, CACHE_HIT_LABEL, len(model_ids))
    input_prices = _parse_price_values(rows, CACHE_MISS_LABEL, len(model_ids))
    output_prices = _parse_price_values(rows, OUTPUT_LABEL, len(model_ids))

    return [
        {
            "id": "deepseek",
            "pricing_urls": [source_url],
            "models": [
                _build_model(
                    model_id=model_ids[index],
                    model_version=model_versions[index],
                    context_window=context_window,
                    max_output_tokens=max_outputs[index],
                    cache_price=cache_prices[index],
                    input_price=input_prices[index],
                    output_price=output_prices[index],
                )
                for index in range(len(model_ids))
            ],
        }
    ]


def _extract_pricing_table_rows(html_text: str) -> list[list[str]]:
    for table_match in TABLE_PATTERN.finditer(html_text):
        rows = _extract_rows(table_match.group("body"))
        if not rows:
            continue
        labels = {row[0] for row in rows if row}
        if {MODEL_LABEL, MODEL_VERSION_LABEL, CONTEXT_LENGTH_LABEL, MAX_OUTPUT_LABEL} <= labels:
            return rows
    raise ValueError("unable to locate DeepSeek pricing table")


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
    normalized = normalized.replace("（", "(").replace("）", ")")
    normalized = re.sub(r"[ \t\r\f\v]+", " ", normalized)
    normalized = re.sub(r" *\n *", "\n", normalized)
    return normalized.strip()


def _row_values(rows: list[list[str]], label: str) -> list[str]:
    for row in rows:
        if row and row[0] == label:
            return row[1:]
    raise ValueError(f"unable to locate DeepSeek '{label}' row")


def _shared_row_value(rows: list[list[str]], label: str) -> str:
    values = _row_values(rows, label)
    if not values:
        raise ValueError(f"DeepSeek '{label}' row is empty")
    return values[0]


def _expand_shared_values(values: list[str], expected_count: int) -> list[str]:
    if len(values) == expected_count:
        return values
    if len(values) == 1 and expected_count > 1:
        return values * expected_count
    raise ValueError("DeepSeek pricing row shape does not match model columns")


def _parse_price_values(rows: list[list[str]], label: str, expected_count: int) -> list[float]:
    for row in rows:
        if not row:
            continue
        if row[0] == label:
            raw_values = row[1:]
            break
        if len(row) >= 2 and row[1] == label:
            raw_values = row[2:]
            break
    else:
        raise ValueError(f"unable to locate DeepSeek pricing row '{label}'")

    return [_parse_price(value) for value in _expand_shared_values(raw_values, expected_count)]


def _parse_price(value: str) -> float:
    match = PRICE_PATTERN.search(value)
    if match is None:
        raise ValueError(f"unable to parse DeepSeek price from {value!r}")
    return float(match.group(1))


def _parse_token_count(value: str) -> int:
    match = TOKEN_COUNT_PATTERN.search(value)
    if match is None:
        raise ValueError(f"unable to parse DeepSeek token count from {value!r}")
    magnitude = 1_000 if match.group(2).upper() == "K" else 1_000_000
    return int(float(match.group(1)) * magnitude)


def _parse_max_output_tokens(value: str) -> int:
    maximum_match = re.search(r"MAXIMUM:\s*(\d+(?:\.\d+)?\s*[kKmM])", value)
    if maximum_match is not None:
        return _parse_token_count(maximum_match.group(1))

    counts = TOKEN_COUNT_PATTERN.findall(value)
    if counts:
        return max(_parse_token_count(f"{amount}{suffix}") for amount, suffix in counts)

    raise ValueError(f"unable to parse DeepSeek max output from {value!r}")


def _normalize_model_id(model_id: str) -> str:
    return re.sub(r"\s*\*+\s*$", "", model_id.strip())


def _build_model(
    *,
    model_id: str,
    model_version: str,
    context_window: int,
    max_output_tokens: int,
    cache_price: float,
    input_price: float,
    output_price: float,
) -> dict[str, Any]:
    match_entries = [{"equals": alias} for alias in DEPRECATED_EXACT_ALIASES.get(model_id, ())]

    model: dict[str, Any] = {
        "id": model_id,
        "name": _display_name_from_model_id(model_id),
        "description": model_version,
        "context_window": context_window,
        "max_output_tokens": max_output_tokens,
        "mode": "chat",
        "prices": {
            "input_mtok": input_price,
            "cache_read_mtok": cache_price,
            "output_mtok": output_price,
        },
    }
    if match_entries:
        model["match"] = {"or": match_entries}
    return model


def _display_name_from_model_id(model_id: str) -> str:
    tokens = [token for token in model_id.split("-") if token]
    display_tokens: list[str] = []
    for token in tokens:
        lowered = token.lower()
        if lowered == "deepseek":
            display_tokens.append("DeepSeek")
        elif re.fullmatch(r"v\d+(?:\.\d+)?", lowered):
            display_tokens.append(lowered.upper())
        elif token.replace(".", "", 1).isdigit():
            display_tokens.append(token)
        else:
            display_tokens.append(token.title())
    return " ".join(display_tokens) if display_tokens else model_id
