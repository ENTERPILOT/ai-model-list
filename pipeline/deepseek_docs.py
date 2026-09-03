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
FOOTNOTE_SUFFIX_PATTERN = re.compile(r"(?:\s*(?:\(\d+\)|\[\d+\]|\*+))+\s*$")
LABEL_WHITESPACE_PATTERN = re.compile(r"\s+")

MODEL_LABEL = "MODEL"
MODEL_VERSION_LABEL = "MODEL VERSION"
CONTEXT_LENGTH_LABEL = "CONTEXT LENGTH"
MAX_OUTPUT_LABEL = "MAX OUTPUT"
CACHE_HIT_LABEL = "1M INPUT TOKENS (CACHE HIT)"
CACHE_MISS_LABEL = "1M INPUT TOKENS (CACHE MISS)"
OUTPUT_LABEL = "1M OUTPUT TOKENS"

# DeepSeek splits every pricing row into time-of-day tiers. The peak tier is the
# standard list price, published as the base price; the off-peak tier becomes a
# pricing time window keyed off the peak hours documented below the table.
PREFERRED_PRICING_TIER = "PEAK"
OFF_PEAK_TIER_LABELS = ("OFF-PEAK", "OFF PEAK")
PRICING_TIER_LABELS = (PREFERRED_PRICING_TIER,) + OFF_PEAK_TIER_LABELS
UNTIERED_KEY = ""
OFF_PEAK_WINDOW_LABEL = "off_peak"

TAG_PATTERN = re.compile(r"<[^>]+>")
PEAK_HOURS_PATTERN = re.compile(r"peak hours are(?P<body>.*?)utc", re.IGNORECASE | re.S)
HOUR_RANGE_PATTERN = re.compile(r"(\d{1,2}:\d{2})\s*[-–—~]\s*(\d{1,2}:\d{2})")
MINUTES_PER_DAY = 24 * 60

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
    cache_tiers = _parse_tiered_price_values(rows, CACHE_HIT_LABEL, len(model_ids))
    input_tiers = _parse_tiered_price_values(rows, CACHE_MISS_LABEL, len(model_ids))
    output_tiers = _parse_tiered_price_values(rows, OUTPUT_LABEL, len(model_ids))

    base_tier = _base_tier_key(input_tiers)
    off_peak_ranges = _off_peak_utc_ranges(html_text)

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
                    cache_price=cache_tiers[base_tier][index],
                    input_price=input_tiers[base_tier][index],
                    output_price=output_tiers[base_tier][index],
                    time_windows=_build_time_windows(
                        index,
                        cache_tiers=cache_tiers,
                        input_tiers=input_tiers,
                        output_tiers=output_tiers,
                        off_peak_ranges=off_peak_ranges,
                    ),
                )
                for index in range(len(model_ids))
            ],
        }
    ]


def _base_tier_key(tiers: dict[str, list[float]]) -> str:
    """The tier whose rates are published as the base prices."""
    preferred = _normalize_label(PREFERRED_PRICING_TIER)
    if preferred in tiers:
        return preferred
    return next(iter(tiers))


def _off_peak_tier_key(tiers: dict[str, list[float]]) -> str | None:
    for label in OFF_PEAK_TIER_LABELS:
        normalized = _normalize_label(label)
        if normalized in tiers:
            return normalized
    return None


def _build_time_windows(
    index: int,
    *,
    cache_tiers: dict[str, list[float]],
    input_tiers: dict[str, list[float]],
    output_tiers: dict[str, list[float]],
    off_peak_ranges: list[dict[str, str]],
) -> list[dict[str, Any]]:
    if not off_peak_ranges:
        return []

    rates: dict[str, float] = {}
    for key, tiers in (
        ("input_mtok", input_tiers),
        ("cache_read_mtok", cache_tiers),
        ("output_mtok", output_tiers),
    ):
        base_tier = _base_tier_key(tiers)
        off_peak_tier = _off_peak_tier_key(tiers)
        if off_peak_tier is None or off_peak_tier == base_tier:
            continue
        rates[key] = tiers[off_peak_tier][index]

    if not rates:
        return []

    return [
        {
            "label": OFF_PEAK_WINDOW_LABEL,
            "utc_ranges": off_peak_ranges,
            "prices": rates,
        }
    ]


def _off_peak_utc_ranges(html_text: str) -> list[dict[str, str]]:
    """Daily UTC ranges outside DeepSeek's documented peak hours.

    The hours live in prose below the table rather than in it, so an
    unrecognized footnote drops the window instead of failing the build; the
    peak rates are still published as the base prices.
    """
    peak_match = PEAK_HOURS_PATTERN.search(_document_text(html_text))
    if peak_match is None:
        return []

    peak_minutes: set[int] = set()
    for raw_start, raw_end in HOUR_RANGE_PATTERN.findall(peak_match.group("body")):
        start = _parse_clock_minutes(raw_start)
        end = _parse_clock_minutes(raw_end)
        if start is None or end is None:
            continue
        span = (end - start) % MINUTES_PER_DAY or MINUTES_PER_DAY
        peak_minutes.update((start + offset) % MINUTES_PER_DAY for offset in range(span))

    if not peak_minutes or len(peak_minutes) == MINUTES_PER_DAY:
        return []

    return _minutes_to_ranges(set(range(MINUTES_PER_DAY)) - peak_minutes)


def _minutes_to_ranges(minutes: set[int]) -> list[dict[str, str]]:
    bounds: list[tuple[int, int]] = []
    for minute in sorted(minutes):
        if bounds and bounds[-1][1] == minute:
            bounds[-1] = (bounds[-1][0], minute + 1)
        else:
            bounds.append((minute, minute + 1))

    # A range touching both ends of the day is one window that wraps midnight.
    if len(bounds) > 1 and bounds[0][0] == 0 and bounds[-1][1] == MINUTES_PER_DAY:
        first = bounds.pop(0)
        last = bounds.pop()
        bounds.append((last[0], first[1]))

    return [
        {"start": _format_clock(start), "end": _format_clock(end % MINUTES_PER_DAY)}
        for start, end in sorted(bounds)
    ]


def _parse_clock_minutes(value: str) -> int | None:
    hours, _, minutes = value.partition(":")
    hour = int(hours)
    minute = int(minutes)
    if hour > 24 or minute > 59 or (hour == 24 and minute):
        return None
    return (hour * 60 + minute) % MINUTES_PER_DAY


def _format_clock(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _document_text(html_text: str) -> str:
    text = TAG_PATTERN.sub(" ", html_text)
    return LABEL_WHITESPACE_PATTERN.sub(" ", html.unescape(text).replace("\xa0", " "))


def _extract_pricing_table_rows(html_text: str) -> list[list[str]]:
    required_labels = {
        _normalize_label(MODEL_LABEL),
        _normalize_label(MODEL_VERSION_LABEL),
        _normalize_label(CONTEXT_LENGTH_LABEL),
        _normalize_label(MAX_OUTPUT_LABEL),
    }
    for table_match in TABLE_PATTERN.finditer(html_text):
        rows = _extract_rows(table_match.group("body"))
        if not rows:
            continue
        labels = {_normalize_label(row[0]) for row in rows if row}
        if required_labels <= labels:
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


def _strip_trailing_footnotes(value: str) -> str:
    return FOOTNOTE_SUFFIX_PATTERN.sub("", value.strip()).strip()


def _normalize_label(value: str) -> str:
    normalized = _strip_trailing_footnotes(value)
    return LABEL_WHITESPACE_PATTERN.sub(" ", normalized).upper()


def _cell_matches_label(value: str, label: str) -> bool:
    return _normalize_label(value) == _normalize_label(label)


def _row_values(rows: list[list[str]], label: str) -> list[str]:
    for row in rows:
        if row and _cell_matches_label(row[0], label):
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


def _parse_tiered_price_values(
    rows: list[list[str]], label: str, expected_count: int
) -> dict[str, list[float]]:
    """Prices for one pricing row, keyed by time-of-day tier.

    An untiered row yields a single entry under ``UNTIERED_KEY``.
    """
    for index, row in enumerate(rows):
        if not row:
            continue
        if _cell_matches_label(row[0], label):
            raw_values = row[1:]
            break
        if len(row) >= 2 and _cell_matches_label(row[1], label):
            raw_values = row[2:]
            break
    else:
        raise ValueError(f"unable to locate DeepSeek pricing row '{label}'")

    raw_tiers = _split_pricing_tiers(raw_values, rows[index + 1 :])
    return {
        tier: [_parse_price(value) for value in _expand_shared_values(values, expected_count)]
        for tier, values in raw_tiers.items()
    }


def _is_pricing_tier_label(value: str) -> bool:
    return any(_cell_matches_label(value, tier) for tier in PRICING_TIER_LABELS)


def _split_pricing_tiers(
    raw_values: list[str], following_rows: list[list[str]]
) -> dict[str, list[str]]:
    """Split a tiered pricing row (off-peak/peak) into its per-tier cells."""
    if not raw_values or not _is_pricing_tier_label(raw_values[0]):
        return {UNTIERED_KEY: raw_values}

    tiers: dict[str, list[str]] = {_normalize_label(raw_values[0]): raw_values[1:]}
    for row in following_rows:
        if not row or not _is_pricing_tier_label(row[0]):
            break
        tiers.setdefault(_normalize_label(row[0]), row[1:])
    return tiers


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
    return _strip_trailing_footnotes(model_id)


def _build_model(
    *,
    model_id: str,
    model_version: str,
    context_window: int,
    max_output_tokens: int,
    cache_price: float,
    input_price: float,
    output_price: float,
    time_windows: list[dict[str, Any]],
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
    if time_windows:
        model["prices"]["time_windows"] = time_windows
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
