"""Helpers for extracting OpenAI's published token prices from its docs Markdown.

``https://developers.openai.com/api/docs/pricing.md`` is the Markdown rendering of
OpenAI's pricing page (every docs page serves one by appending ``.md``). Its
per-tier tables share one shape::

    | Model | Short context input | Short context cached input |
      Short context cache writes | Short context output | Long context ... |

We read the Standard table for the headline rates and the Batch table for the
discounted asynchronous rates, and emit a snapshot in the same shape the
aggregated price feed uses so the existing catalog normalizer can consume it.

Only prices are extracted, and only under the exact model ids the page prints.
The page carries no context windows, modalities or display names, so these
records are a pricing overlay over the model catalog rather than a replacement
for it — see ``normalize_openai_docs_rows``. Inventing id spellings here would
be inventing catalog entries: an ``openai/gpt-4-1`` provider model claims OpenAI
serves that id, which only the vendor can tell us.

Long-context rates are deliberately left to the aggregators: the registry models
those as ``pricing.tiers``, which the shared catalog extractor does not read.
The short-context column is the headline standard rate and is what
``input_per_mtok``/``output_per_mtok`` mean.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping


# Every table on the page sits under an "### ..." heading. Only the two named
# here are read, but each section must be bounded by the *next* heading of any
# kind — the page also renders Flex, Fast mode and a dozen grouped tables, and a
# section that runs past its own table silently picks up the next tier's rates.
SECTION_HEADING_PATTERN = re.compile(r"^###\s+(?P<title>.+?)\s*$", re.M)
TIER_HEADING_PATTERN = re.compile(r"^(?P<tier>Standard|Batch)\s+pricing data$", re.IGNORECASE)
TABLE_ROW_PATTERN = re.compile(r"^\|(?P<cells>.+)\|\s*$", re.M)
SEPARATOR_ROW_PATTERN = re.compile(r"^[\s|:-]+$")
PRICE_PATTERN = re.compile(r"^\$(?P<amount>\d+(?:\.\d+)?)$")
# "gpt-5.5 (<272K context length)" — an annotation on the name, not part of the id.
MODEL_ANNOTATION_PATTERN = re.compile(r"\s*\([^)]*\)\s*$")

# Columns we read, keyed by the header text they sit under.
STANDARD_COLUMNS = {
    "short context input": "input_mtok",
    "short context cached input": "cache_read_mtok",
    "short context cache writes": "cache_write_mtok",
    "short context output": "output_mtok",
}
BATCH_COLUMNS = {
    "short context input": "batch_input_mtok",
    "short context output": "batch_output_mtok",
}
COLUMNS_BY_TIER = {"standard": STANDARD_COLUMNS, "batch": BATCH_COLUMNS}


def build_openai_models_snapshot(pricing_markdown: str, source_url: str) -> list[dict[str, Any]]:
    prices_by_model: dict[str, dict[str, float]] = {}
    for tier, columns in _iter_tier_tables(pricing_markdown):
        for model_id, prices in _parse_pricing_table(tier, columns):
            prices_by_model.setdefault(model_id, {}).update(prices)

    if not prices_by_model:
        raise ValueError("unable to locate OpenAI standard pricing table")

    models = [
        {
            "id": model_id,
            "match": {"or": [{"equals": model_id}]},
            "prices": prices,
        }
        for model_id, prices in sorted(prices_by_model.items())
    ]

    return [
        {
            "id": "openai",
            "pricing_urls": [source_url],
            "models": models,
        }
    ]


def _iter_tier_tables(markdown: str) -> Iterable[tuple[str, Mapping[str, str]]]:
    """Yield each recognized tier table body paired with the columns to read."""
    headings = list(SECTION_HEADING_PATTERN.finditer(markdown))
    for index, heading in enumerate(headings):
        tier_match = TIER_HEADING_PATTERN.match(heading.group("title"))
        if tier_match is None:
            continue
        columns = COLUMNS_BY_TIER.get(tier_match.group("tier").lower())
        if columns is None:
            continue
        end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
        yield markdown[heading.end() : end], columns


def _parse_pricing_table(
    table_markdown: str,
    columns: Mapping[str, str],
) -> Iterable[tuple[str, dict[str, float]]]:
    header: list[str] | None = None
    for match in TABLE_ROW_PATTERN.finditer(table_markdown):
        raw = match.group("cells")
        if SEPARATOR_ROW_PATTERN.fullmatch(raw):
            continue
        cells = [cell.strip() for cell in raw.split("|")]
        if header is None:
            header = [cell.lower() for cell in cells]
            continue

        model_id = _normalize_model_id(cells[0])
        if not model_id:
            continue

        prices = {
            columns[column]: price
            for column, cell in zip(header[1:], cells[1:])
            if column in columns and (price := _parse_price(cell)) is not None
        }
        if prices:
            yield model_id, prices


def _normalize_model_id(cell: str) -> str | None:
    model_id = MODEL_ANNOTATION_PATTERN.sub("", cell.strip().strip("`")).strip()
    if not model_id or model_id.lower() == "model":
        return None
    return model_id


def _parse_price(cell: str) -> float | None:
    match = PRICE_PATTERN.match(cell.strip())
    return float(match.group("amount")) if match else None

