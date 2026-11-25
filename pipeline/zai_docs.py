"""Helpers for extracting the official Z.ai flagship model from docs Markdown.

Z.ai documents each flagship GLM release on its own page (``/guides/llm/<model>``)
and keeps per-token prices on the shared pricing page. We scrape the flagship
page for capabilities/limits and the pricing page for its per-token rates, so a
new flagship release is tracked from the vendor even when no reseller lists it
yet. Older GLM models keep coming from the resellers and aggregators.

A flagship that is not on sale as a per-token model yet (Z.ai publishes it to
Coding Plan subscribers first) simply has no pricing attached; the model still
enters the registry so consumers can see its modalities and limits.

The flagship page is pinned in ``scripts/fetch_sources.py``
(``ZAI_MODELS_SOURCE_URL``) and mirrored by ``providers.zai.docs_url``, so a new
flagship release is picked up by bumping those two URLs.
"""

from __future__ import annotations

import re
from typing import Any

CARD_PATTERN = re.compile(r"^>\s*\[card\]\s*(?P<label>[^\n]+?)\s*$")
CARD_VALUE_PATTERN = re.compile(r"^>\s*(?P<value>\S[^\n]*)$")
TITLE_PATTERN = re.compile(r"^#\s+(?P<title>\S[^\n]*)$")
PRICE_PATTERN = re.compile(r"\\?\$(\d+(?:\.\d+)?)")
TOKEN_COUNT_PATTERN = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*([KMB])?", re.IGNORECASE)

CAPABILITY_SECTION_HEADING = "## Capability Support"
CAPABILITY_BY_PHRASE = {
    "thinking mode": "reasoning",
    "streaming messages": "streaming",
    "function calling": "function_calling",
    "context caching": "prompt_caching",
    "structured output": "structured_output",
}

INPUT_MODALITY_BY_LABEL = {
    "text": "text",
    "image": "image",
    "video": "video",
    "audio": "audio",
}
# Z.ai lists documents under "File"; the registry expresses that as a capability.
FILE_LABELS = {"file", "files", "pdf"}

# The pricing tables publish "1M"/"128K" style limits, and the rest of the
# registry spells 1M contexts as 1_048_576, so use binary multipliers.
TOKEN_MULTIPLIERS = {"": 1, "K": 1024, "M": 1024**2, "B": 1024**3}

PRICE_COLUMN_TO_KEY = {
    "input": "input_mtok",
    "cached input": "cache_read_mtok",
    "output": "output_mtok",
}


def build_zai_models_snapshot(
    model_markdown: str,
    pricing_markdown: str | None,
    *,
    model_source_url: str,
    pricing_source_url: str,
) -> list[dict[str, Any]]:
    model = _extract_model_card(model_markdown)
    if model is None:
        raise ValueError("unable to locate Z.ai model page metadata")

    # A missing pricing document leaves the flagship unpriced; a document that
    # is present but no longer parses still raises, so a real table change is
    # not silently dropped.
    if pricing_markdown is not None:
        prices = _extract_per_token_prices(pricing_markdown).get(model["id"])
        if prices:
            model["prices"] = prices

    return [
        {
            "id": "zai",
            "pricing_urls": [pricing_source_url],
            "model_urls": [model_source_url],
            "models": [model],
        }
    ]


def _extract_model_card(markdown: str) -> dict[str, Any] | None:
    title = _extract_title(markdown)
    if title is None:
        return None

    card = _extract_cards(markdown)
    context_window = _parse_token_count(card.get("context length"))
    if context_window is None:
        return None

    model_id = title.lower()
    model: dict[str, Any] = {
        "id": model_id,
        "name": _display_name_from_title(title),
        "mode": "chat",
        "context_window": context_window,
    }

    modalities = _build_modalities(card.get("input modality"), card.get("output modality"))
    if modalities:
        model["modalities"] = modalities

    max_output_tokens = _parse_token_count(card.get("maximum output tokens"))
    if max_output_tokens is not None:
        model["max_output_tokens"] = max_output_tokens

    capabilities = _extract_capabilities(markdown)
    described = card.get("input modality", "").lower()
    if any(label in described for label in FILE_LABELS):
        capabilities["pdf_input"] = True
    if "image" in described or "video" in described:
        capabilities["vision"] = True
    if capabilities:
        model["capabilities"] = dict(sorted(capabilities.items()))

    return model


def _extract_title(markdown: str) -> str | None:
    for line in markdown.splitlines():
        match = TITLE_PATTERN.match(line.strip())
        if match is not None:
            return match.group("title").strip()
    return None


def _extract_cards(markdown: str) -> dict[str, str]:
    cards: dict[str, str] = {}
    lines = [line.strip() for line in markdown.splitlines()]
    for index, line in enumerate(lines):
        label_match = CARD_PATTERN.match(line)
        if label_match is None:
            continue
        for value_line in lines[index + 1 :]:
            if not value_line.startswith(">"):
                break
            value_match = CARD_VALUE_PATTERN.match(value_line)
            if value_match is not None:
                cards[label_match.group("label").strip().lower()] = value_match.group("value").strip()
                break
    return cards


def _build_modalities(input_labels: str | None, output_labels: str | None) -> dict[str, list[str]]:
    modalities: dict[str, list[str]] = {}
    inputs = _ordered_modalities(input_labels)
    outputs = _ordered_modalities(output_labels)
    if inputs:
        modalities["input"] = inputs
    if outputs:
        modalities["output"] = outputs
    return modalities


def _ordered_modalities(labels: str | None) -> list[str]:
    if not labels:
        return []
    seen: list[str] = []
    for label in re.split(r"[/,]", labels):
        modality = INPUT_MODALITY_BY_LABEL.get(label.strip().lower())
        if modality is not None and modality not in seen:
            seen.append(modality)
    return seen


def _extract_capabilities(markdown: str) -> dict[str, bool]:
    section = _extract_section(markdown, CAPABILITY_SECTION_HEADING)
    if not section:
        return {}
    lowered = section.lower()
    return {
        capability: True
        for phrase, capability in CAPABILITY_BY_PHRASE.items()
        if phrase in lowered
    }


def _extract_section(markdown: str, heading: str) -> str:
    lines = markdown.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != heading:
            continue
        body: list[str] = []
        for body_line in lines[index + 1 :]:
            if body_line.startswith("## "):
                break
            body.append(body_line)
        return "\n".join(body)
    return ""


def _extract_per_token_prices(markdown: str) -> dict[str, dict[str, float]]:
    """Map model ids to prices from every per-1M-token pricing table."""
    prices: dict[str, dict[str, float]] = {}
    for table in _extract_pipe_tables(markdown):
        header = [cell.lower() for cell in table[0]]
        if not header or header[0] != "model":
            continue
        column_by_key = {
            PRICE_COLUMN_TO_KEY[cell]: index
            for index, cell in enumerate(header)
            if cell in PRICE_COLUMN_TO_KEY
        }
        if "input_mtok" not in column_by_key or "output_mtok" not in column_by_key:
            continue
        for row in table[1:]:
            model_id = _clean_cell(row[0]).lower()
            if not model_id:
                continue
            model_prices: dict[str, float] = {}
            for key, index in column_by_key.items():
                if index >= len(row):
                    continue
                price_match = PRICE_PATTERN.search(row[index])
                if price_match is not None:
                    model_prices[key] = float(price_match.group(1))
            if model_prices:
                prices[model_id] = model_prices
    return prices


def _clean_cell(value: str) -> str:
    return value.strip().strip("`").strip()


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


def _parse_token_count(value: str | None) -> int | None:
    if not value:
        return None
    match = TOKEN_COUNT_PATTERN.search(value)
    if match is None:
        return None
    multiplier = TOKEN_MULTIPLIERS.get((match.group(2) or "").upper(), 1)
    return int(round(float(match.group(1).replace(",", "")) * multiplier))


def _display_name_from_title(title: str) -> str:
    tokens = []
    for token in title.split("-"):
        if token.lower() in {"glm", "ocr"}:
            tokens.append(token.upper())
        elif token:
            tokens.append(token)
    return " ".join(tokens)
