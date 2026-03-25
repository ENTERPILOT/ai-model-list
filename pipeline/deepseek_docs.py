"""Helpers for extracting the official DeepSeek pricing catalog from docs HTML."""

from __future__ import annotations

import html
import re
from typing import Any


TABLE_PATTERN = re.compile(
    r"<tr><td colspan=\"2\">MODEL</td><td>(?P<chat_id>[^<]+)</td><td>(?P<reasoner_id>[^<]+)</td></tr>"
    r".*?<tr><td colspan=\"2\">MODEL VERSION</td><td>(?P<chat_version>.*?)</td><td>(?P<reasoner_version>.*?)</td></tr>"
    r".*?<tr><td colspan=\"2\">CONTEXT LENGTH</td><td colspan=\"2\">(?P<context>\d+)K</td></tr>"
    r".*?<tr><td colspan=\"2\">MAX OUTPUT</td><td>DEFAULT: (?P<chat_default>\d+)K<br>MAXIMUM: (?P<chat_max>\d+)K</td>"
    r"<td>DEFAULT: (?P<reasoner_default>\d+)K<br>MAXIMUM: (?P<reasoner_max>\d+)K</td></tr>"
    r".*?<tr><td rowspan=\"3\">PRICING</td><td>1M INPUT TOKENS \(CACHE HIT\)</td><td colspan=\"2\">\$(?P<cache>[0-9.]+)</td></tr>"
    r"<tr><td>1M INPUT TOKENS \(CACHE MISS\)</td><td colspan=\"2\">\$(?P<input>[0-9.]+)</td></tr>"
    r"<tr><td>1M OUTPUT TOKENS</td><td colspan=\"2\">\$(?P<output>[0-9.]+)</td></tr>",
    re.S,
)


def build_deepseek_models_snapshot(html_text: str, source_url: str) -> list[dict[str, Any]]:
    match = TABLE_PATTERN.search(html_text)
    if match is None:
        raise ValueError("unable to locate DeepSeek pricing table")

    context_window = int(match.group("context")) * 1_000
    cache_price = float(match.group("cache"))
    input_price = float(match.group("input"))
    output_price = float(match.group("output"))

    return [
        {
            "id": "deepseek",
            "pricing_urls": [source_url],
            "models": [
                _build_model(
                    model_id=match.group("chat_id"),
                    model_version=match.group("chat_version"),
                    context_window=context_window,
                    max_output_tokens=int(match.group("chat_max")) * 1_000,
                    cache_price=cache_price,
                    input_price=input_price,
                    output_price=output_price,
                    reasoning=False,
                ),
                _build_model(
                    model_id=match.group("reasoner_id"),
                    model_version=match.group("reasoner_version"),
                    context_window=context_window,
                    max_output_tokens=int(match.group("reasoner_max")) * 1_000,
                    cache_price=cache_price,
                    input_price=input_price,
                    output_price=output_price,
                    reasoning=True,
                ),
            ],
        }
    ]


def _build_model(
    *,
    model_id: str,
    model_version: str,
    context_window: int,
    max_output_tokens: int,
    cache_price: float,
    input_price: float,
    output_price: float,
    reasoning: bool,
) -> dict[str, Any]:
    normalized_id = model_id.strip()
    match_entries: list[dict[str, str]] = [{"equals": normalized_id}]
    if normalized_id == "deepseek-chat":
        match_entries.append({"starts_with": "deepseek-chat"})
    if normalized_id == "deepseek-reasoner":
        match_entries.extend(
            [
                {"starts_with": "deepseek-reasoner"},
                {"starts_with": "deepseek-r1"},
            ]
        )

    capabilities = {
        "supports_json_mode": True,
        "supports_function_calling": True,
    }
    if not reasoning:
        capabilities["supports_reasoning"] = False
    else:
        capabilities["supports_reasoning"] = True

    return {
        "id": normalized_id,
        "name": "DeepSeek Chat" if normalized_id == "deepseek-chat" else "DeepSeek Reasoner",
        "description": _strip_tags(model_version),
        "context_window": context_window,
        "max_output_tokens": max_output_tokens,
        "mode": "chat",
        "match": {"or": match_entries},
        "prices": {
            "input_mtok": input_price,
            "cache_read_mtok": cache_price,
            "output_mtok": output_price,
        },
        **capabilities,
    }


def _strip_tags(value: str) -> str:
    return re.sub(r"<[^>]+>", " ", html.unescape(value)).replace("\xa0", " ").strip()
