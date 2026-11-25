import pytest

from pipeline.zai_docs import build_zai_models_snapshot


MODELS_SOURCE_URL = "https://docs.z.ai/guides/llm/glm-5.4.md"
PRICING_SOURCE_URL = "https://docs.z.ai/guides/overview/pricing.md"

MODELS_MARKDOWN = """
# GLM-5.4

> [!TIP]
>
> GLM-5.4 is now available to all GLM Coding Plan users. The Model API will be made available upon completion of the security assessment.

## Overview

**GLM-5.4** is Z.ai's latest flagship model.

> [card] Input Modality
>
> Video / Image / Text / File

> [card] Output Modality
>
> Text

> [card] Context Length
>
> 1M

> [card] Maximum Output Tokens
>
> 128K

## How to Use

#### **GLM Coding Plan**

GLM-5.4 is now available to all GLM Coding Plan users.

## Capability Support

Thinking Mode
：Supports thinking mode for deeper reasoning.

Streaming Messages
：Supports real-time streaming responses.

Function Calling
：Provides powerful tool-calling capabilities.

Context Caching
：Uses an intelligent caching mechanism.

Structured Output
：Supports structured output formats such as JSON.
"""

PRICING_MARKDOWN = """
# Pricing

> This page provides pricing information for Z.AI's models and tools. All prices are in USD.

## Models

### Latest Models

Prices per 1M tokens.

| Model | Input | Cached Input | Cached Input Storage | Output |
| :- | :- | :- | :- | :- |
| GLM-5.3 | \\$1.4 | \\$0.26 | Limited-time Free | \\$4.4 |
| GLM-5.3-Flash | \\$0.15 | \\$0.03 | Limited-time Free | \\$0.50 |

### Image Generation Models

Prices per image.

| Model | Price |
| :- | :- |
| GLM-Image | \\$0.015 |
"""


def test_build_zai_models_snapshot_parses_model_card() -> None:
    payload = build_zai_models_snapshot(
        MODELS_MARKDOWN,
        PRICING_MARKDOWN,
        model_source_url=MODELS_SOURCE_URL,
        pricing_source_url=PRICING_SOURCE_URL,
    )

    assert payload[0]["id"] == "zai"
    assert payload[0]["pricing_urls"] == [PRICING_SOURCE_URL]
    assert payload[0]["model_urls"] == [MODELS_SOURCE_URL]

    # GLM-5.4 is Coding Plan only, so the pricing page lists no rate for it.
    assert payload[0]["models"] == [
        {
            "id": "glm-5.4",
            "name": "GLM 5.4",
            "mode": "chat",
            "context_window": 1_048_576,
            "max_output_tokens": 131_072,
            "modalities": {
                "input": ["video", "image", "text"],
                "output": ["text"],
            },
            "capabilities": {
                "function_calling": True,
                "pdf_input": True,
                "prompt_caching": True,
                "reasoning": True,
                "streaming": True,
                "structured_output": True,
                "vision": True,
            },
        }
    ]


def test_build_zai_models_snapshot_attaches_published_rate() -> None:
    models_markdown = MODELS_MARKDOWN.replace("# GLM-5.4", "# GLM-5.3")
    payload = build_zai_models_snapshot(
        models_markdown,
        PRICING_MARKDOWN,
        model_source_url=MODELS_SOURCE_URL,
        pricing_source_url=PRICING_SOURCE_URL,
    )

    model = payload[0]["models"][0]
    assert model["id"] == "glm-5.3"
    assert model["prices"] == {
        "input_mtok": 1.4,
        "cache_read_mtok": 0.26,
        "output_mtok": 4.4,
    }


def test_build_zai_models_snapshot_tolerates_missing_pricing_page() -> None:
    payload = build_zai_models_snapshot(
        MODELS_MARKDOWN,
        None,
        model_source_url=MODELS_SOURCE_URL,
        pricing_source_url=PRICING_SOURCE_URL,
    )

    assert "prices" not in payload[0]["models"][0]


def test_build_zai_models_snapshot_rejects_page_without_model_card() -> None:
    with pytest.raises(ValueError):
        build_zai_models_snapshot(
            "# GLM-5.4\n\nNo cards here.\n",
            PRICING_MARKDOWN,
            model_source_url=MODELS_SOURCE_URL,
            pricing_source_url=PRICING_SOURCE_URL,
        )
