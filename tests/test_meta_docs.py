import pytest

from pipeline.meta_docs import build_meta_models_snapshot


MODELS_SOURCE_URL = "https://dev.meta.ai/docs/getting-started/models.md"
PRICING_SOURCE_URL = "https://dev.meta.ai/docs/getting-started/pricing-rate-limits.md"

MODELS_MARKDOWN = """
# Models

## Available models {#available-models}

| Model ID | Input modalities | Output modalities | Context window |
| :---- | :---- | :---- | :---- |
| `muse-spark-1.1` | Text, image, video, PDF | Text | 1,048,576 tokens |

### Muse Spark 1.1 {#muse-spark}

**Model ID:** `muse-spark-1.1`

| Context window | Input modalities | Output modalities |
| :---- | :---- | :---- |
| 1,048,576 tokens | Text, image, video, PDF | Text |
"""

PRICING_MARKDOWN = """
# Pricing and rate limits

## Pricing {#pricing}

| Usage | Price per 1M tokens |
| :---- | :---- |
| Input | $1.25 |
| Cached input | $0.15 |
| Output | $4.25 |

## Rate limits {#rate-limits}

| Tier | Requests per minute (RPM) | Tokens per minute (TPM) |
| :---- | :---- | :---- |
| Free | 60 | 2,000,000 |
| Paid | 3,000 | 4,000,000 |
"""


def test_build_meta_models_snapshot_parses_catalog_and_pricing() -> None:
    payload = build_meta_models_snapshot(
        MODELS_MARKDOWN,
        PRICING_MARKDOWN,
        models_source_url=MODELS_SOURCE_URL,
        pricing_source_url=PRICING_SOURCE_URL,
    )

    assert payload[0]["id"] == "meta"
    assert payload[0]["pricing_urls"] == [PRICING_SOURCE_URL]
    assert payload[0]["model_urls"] == [MODELS_SOURCE_URL]

    models = {model["id"]: model for model in payload[0]["models"]}
    assert models["muse-spark-1.1"] == {
        "id": "muse-spark-1.1",
        "name": "Muse Spark 1.1",
        "mode": "chat",
        "context_window": 1_048_576,
        "prices": {
            "input_mtok": 1.25,
            "cache_read_mtok": 0.15,
            "output_mtok": 4.25,
        },
    }


def test_build_meta_models_snapshot_applies_shared_pricing_to_every_model() -> None:
    models_markdown = MODELS_MARKDOWN.replace(
        "| `muse-spark-1.1` | Text, image, video, PDF | Text | 1,048,576 tokens |",
        "| `muse-spark-1.1` | Text, image, video, PDF | Text | 1,048,576 tokens |\n"
        "| `muse-spark-2` | Text | Text | 2,097,152 tokens |",
    )

    payload = build_meta_models_snapshot(
        models_markdown,
        PRICING_MARKDOWN,
        models_source_url=MODELS_SOURCE_URL,
        pricing_source_url=PRICING_SOURCE_URL,
    )

    models = {model["id"]: model for model in payload[0]["models"]}
    assert models["muse-spark-2"]["name"] == "Muse Spark 2"
    assert models["muse-spark-2"]["context_window"] == 2_097_152
    assert models["muse-spark-2"]["prices"] == models["muse-spark-1.1"]["prices"]


def test_build_meta_models_snapshot_ignores_rate_limit_table() -> None:
    payload = build_meta_models_snapshot(
        MODELS_MARKDOWN,
        PRICING_MARKDOWN,
        models_source_url=MODELS_SOURCE_URL,
        pricing_source_url=PRICING_SOURCE_URL,
    )

    prices = payload[0]["models"][0]["prices"]
    assert set(prices) == {"input_mtok", "cache_read_mtok", "output_mtok"}


def test_build_meta_models_snapshot_raises_without_models_table() -> None:
    with pytest.raises(ValueError, match="available-models table"):
        build_meta_models_snapshot(
            "# Models\n\nNo table here.\n",
            PRICING_MARKDOWN,
            models_source_url=MODELS_SOURCE_URL,
            pricing_source_url=PRICING_SOURCE_URL,
        )


def test_build_meta_models_snapshot_raises_without_pricing_table() -> None:
    with pytest.raises(ValueError, match="pricing table"):
        build_meta_models_snapshot(
            MODELS_MARKDOWN,
            "# Pricing\n\nSee sales.\n",
            models_source_url=MODELS_SOURCE_URL,
            pricing_source_url=PRICING_SOURCE_URL,
        )
