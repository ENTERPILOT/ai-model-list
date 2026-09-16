from pipeline.zai_docs import build_zai_models_snapshot


PRICING_MARKDOWN = """
# Z.ai Pricing

## Chat models

| Model | Input (per 1M) | Output (per 1M) | Cached Input (per 1M) |
|---|---|---|---|
| `glm-5.3-flash` | $0.15 | $0.50 | $0.03 |
| `glm-4.7-flash` | $0 | $0 | $0 |
| `glm-4.7-flashx` | $0.07 | $0.40 | $0.01 |

## Vision models

| Model | Input (per 1M) | Output (per 1M) |
|---|---|---|
| `glm-4.6v-flash` | $0 | $0 |

## Image generation

| Model | Price |
|---|---|
| `glm-image` | $0.015 per image |
| `cogview-4` | $0.01 per image |
"""


def test_build_zai_models_snapshot_parses_chat_pricing() -> None:
    payload = build_zai_models_snapshot(
        PRICING_MARKDOWN,
        "https://docs.z.ai/guides/overview/pricing",
    )

    assert payload[0]["id"] == "zai"
    assert payload[0]["pricing_urls"] == ["https://docs.z.ai/guides/overview/pricing"]

    models = {model["id"]: model for model in payload[0]["models"]}

    assert models["glm-5.3-flash"]["prices"] == {
        "input_per_mtok": 0.15,
        "output_per_mtok": 0.50,
        "cached_input_per_mtok": 0.03,
    }

    assert models["glm-5.3-flash"]["context_window"] == 1_048_576
    assert models["glm-5.3-flash"]["max_output_tokens"] == 128_000
    assert models["glm-5.3-flash"]["mode"] == "chat"


def test_build_zai_models_snapshot_handles_free_model() -> None:
    models = {model["id"]: model for model in build_zai_models_snapshot(
        PRICING_MARKDOWN,
        "https://docs.z.ai/guides/overview/pricing",
    )[0]["models"]}

    assert models["glm-4.7-flash"].get("prices") is None


def test_build_zai_models_snapshot_includes_static_metadata() -> None:
    payload = build_zai_models_snapshot(
        PRICING_MARKDOWN,
        "https://docs.z.ai/guides/overview/pricing",
    )
    models = {model["id"]: model for model in payload[0]["models"]}

    # Model without pricing row still gets static metadata
    assert models["glm-image"]["mode"] == "image_generation"
    assert models["cogview-4"]["mode"] == "image_generation"


def test_build_zai_models_snapshot_still_emits_static_models() -> None:
    """Even with unparseable markdown the static fallback produces models."""
    payload = build_zai_models_snapshot(
        "No pricing data here.\n\nNo tables.",
        "https://docs.z.ai/guides/overview/pricing",
    )

    models = {model["id"]: model for model in payload[0]["models"]}
    assert len(models) > 0
