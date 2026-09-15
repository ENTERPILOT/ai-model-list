from pipeline.minimax_docs import build_minimax_models_snapshot


PRICING_MARKDOWN = """
# MiniMax Pricing

| Model | Input (per 1M) | Output (per 1M) | Cached (per 1M) |
|---|---|---|---|
| `MiniMax-M3` | $0.30 | $1.20 | $0.06 |

## Speech

| Model | Price per 1M chars |
|---|---|
| `speech-2.8-hd` | $100 |
"""


def test_build_minimax_models_snapshot_includes_chat_models() -> None:
    payload = build_minimax_models_snapshot(
        PRICING_MARKDOWN,
        "https://platform.minimax.io/docs/guides/pricing-paygo.md",
    )

    assert payload[0]["id"] == "minimax"
    assert payload[0]["pricing_urls"] == ["https://platform.minimax.io/docs/guides/pricing-paygo.md"]

    models = {model["id"]: model for model in payload[0]["models"]}

    assert models["MiniMax-M3"]["mode"] == "chat"
    assert models["MiniMax-M3"]["context_window"] == 1_000_000
    assert models["MiniMax-M3"]["max_output_tokens"] == 524_288
    assert models["MiniMax-M3"]["prices"] == {
        "input_per_mtok": 0.30,
        "output_per_mtok": 1.20,
        "cached_input_per_mtok": 0.06,
    }


def test_build_minimax_models_snapshot_includes_all_categories() -> None:
    payload = build_minimax_models_snapshot(
        PRICING_MARKDOWN,
        "https://platform.minimax.io/docs/guides/pricing-paygo.md",
    )
    models = {model["id"]: model for model in payload[0]["models"]}

    # Chat models
    assert "MiniMax-M3" in models
    assert "M2-her" in models
    assert models["M2-her"]["mode"] == "chat"

    # Speech models
    assert "speech-2.8-hd" in models
    assert models["speech-2.8-hd"]["mode"] == "audio_speech"
    assert models["speech-2.8-hd"]["prices"] is not None

    # Video models
    assert "MiniMax-H3" in models
    assert models["MiniMax-H3"]["mode"] == "video_generation"

    # Image models
    assert "image-01" in models
    assert models["image-01"]["mode"] == "image_generation"

    # Music models (deprecated)
    assert "music-3.0" in models
    assert models["music-3.0"]["deprecation_date"] == "2026-08-20"


def test_build_minimax_models_snapshot_excludes_retired_models() -> None:
    payload = build_minimax_models_snapshot(
        PRICING_MARKDOWN,
        "https://platform.minimax.io/docs/guides/pricing-paygo.md",
    )
    models = {model["id"]: model for model in payload[0]["models"]}

    assert "MiniMax-Text-01" not in models
    assert "MiniMax-VL-01" not in models
    assert "MiniMax-M1" not in models


def test_build_minimax_models_snapshot_unpublished_has_no_pricing() -> None:
    payload = build_minimax_models_snapshot(
        PRICING_MARKDOWN,
        "https://platform.minimax.io/docs/guides/pricing-paygo.md",
    )
    models = {model["id"]: model for model in payload[0]["models"]}

    assert "M2-her" in models
    assert models["M2-her"].get("prices") is None


def test_build_minimax_models_snapshot_speech_has_per_character_pricing() -> None:
    payload = build_minimax_models_snapshot(
        PRICING_MARKDOWN,
        "https://platform.minimax.io/docs/guides/pricing-paygo.md",
    )
    models = {model["id"]: model for model in payload[0]["models"]}

    assert models["speech-2.8-hd"]["prices"]["per_character_input"] == 0.0001
