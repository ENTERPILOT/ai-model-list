from pipeline.kimi_docs import build_kimi_models_snapshot


PRICING_MARKDOWN = None  # Moonshot docs are an SPA with no public token pricing


def test_build_kimi_models_snapshot_includes_chat_models() -> None:
    payload = build_kimi_models_snapshot(
        PRICING_MARKDOWN,
        "https://platform.moonshot.cn/docs/faq",
        model_source_url="https://platform.moonshot.cn/docs/api/chat",
    )

    assert payload[0]["id"] == "kimicode"
    assert payload[0]["model_urls"] == ["https://platform.moonshot.cn/docs/api/chat"]

    models = {model["id"]: model for model in payload[0]["models"]}

    assert models["kimi-for-coding"]["mode"] == "chat"
    assert models["kimi-for-coding"]["context_window"] == 262_144
    assert "Kimi K2.7 Code" in models["kimi-for-coding"]["description"]


def test_build_kimi_models_snapshot_includes_k3() -> None:
    payload = build_kimi_models_snapshot(
        PRICING_MARKDOWN,
        "https://platform.moonshot.cn/docs/faq",
    )
    models = {model["id"]: model for model in payload[0]["models"]}

    assert models["k3"]["mode"] == "chat"
    assert models["k3"]["context_window"] == 1_000_000


def test_build_kimi_models_snapshot_has_no_pricing() -> None:
    payload = build_kimi_models_snapshot(
        PRICING_MARKDOWN,
        "https://platform.moonshot.cn/docs/faq",
    )
    models = {model["id"]: model for model in payload[0]["models"]}

    for model_id, model in models.items():
        assert model.get("prices") is None, f"{model_id} should have no pricing"


def test_build_kimi_models_snapshot_includes_embedding() -> None:
    payload = build_kimi_models_snapshot(
        PRICING_MARKDOWN,
        "https://platform.moonshot.cn/docs/faq",
    )
    models = {model["id"]: model for model in payload[0]["models"]}

    assert models["bge_m3_embed"]["mode"] == "embedding"
    assert models["bge_m3_embed"]["context_window"] == 8_192


def test_build_kimi_models_snapshot_has_static_only() -> None:
    """All kimi data is static — no scraping needed."""
    payload = build_kimi_models_snapshot(
        None,  # markdown not used
        "https://platform.moonshot.cn/docs/faq",
    )

    models = {model["id"]: model for model in payload[0]["models"]}
    expected = {"kimi-for-coding", "kimi-for-coding-highspeed", "k3", "k3-256k", "bge_m3_embed"}
    assert set(models.keys()) == expected
