from pipeline.opencode_zen_docs import build_opencode_zen_models_snapshot


SOURCE_URL = "https://opencode.ai/docs/zen"

SAMPLE_HTML = """
<html><body>
<table><thead><tr><th>Model</th><th>Model ID</th><th>Endpoint</th><th>AI SDK Package</th></tr></thead><tbody>
<tr><td>GPT 5.5</td><td>gpt-5.5</td><td><code dir="auto">https://opencode.ai/zen/v1/responses</code></td><td><code dir="auto">@ai-sdk/openai</code></td></tr>
<tr><td>Claude Opus 4.7</td><td>claude-opus-4-7</td><td><code dir="auto">https://opencode.ai/zen/v1/messages</code></td><td><code dir="auto">@ai-sdk/anthropic</code></td></tr>
<tr><td>Kimi K2.6</td><td>kimi-k2.6</td><td><code dir="auto">https://opencode.ai/zen/v1/chat/completions</code></td><td><code dir="auto">@ai-sdk/openai-compatible</code></td></tr>
<tr><td>Big Pickle</td><td>big-pickle</td><td><code dir="auto">https://opencode.ai/zen/v1/chat/completions</code></td><td><code dir="auto">@ai-sdk/openai-compatible</code></td></tr>
<tr><td>Ling 2.6 Flash</td><td>ling-2.6-flash</td><td><code dir="auto">https://opencode.ai/zen/v1/chat/completions</code></td><td><code dir="auto">@ai-sdk/openai-compatible</code></td></tr>
</tbody></table>

<table><thead><tr><th>Model</th><th>Input</th><th>Output</th><th>Cached Read</th><th>Cached Write</th></tr></thead><tbody>
<tr><td>Big Pickle</td><td>Free</td><td>Free</td><td>Free</td><td>-</td></tr>
<tr><td>Kimi K2.6</td><td>$0.95</td><td>$4.00</td><td>$0.16</td><td>-</td></tr>
<tr><td>Claude Opus 4.7</td><td>$5.00</td><td>$25.00</td><td>$0.50</td><td>$6.25</td></tr>
<tr><td>GPT 5.5 (≤ 272K tokens)</td><td>$5.00</td><td>$30.00</td><td>$0.50</td><td>-</td></tr>
<tr><td>GPT 5.5 (> 272K tokens)</td><td>$10.00</td><td>$45.00</td><td>$1.00</td><td>-</td></tr>
</tbody></table>
</body></html>
"""


def test_build_opencode_zen_models_snapshot_joins_pricing_by_display_name() -> None:
    payload = build_opencode_zen_models_snapshot(SAMPLE_HTML, SOURCE_URL)

    assert payload[0]["id"] == "opencode_zen"
    assert payload[0]["pricing_urls"] == [SOURCE_URL]

    models = {model["id"]: model for model in payload[0]["models"]}
    assert set(models) == {"gpt-5.5", "claude-opus-4-7", "kimi-k2.6", "big-pickle", "ling-2.6-flash"}
    assert models["claude-opus-4-7"]["prices"] == {
        "input_mtok": 5.0,
        "output_mtok": 25.0,
        "cache_read_mtok": 0.5,
        "cache_write_mtok": 6.25,
    }
    assert models["kimi-k2.6"]["prices"] == {
        "input_mtok": 0.95,
        "output_mtok": 4.0,
        "cache_read_mtok": 0.16,
    }
    assert models["big-pickle"]["prices"] == {
        "input_mtok": 0.0,
        "output_mtok": 0.0,
        "cache_read_mtok": 0.0,
    }
    assert "prices" not in models["ling-2.6-flash"]


def test_build_opencode_zen_models_snapshot_uses_lowest_tier_for_threshold_pricing() -> None:
    payload = build_opencode_zen_models_snapshot(SAMPLE_HTML, SOURCE_URL)
    models = {model["id"]: model for model in payload[0]["models"]}
    assert models["gpt-5.5"]["prices"] == {
        "input_mtok": 5.0,
        "output_mtok": 30.0,
        "cache_read_mtok": 0.5,
    }


def test_build_opencode_zen_models_snapshot_assigns_chat_mode() -> None:
    payload = build_opencode_zen_models_snapshot(SAMPLE_HTML, SOURCE_URL)
    for model in payload[0]["models"]:
        assert model["mode"] == "chat"
