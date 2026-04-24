from pipeline.deepseek_docs import build_deepseek_models_snapshot


SOURCE_URL = "https://api-docs.deepseek.com/quick_start/pricing"

LEGACY_PRICING_HTML = """
<html>
  <body>
    <table style="text-align:center">
      <tr><td colspan="2">MODEL</td><td>deepseek-chat</td><td>deepseek-reasoner</td></tr>
      <tr><td colspan="2">MODEL VERSION</td><td>DeepSeek-V3</td><td>DeepSeek-R1</td></tr>
      <tr><td colspan="2">CONTEXT LENGTH</td><td colspan="2">128K</td></tr>
      <tr><td colspan="2">MAX OUTPUT</td><td>DEFAULT: 8K<br>MAXIMUM: 8K</td><td>DEFAULT: 32K<br>MAXIMUM: 64K</td></tr>
      <tr><td rowspan="3">PRICING</td><td>1M INPUT TOKENS (CACHE HIT)</td><td colspan="2">$0.028</td></tr>
      <tr><td>1M INPUT TOKENS (CACHE MISS)</td><td colspan="2">$0.28</td></tr>
      <tr><td>1M OUTPUT TOKENS</td><td colspan="2">$0.42</td></tr>
    </table>
  </body>
</html>
"""

CURRENT_PRICING_HTML = """
<html>
  <body>
    <table><tr><td>Not the pricing table</td></tr></table>
    <table style="text-align:center">
      <tr><td colspan="2" style="text-align:center">MODEL</td><td>deepseek-v4-flash<sup>*</sup></td><td>deepseek-v4-pro</td></tr>
      <tr><td colspan="2">BASE URL (OpenAI Format)</td><td colspan="2">https://api.deepseek.com</td></tr>
      <tr><td colspan="2" style="text-align:center">MODEL VERSION</td><td>DeepSeek-V4-Flash</td><td>DeepSeek-V4-Pro</td></tr>
      <tr><td colspan="2">THINKING MODE</td><td colspan="2">Supports both non-thinking and thinking (default) modes<br>See Thinking Mode for how to switch</td></tr>
      <tr><td colspan="2">CONTEXT LENGTH</td><td colspan="2">1M</td></tr>
      <tr><td colspan="2">MAX OUTPUT</td><td colspan="2">MAXIMUM: 384K</td></tr>
      <tr><td rowspan="4">FEATURES</td><td>Json Output</td><td>✓</td><td>✓</td></tr>
      <tr><td>Tool Calls</td><td>✓</td><td>✓</td></tr>
      <tr><td>Chat Prefix Completion (Beta)</td><td>✓</td><td>✓</td></tr>
      <tr><td>FIM Completion (Beta)</td><td>Non-thinking mode only</td><td>Non-thinking mode only</td></tr>
      <tr><td rowspan="3">PRICING</td><td>1M INPUT TOKENS (CACHE HIT)</td><td>$0.028</td><td>$0.145</td></tr>
      <tr><td>1M INPUT TOKENS (CACHE MISS)</td><td>$0.14</td><td>$1.74</td></tr>
      <tr><td>1M OUTPUT TOKENS</td><td>$0.28</td><td>$3.48</td></tr>
    </table>
    <p>* The model names <code>deepseek-chat</code> and <code>deepseek-reasoner</code> will be deprecated in the future.</p>
  </body>
</html>
"""


def test_build_deepseek_models_snapshot_parses_legacy_pricing_table() -> None:
    payload = build_deepseek_models_snapshot(LEGACY_PRICING_HTML, SOURCE_URL)

    assert payload[0]["id"] == "deepseek"
    assert payload[0]["pricing_urls"] == [SOURCE_URL]

    models = {model["id"]: model for model in payload[0]["models"]}
    assert set(models) == {"deepseek-chat", "deepseek-reasoner"}
    assert models["deepseek-chat"] == {
        "id": "deepseek-chat",
        "name": "DeepSeek Chat",
        "description": "DeepSeek-V3",
        "context_window": 128_000,
        "max_output_tokens": 8_000,
        "mode": "chat",
        "prices": {
            "input_mtok": 0.28,
            "cache_read_mtok": 0.028,
            "output_mtok": 0.42,
        },
    }
    assert models["deepseek-reasoner"] == {
        "id": "deepseek-reasoner",
        "name": "DeepSeek Reasoner",
        "description": "DeepSeek-R1",
        "context_window": 128_000,
        "max_output_tokens": 64_000,
        "mode": "chat",
        "prices": {
            "input_mtok": 0.28,
            "cache_read_mtok": 0.028,
            "output_mtok": 0.42,
        },
    }


def test_build_deepseek_models_snapshot_parses_current_pricing_table() -> None:
    payload = build_deepseek_models_snapshot(CURRENT_PRICING_HTML, SOURCE_URL)

    models = {model["id"]: model for model in payload[0]["models"]}
    assert set(models) == {"deepseek-v4-flash", "deepseek-v4-pro"}
    assert models["deepseek-v4-flash"] == {
        "id": "deepseek-v4-flash",
        "name": "DeepSeek V4 Flash",
        "description": "DeepSeek-V4-Flash",
        "context_window": 1_000_000,
        "max_output_tokens": 384_000,
        "mode": "chat",
        "match": {
            "or": [
                {"equals": "deepseek-chat"},
                {"equals": "deepseek-reasoner"},
            ]
        },
        "prices": {
            "input_mtok": 0.14,
            "cache_read_mtok": 0.028,
            "output_mtok": 0.28,
        },
    }
    assert models["deepseek-v4-pro"] == {
        "id": "deepseek-v4-pro",
        "name": "DeepSeek V4 Pro",
        "description": "DeepSeek-V4-Pro",
        "context_window": 1_000_000,
        "max_output_tokens": 384_000,
        "mode": "chat",
        "prices": {
            "input_mtok": 1.74,
            "cache_read_mtok": 0.145,
            "output_mtok": 3.48,
        },
    }
