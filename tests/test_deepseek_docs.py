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
      <tr><td colspan="2" style="text-align:center">MODEL</td><td>deepseek-v4-flash<sup>(1)</sup></td><td>deepseek-v4-pro</td></tr>
      <tr><td colspan="2">BASE URL (OpenAI Format)</td><td colspan="2">https://api.deepseek.com</td></tr>
      <tr><td colspan="2" style="text-align:center">MODEL VERSION</td><td>DeepSeek-V4-Flash</td><td>DeepSeek-V4-Pro</td></tr>
      <tr><td colspan="2">THINKING MODE</td><td colspan="2">Supports both non-thinking and thinking (default) modes<br>See Thinking Mode for how to switch</td></tr>
      <tr><td colspan="2">CONTEXT LENGTH</td><td colspan="2">1M</td></tr>
      <tr><td colspan="2">MAX OUTPUT</td><td colspan="2">MAXIMUM: 384K</td></tr>
      <tr><td rowspan="4">FEATURES</td><td>Json Output</td><td>✓</td><td>✓</td></tr>
      <tr><td>Tool Calls</td><td>✓</td><td>✓</td></tr>
      <tr><td>Chat Prefix Completion (Beta)</td><td>✓</td><td>✓</td></tr>
      <tr><td>FIM Completion (Beta)</td><td>Non-thinking mode only</td><td>Non-thinking mode only</td></tr>
      <tr><td rowspan="3">PRICING</td><td>1M INPUT TOKENS (CACHE HIT)<sup>(2)</sup></td><td>$0.0028</td><td>$0.003625 (limited-time 75% off<sup>(3)</sup>)<del>$0.0145</del></td></tr>
      <tr><td>1M INPUT TOKENS (CACHE MISS)</td><td>$0.14</td><td>$0.435 (limited-time 75% off<sup>(3)</sup>)<del>$1.74</del></td></tr>
      <tr><td>1M OUTPUT TOKENS</td><td>$0.28</td><td>$0.87 (limited-time 75% off<sup>(3)</sup>)<del>$3.48</del></td></tr>
    </table>
    <p>(1) The model names <code>deepseek-chat</code> and <code>deepseek-reasoner</code> will be deprecated in the future.</p>
  </body>
</html>
"""


TIERED_PRICING_HTML = """
<html>
  <body>
    <table style="text-align:center">
      <tr><td colspan="2">MODEL</td><td>deepseek-v4-flash<sup>(1)</sup></td><td>deepseek-v4-pro</td></tr>
      <tr><td colspan="2">MODEL VERSION</td><td>DeepSeek-V4-Flash-0731</td><td>DeepSeek-V4-Pro-0813</td></tr>
      <tr><td colspan="2">CONTEXT LENGTH</td><td colspan="2">1M</td></tr>
      <tr><td colspan="2">MAX OUTPUT</td><td colspan="2">MAXIMUM: 384K</td></tr>
      <tr><td rowspan="6">PRICING<sup>(1)</sup></td><td rowspan="2">1M INPUT TOKENS (CACHE HIT)</td><td>OFF-PEAK</td><td>$0.007</td><td>$0.022</td></tr>
      <tr><td>PEAK</td><td>$0.014</td><td>$0.044</td></tr>
      <tr><td rowspan="2">1M INPUT TOKENS (CACHE MISS)</td><td>OFF-PEAK</td><td>$0.22</td><td>$0.66</td></tr>
      <tr><td>PEAK</td><td>$0.44</td><td>$1.32</td></tr>
      <tr><td rowspan="2">1M OUTPUT TOKENS</td><td>OFF-PEAK</td><td>$0.66</td><td>$1.98</td></tr>
      <tr><td>PEAK</td><td>$1.32</td><td>$3.96</td></tr>
    </table>
    <p>(1) Off-peak rates are half of the peak rates. Peak hours are 01:00 - 04:00 and 06:00 - 10:00 UTC, Monday through Friday (all other hours are off-peak).</p>
    <p>(2) Images sent to deepseek-v4-flash are billed as input tokens.</p>
  </body>
</html>
"""

PEAK_HOURS_SENTENCE = "Peak hours are 01:00 - 04:00 and 06:00 - 10:00 UTC, Monday through Friday"

TIERED_PRICING_HTML_EVERY_DAY = TIERED_PRICING_HTML.replace(
    PEAK_HOURS_SENTENCE, "Peak hours are 01:00 - 04:00 and 06:00 - 10:00 UTC"
)

TIERED_PRICING_HTML_WITHOUT_HOURS = TIERED_PRICING_HTML.replace(PEAK_HOURS_SENTENCE, "Peak hours vary by region")

WEEKDAY_OFF_PEAK_RANGES = [
    {"days": ["mon", "tue", "wed", "thu", "fri"], "start": "00:00", "end": "01:00"},
    {"days": ["mon", "tue", "wed", "thu", "fri"], "start": "04:00", "end": "06:00"},
    {"days": ["mon", "tue", "wed", "thu", "fri"], "start": "10:00", "end": "24:00"},
    {"days": ["sat", "sun"], "start": "00:00", "end": "24:00"},
]


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
            "cache_read_mtok": 0.0028,
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
            "input_mtok": 0.435,
            "cache_read_mtok": 0.003625,
            "output_mtok": 0.87,
        },
    }


def test_build_deepseek_models_snapshot_uses_peak_tier_pricing() -> None:
    payload = build_deepseek_models_snapshot(TIERED_PRICING_HTML, SOURCE_URL)

    models = {model["id"]: model for model in payload[0]["models"]}
    assert set(models) == {"deepseek-v4-flash", "deepseek-v4-pro"}
    flash_prices = models["deepseek-v4-flash"]["prices"]
    pro_prices = models["deepseek-v4-pro"]["prices"]
    assert {key: flash_prices[key] for key in ("input_mtok", "cache_read_mtok", "output_mtok")} == {
        "input_mtok": 0.44,
        "cache_read_mtok": 0.014,
        "output_mtok": 1.32,
    }
    assert {key: pro_prices[key] for key in ("input_mtok", "cache_read_mtok", "output_mtok")} == {
        "input_mtok": 1.32,
        "cache_read_mtok": 0.044,
        "output_mtok": 3.96,
    }
    assert models["deepseek-v4-flash"]["context_window"] == 1_000_000
    assert models["deepseek-v4-flash"]["max_output_tokens"] == 384_000


def test_build_deepseek_models_snapshot_emits_off_peak_time_window() -> None:
    payload = build_deepseek_models_snapshot(TIERED_PRICING_HTML, SOURCE_URL)

    models = {model["id"]: model for model in payload[0]["models"]}
    # Peak hours are 01:00-04:00 and 06:00-10:00 on weekdays, so off-peak is
    # the complement on weekdays plus the whole weekend.
    assert models["deepseek-v4-flash"]["prices"]["time_windows"] == [
        {
            "label": "off_peak",
            "utc_ranges": WEEKDAY_OFF_PEAK_RANGES,
            "prices": {
                "input_mtok": 0.22,
                "cache_read_mtok": 0.007,
                "output_mtok": 0.66,
            },
        }
    ]
    assert models["deepseek-v4-pro"]["prices"]["time_windows"][0]["prices"] == {
        "input_mtok": 0.66,
        "cache_read_mtok": 0.022,
        "output_mtok": 1.98,
    }


def test_build_deepseek_models_snapshot_off_peak_window_without_weekdays_applies_every_day() -> None:
    payload = build_deepseek_models_snapshot(TIERED_PRICING_HTML_EVERY_DAY, SOURCE_URL)

    models = {model["id"]: model for model in payload[0]["models"]}
    # No weekday restriction: the complement is daily, with the range covering
    # midnight expressed as a single wrapping window.
    assert models["deepseek-v4-flash"]["prices"]["time_windows"][0]["utc_ranges"] == [
        {"start": "04:00", "end": "06:00"},
        {"start": "10:00", "end": "01:00"},
    ]


def test_build_deepseek_models_snapshot_reads_peak_ranges_after_a_mid_sentence_utc() -> None:
    html_text = TIERED_PRICING_HTML.replace(
        PEAK_HOURS_SENTENCE, "Peak hours are 01:00 - 04:00 UTC and 06:00 - 10:00 UTC, Monday to Friday"
    )
    payload = build_deepseek_models_snapshot(html_text, SOURCE_URL)

    models = {model["id"]: model for model in payload[0]["models"]}
    # The second peak range must not leak into the off-peak window.
    assert models["deepseek-v4-flash"]["prices"]["time_windows"][0]["utc_ranges"] == WEEKDAY_OFF_PEAK_RANGES


def test_build_deepseek_models_snapshot_omits_time_window_for_implausible_peak_hours() -> None:
    html_text = TIERED_PRICING_HTML.replace(PEAK_HOURS_SENTENCE, "Peak hours are 01:00 - 01:30 UTC")
    payload = build_deepseek_models_snapshot(html_text, SOURCE_URL)

    for model in payload[0]["models"]:
        assert "time_windows" not in model["prices"]


def test_build_deepseek_models_snapshot_handles_mixed_tiered_and_untiered_rows() -> None:
    tiered_cache_rows = (
        '<td rowspan="2">1M INPUT TOKENS (CACHE HIT)</td><td>OFF-PEAK</td><td>$0.007</td><td>$0.022</td></tr>\n'
        "      <tr><td>PEAK</td><td>$0.014</td><td>$0.044</td></tr>"
    )
    assert tiered_cache_rows in TIERED_PRICING_HTML
    html_text = TIERED_PRICING_HTML.replace(
        tiered_cache_rows, "<td>1M INPUT TOKENS (CACHE HIT)</td><td>$0.014</td><td>$0.044</td></tr>"
    )
    payload = build_deepseek_models_snapshot(html_text, SOURCE_URL)

    models = {model["id"]: model for model in payload[0]["models"]}
    flash_prices = models["deepseek-v4-flash"]["prices"]
    assert flash_prices["cache_read_mtok"] == 0.014
    assert flash_prices["input_mtok"] == 0.44
    # The untiered row has no off-peak rate, so the window overrides only the
    # tiered ones.
    assert flash_prices["time_windows"][0]["prices"] == {"input_mtok": 0.22, "output_mtok": 0.66}


def test_build_deepseek_models_snapshot_omits_time_window_without_documented_hours() -> None:
    payload = build_deepseek_models_snapshot(TIERED_PRICING_HTML_WITHOUT_HOURS, SOURCE_URL)

    for model in payload[0]["models"]:
        assert "time_windows" not in model["prices"]
    # Peak stays the published base price even when the hours are unreadable.
    models = {model["id"]: model for model in payload[0]["models"]}
    assert models["deepseek-v4-flash"]["prices"]["input_mtok"] == 0.44


def test_build_deepseek_models_snapshot_omits_time_window_for_untiered_table() -> None:
    payload = build_deepseek_models_snapshot(CURRENT_PRICING_HTML, SOURCE_URL)

    for model in payload[0]["models"]:
        assert "time_windows" not in model["prices"]
