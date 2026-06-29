from pipeline.xiaomi_docs import build_xiaomi_models_snapshot


SOURCE_URL = "https://mimo.mi.com/static/docs/price/pay-as-you-go.md"
MODEL_SOURCE_URL = "https://mimo.mi.com/static/docs/quick-start/summary/model.md"

PRICING_MARKDOWN = """
# API Pricing

### Domestic Pricing of the Model

<table>
<tr><th></th><th>**Input (Cache Hit)**</th><th>**Input (Cache Miss)**</th><th>**Output**</th></tr>
<tr><td>`mimo-v2.5`</td><td>¥0.02</td><td>¥1.00</td><td>¥2.00</td></tr>
</table>

### Overseas Pricing of the Model

**MiMo-V2.5 Series**

<table>
<thead>
<tr>
<th></th>
<th>**Input (Cache Hit)**</th>
<th>**Input (Cache Miss)**</th>
<th>**Output**</th>
</tr>
</thead>
<tbody>
<tr>
<td>`mimo-v2.5-pro`</td>
<td><p>$0.0036</p></td>
<td><p>$0.435</p></td>
<td><p>$0.87</p></td>
</tr>
<tr>
<td>`mimo-v2.5`</td>
<td><p>$0.0028</p></td>
<td><p>$0.14</p></td>
<td><p>$0.28</p></td>
</tr>
</tbody>
</table>
"""


def test_build_xiaomi_models_snapshot_parses_overseas_mimo_v25_prices() -> None:
    payload = build_xiaomi_models_snapshot(
        PRICING_MARKDOWN,
        SOURCE_URL,
        model_source_url=MODEL_SOURCE_URL,
    )

    assert payload[0]["id"] == "xiaomi"
    assert payload[0]["pricing_urls"] == [SOURCE_URL]
    assert payload[0]["model_urls"] == [MODEL_SOURCE_URL]

    models = {model["id"]: model for model in payload[0]["models"]}
    assert models["mimo-v2.5"] == {
        "id": "mimo-v2.5",
        "name": "Xiaomi MiMo V2.5",
        "mode": "chat",
        "context_window": 1_000_000,
        "max_output_tokens": 128_000,
        "prices": {
            "input_mtok": 0.14,
            "cache_read_mtok": 0.0028,
            "output_mtok": 0.28,
        },
    }
    assert models["mimo-v2.5-pro"]["prices"] == {
        "input_mtok": 0.435,
        "cache_read_mtok": 0.0036,
        "output_mtok": 0.87,
    }
