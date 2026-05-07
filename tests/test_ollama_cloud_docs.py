from pipeline.ollama_cloud_docs import build_ollama_cloud_models_snapshot


SOURCE_URL = "https://ollama.com/pricing"

SAMPLE_HTML = """
<html><body>
<ul role="list">
<li x-test-model>
  <a href="/library/kimi-k2.6">
    <h2><span x-test-search-response-title>kimi-k2.6</span></h2>
    <p class="break-words">Kimi K2.6 is an open-source agentic model.</p>
    <span x-test-capability>vision</span>
    <span x-test-capability>tools</span>
    <span x-test-capability>thinking</span>
    <span class="bg-cyan-50">cloud</span>
  </a>
</li>
<li x-test-model>
  <a href="/library/glm-5.1">
    <h2><span x-test-search-response-title>glm-5.1</span></h2>
    <p class="break-words">GLM-5.1 is the next-gen flagship.</p>
    <span x-test-capability>tools</span>
    <span class="bg-cyan-50">cloud</span>
  </a>
</li>
<li x-test-model>
  <a href="/library/llama-local-only">
    <h2><span x-test-search-response-title>llama-local-only</span></h2>
    <p class="break-words">Local-only model, not cloud-eligible.</p>
    <span x-test-capability>tools</span>
  </a>
</li>
</ul>
</body></html>
"""


def test_build_ollama_cloud_models_snapshot_keeps_only_cloud_eligible_models() -> None:
    payload = build_ollama_cloud_models_snapshot(SAMPLE_HTML, SOURCE_URL)

    assert payload[0]["id"] == "ollama_cloud"
    assert payload[0]["pricing_urls"] == [SOURCE_URL]

    ids = [model["id"] for model in payload[0]["models"]]
    assert ids == ["kimi-k2.6", "glm-5.1"]


def test_build_ollama_cloud_models_snapshot_omits_prices() -> None:
    payload = build_ollama_cloud_models_snapshot(SAMPLE_HTML, SOURCE_URL)
    for model in payload[0]["models"]:
        assert "prices" not in model
        assert model["mode"] == "chat"


def test_build_ollama_cloud_models_snapshot_captures_description() -> None:
    payload = build_ollama_cloud_models_snapshot(SAMPLE_HTML, SOURCE_URL)
    models = {model["id"]: model for model in payload[0]["models"]}
    assert "open-source agentic" in models["kimi-k2.6"]["description"]
