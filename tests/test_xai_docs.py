from pipeline.xai_docs import build_xai_models_snapshot


def test_build_xai_models_snapshot_extracts_embedded_model_objects() -> None:
    html = """
    <html><body><script>
    self.__next_f.push([1,"{\\"$typeName\\":\\"auth_mgmt.LanguageModel\\",\\"name\\":\\"grok-4.20-0309-reasoning\\",\\"aliases\\":[\\"grok-4.20\\"],\\"maxPromptLength\\":2000000,\\"promptTextTokenPrice\\":\\"$n20000\\",\\"cachedPromptTokenPrice\\":\\"$n2000\\",\\"completionTextTokenPrice\\":\\"$n60000\\"}"]);
    self.__next_f.push([1,"{\\"$typeName\\":\\"auth_mgmt.ImageGenerationModel\\",\\"name\\":\\"grok-imagine-image\\",\\"aliases\\":[\\"grok-imagine-image-2026-03-02\\"],\\"imagePrice\\":\\"$n200000000\\",\\"pricePerInputImage\\":\\"$n20000000\\"}"]);
    self.__next_f.push([1,"{\\"$typeName\\":\\"auth_mgmt.VideoGenerationModel\\",\\"name\\":\\"grok-imagine-video\\",\\"aliases\\":[],\\"resolutionPricing\\":[{\\"pricePerSecond\\":\\"$n500000000\\"}],\\"pricePerInputImage\\":\\"$n20000000\\",\\"pricePerInputVideoSecond\\":\\"$n100000000\\"}"]);
    </script></body></html>
    """

    snapshot = build_xai_models_snapshot(html, "https://docs.x.ai/developers/models?cluster=us-east-1")

    assert snapshot["source_url"] == "https://docs.x.ai/developers/models?cluster=us-east-1"
    assert [model["name"] for model in snapshot["language_models"]] == ["grok-4.20-0309-reasoning"]
    assert [model["name"] for model in snapshot["image_generation_models"]] == ["grok-imagine-image"]
    assert [model["name"] for model in snapshot["video_generation_models"]] == ["grok-imagine-video"]
