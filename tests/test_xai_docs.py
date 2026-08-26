from pipeline.xai_docs import build_xai_models_snapshot


def test_build_xai_models_snapshot_extracts_legacy_embedded_model_objects() -> None:
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


def test_build_xai_models_snapshot_reads_public_models_payload() -> None:
    # Current docs ship one plain JSON assignment with every model list grouped
    # per serving cluster. Models repeat across clusters; the first occurrence
    # wins so the snapshot lists each model once with stable pricing.
    html = """
    <html><body><script>globalThis.__XAI_PUBLIC_MODELS__={"clusterConfigs":[
      {"clusterName":"us-east-1",
       "languageModels":[{"name":"grok-4.6","aliases":["grok-4.6-latest"],"promptTextTokenPrice":"20000","completionTextTokenPrice":"60000","maxPromptLength":"2000000"}],
       "imageGenerationModels":[{"name":"grok-imagine-image","aliases":["grok-imagine-image-2026-03-02"],"imagePrice":"200000000","pricePerInputImage":"20000000","resolutionPricing":[{"resolution":"IMAGE_RESOLUTION_1K","pricePerImage":"200000000"}]}],
       "videoGenerationModels":[{"name":"grok-imagine-video","resolutionPricing":[{"resolution":"VIDEO_RESOLUTION_480P","pricePerSecond":"500000000"}],"pricePerInputImage":"20000000"}],
       "audioModels":[{"name":"grok-voice-latest"}]},
      {"clusterName":"us-west-2",
       "languageModels":[{"name":"grok-4.6","promptTextTokenPrice":"99999"}],
       "imageGenerationModels":[{"name":"grok-imagine-image-2.0","imagePrice":"600000000"}]},
      {"clusterName":"us-east-4"}
    ]}</script></body></html>
    """

    snapshot = build_xai_models_snapshot(html, "https://docs.x.ai/developers/models?cluster=us-east-1")

    assert snapshot["source_url"] == "https://docs.x.ai/developers/models?cluster=us-east-1"
    assert [model["name"] for model in snapshot["language_models"]] == ["grok-4.6"]
    assert snapshot["language_models"][0]["promptTextTokenPrice"] == "20000"
    assert [model["name"] for model in snapshot["image_generation_models"]] == ["grok-imagine-image", "grok-imagine-image-2.0"]
    assert snapshot["image_generation_models"][0]["imagePrice"] == "200000000"
    assert [model["name"] for model in snapshot["video_generation_models"]] == ["grok-imagine-video"]
    assert [model["name"] for model in snapshot["audio_models"]] == ["grok-voice-latest"]


def test_build_xai_models_snapshot_returns_empty_buckets_without_catalog() -> None:
    snapshot = build_xai_models_snapshot("<html><body>no catalog here</body></html>", "https://docs.x.ai/developers/models")

    assert snapshot["language_models"] == []
    assert snapshot["image_generation_models"] == []
    assert snapshot["video_generation_models"] == []
    assert snapshot["audio_models"] == []
