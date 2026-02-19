# AI Model List

A public, curated JSON registry of AI model metadata — pricing, capabilities, context limits, and benchmarks. Designed to be consumed by any project that needs structured, up-to-date information about AI models across providers.

[GoModel](https://github.com/ENTERPILOT/GOModel) is one of the primary consumers, fetching this registry at startup to enrich its dynamically discovered models. But the registry is provider-agnostic and self-contained — any AI gateway, cost tracker, model selector, or dashboard can use it.

## How It Works

The registry provides a single `models.json` file with metadata that upstream provider APIs don't expose directly: pricing, capabilities, parameter constraints, and benchmark rankings.

- **Three-layer merge** — provider defaults → model defaults → provider-specific overrides
- **Human-readable** — prices in USD per million tokens, clear field names, no scientific notation
- **Sparse/additive** — only include fields with values; absence = unknown
- **Supplement, not gate** — consumers decide how to handle models missing from the list

## Quick Start

```bash
# Validate the registry
pip install jsonschema
python scripts/validate.py

# Import pricing from external source
python scripts/import_prices.py /path/to/pricing.json

# Import benchmark rankings
python scripts/import_rankings.py /path/to/rankings.json
```

## File Structure

```
ai-model-list/
├── models.json              # The registry — single output file
├── schema.json              # JSON Schema for validation
├── scripts/
│   ├── import_prices.py     # Import pricing from ai-model-price-list repo
│   ├── import_rankings.py   # Import benchmark rankings
│   └── validate.py          # Validate models.json against schema.json
└── README.md
```

## Schema Reference

### Top-Level Structure

```json
{
  "version": 1,
  "updated_at": "2026-02-19T12:00:00Z",
  "providers": { },
  "models": { },
  "provider_models": { }
}
```

### Override / Merge Order

When resolving data for `openai/gpt-4o`:

1. Start with `providers["openai"]` — provider-level defaults
2. Overlay `models["gpt-4o"]` — model-level defaults (provider-agnostic)
3. Overlay `provider_models["openai/gpt-4o"]` — most specific overrides

Null fields in a more specific layer inherit from the layer above. Non-null fields override.

### Providers

Keyed by provider slug (e.g., `openai`, `anthropic`, `google`).

| Field | Type | Description |
|---|---|---|
| `display_name` | string | Human-readable name (required) |
| `website` | string/null | Provider website URL |
| `docs_url` | string/null | API documentation URL |
| `pricing_url` | string/null | Pricing page URL |
| `status_url` | string/null | Status page URL |
| `api_type` | enum | Protocol family: `openai`, `anthropic`, `gemini`, `cohere`, `mistral`, `custom` |
| `default_base_url` | string/null | Default API base URL |
| `auth` | object/null | Auth config: `type` (bearer/header/query_param), `env_var`, `header` |
| `base_url_env` | string/null | Env var for base URL override |
| `supported_modes` | array/null | Supported model modes |
| `default_rate_limits` | object/null | Default `rpm`, `tpm`, `rpd` |

### Models

Keyed by canonical model name (e.g., `gpt-4o`, `claude-opus-4-6`).

| Field | Type | Description |
|---|---|---|
| `display_name` | string | Human-readable name (required) |
| `description` | string/null | Short description |
| `owned_by` | string/null | Originating company slug |
| `family` | string/null | Model family/series |
| `release_date` | date/null | ISO date of release |
| `deprecation_date` | date/null | ISO date of deprecation |
| `tags` | array/null | Curated tags (see Tags enum) |
| `mode` | enum | Primary mode (required, see Mode enum) |
| `modalities` | object/null | `input` and `output` arrays of: `text`, `image`, `audio`, `video` |
| `capabilities` | object/null | Boolean flags (see Capabilities enum) |
| `context_window` | int/null | Max input tokens |
| `max_output_tokens` | int/null | Max output tokens |
| `pricing` | object/null | Pricing in USD per million tokens (see Pricing) |
| `parameters` | object/null | Request parameter constraints |
| `rankings` | object/null | Benchmark scores (see Rankings) |

### Provider Models

Keyed by `provider/model` (e.g., `openai/gpt-4o`, `bedrock/claude-opus-4-6`).

| Field | Type | Description |
|---|---|---|
| `model_ref` | string | References a key in `models` (required) |
| `provider_model_id` | string/null | Actual API model string (if different from canonical) |
| `enabled` | bool | Whether this mapping is active (required) |
| `pricing` | object/null | Override pricing (null = inherit from model) |
| `context_window` | int/null | Override context window |
| `max_output_tokens` | int/null | Override max output |
| `rate_limits` | object/null | Provider-specific `rpm`, `tpm`, `rpd` |
| `endpoints` | array/null | API endpoint paths |
| `regions` | array/null | Region availability (null = global) |

### Pricing Object

All monetary values in USD. Token-based prices are **per million tokens**.

| Field | Type | Description |
|---|---|---|
| `currency` | `"USD"` | Always USD |
| `input_per_mtok` | number/null | Per million input tokens |
| `output_per_mtok` | number/null | Per million output tokens |
| `cached_input_per_mtok` | number/null | Per million cached input tokens |
| `reasoning_output_per_mtok` | number/null | Per million reasoning tokens (o1, etc.) |
| `per_image` | number/null | Per generated image |
| `per_second_input` | number/null | Per second audio/video input |
| `per_second_output` | number/null | Per second audio/video output |
| `per_character_input` | number/null | Per character (TTS) |
| `per_request` | number/null | Flat per request |
| `per_page` | number/null | Per page (OCR) |
| `tiers` | array/null | Context-length pricing tiers |

### Enums

**Mode** — primary model type:

`chat`, `completion`, `embedding`, `image_generation`, `image_edit`, `video_generation`, `video_edit`, `audio_speech`, `audio_transcription`, `rerank`, `moderation`, `ocr`, `search`, `responses`, `code_interpreter`

**Capabilities** (boolean, only include when `true`):

`function_calling`, `parallel_function_calling`, `streaming`, `system_messages`, `vision`, `audio_input`, `audio_output`, `video_input`, `pdf_input`, `json_mode`, `structured_output`, `response_schema`, `reasoning`, `prompt_caching`, `web_search`, `computer_use`, `assistant_prefill`, `video_editing`, `image_input_embedding`

**Tags** (curated, optional):

`flagship`, `budget`, `preview`, `beta`, `deprecated`, `legacy`, `multimodal`, `reasoning`, `long_context`, `fast`, `open_weight`, `fine_tunable`

**Modalities**: `text`, `image`, `audio`, `video`

## Example

Minimal example showing one provider, one model, and one provider_model:

```json
{
  "version": 1,
  "updated_at": "2026-02-19T12:00:00Z",
  "providers": {
    "openai": {
      "display_name": "OpenAI",
      "api_type": "openai",
      "default_base_url": "https://api.openai.com/v1"
    }
  },
  "models": {
    "gpt-4o-mini": {
      "display_name": "GPT-4o Mini",
      "mode": "chat",
      "context_window": 128000,
      "max_output_tokens": 16384,
      "pricing": {
        "currency": "USD",
        "input_per_mtok": 0.15,
        "output_per_mtok": 0.60
      }
    }
  },
  "provider_models": {
    "openai/gpt-4o-mini": {
      "model_ref": "gpt-4o-mini",
      "enabled": true,
      "provider_model_id": "gpt-4o-mini-2024-07-18",
      "endpoints": ["/v1/chat/completions"]
    }
  }
}
```

## Scripts

### `validate.py`

Validates `models.json` against `schema.json` (JSON Schema Draft 2020-12) and checks referential integrity (all `model_ref` and provider slugs resolve).

```bash
pip install jsonschema
python scripts/validate.py
python scripts/validate.py --models path/to/models.json --schema path/to/schema.json
```

### `import_prices.py`

Imports pricing from the `ai-model-price-list` format. Converts per-token costs to per-million-token costs.

```bash
python scripts/import_prices.py /path/to/pricing.json
python scripts/import_prices.py https://example.com/pricing.json
python scripts/import_prices.py /path/to/pricing.json --dry-run
python scripts/import_prices.py /path/to/pricing.json --overwrite  # replace existing values
```

Source format (JSON array):
```json
[
  {
    "model_name": "gpt-4o",
    "input_cost_per_token": 0.0000025,
    "output_cost_per_token": 0.00001,
    "cache_read_input_token_cost": 0.00000125,
    "max_input_tokens": 128000,
    "max_output_tokens": 16384,
    "mode": "chat",
    "supports_function_calling": true,
    "supports_vision": true
  }
]
```

### `import_rankings.py`

Imports benchmark scores and rankings.

```bash
python scripts/import_rankings.py rankings.json
python scripts/import_rankings.py rankings.json --dry-run
```

Source format (JSON array):
```json
[
  {
    "model": "gpt-4o",
    "benchmarks": {
      "chatbot_arena": { "elo": 1287, "rank": 3, "as_of": "2026-02-01" },
      "mmlu_pro": { "score": 0.887, "as_of": "2025-06-01" }
    }
  }
]
```

## Consuming the Registry

Fetch `models.json` via HTTP and resolve model data using the three-layer merge:

```
https://raw.githubusercontent.com/ENTERPILOT/ai-model-list/main/models.json
```

### GoModel

[GoModel](https://github.com/ENTERPILOT/GOModel) is one of the primary consumers. It fetches the registry via the `MODEL_LIST_URL` environment variable:

```bash
export MODEL_LIST_URL=https://raw.githubusercontent.com/ENTERPILOT/ai-model-list/main/models.json
```

- Fetched on startup + hourly refresh
- Non-blocking, best-effort (failures don't prevent startup)
- Merged using the 3-layer override (provider → model → provider_model)
- Attached as `ModelMetadata` to matching registry entries

## Contributing

### Adding a new model

1. Add the model entry to `models` with at minimum `display_name` and `mode`
2. Add `provider_models` entries for each provider that serves it
3. Run `python scripts/validate.py` to verify

### Adding a new provider

1. Add the provider entry to `providers` with `display_name` and `api_type`
2. Add `provider_models` entries linking existing models to this provider
3. Run `python scripts/validate.py` to verify

### Updating pricing

Either edit `models.json` directly or use `import_prices.py` with a source file. Always run `validate.py` after changes.
