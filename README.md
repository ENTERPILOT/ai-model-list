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
# Install validation and test dependencies
pip install jsonschema pytest

# Build the registry with LMArena + LiveBench rankings,
# and optional Artificial Analysis score rankings
export ARTIFICIAL_ANALYSIS_API_KEY=your_api_key_here
python scripts/build_registry.py --report-md tmp/build/report.md

# Validate the generated registry
python scripts/validate.py --models models.json --schema schema.json
```

## File Structure

```
ai-model-list/
├── .github/workflows/
│   └── update-models.yml    # CI/CD entry point
├── models.json              # The registry — single output file
├── models.min.json          # Minified registry artifact
├── pipeline/                # Shared normalization, resolve, render, ranking logic
├── registry/curated/        # Curated provider policies, aliases, and price corrections
├── schema.json              # JSON Schema for validation
├── scripts/
│   ├── build_registry.py    # Fetch sources and build models.json
│   ├── fetch_sources.py     # Snapshot external sources
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
| `modes` | array | Supported operational modes (required, see Mode enum) |
| `modalities` | object/null | `input` and `output` arrays of: `text`, `image`, `audio`, `video` |
| `capabilities` | object/null | Boolean flags — absent key = unsupported or unknown (see Capabilities enum) |
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
| `time_windows` | array/null | Recurring daily windows with different rates (see Pricing Time Windows) |

### Pricing Time Windows

Some providers charge different per-token rates depending on the time of day — DeepSeek, for example, bills off-peak hours at half its peak rates. The base prices above are the **standard (peak) rates**; each entry in `time_windows` lists the UTC ranges when different rates apply.

| Field | Type | Description |
|---|---|---|
| `label` | string | Window name as published by the provider, e.g. `off_peak` |
| `utc_ranges` | array | Daily UTC ranges when the window is in effect |
| `utc_ranges[].start` | string | Inclusive `HH:MM` UTC start |
| `utc_ranges[].end` | string | Exclusive `HH:MM` UTC end; a range whose end is at or before its start wraps midnight |
| `pricing` | object | Rates replacing the base prices during the window: `input_per_mtok`, `output_per_mtok`, `cached_input_per_mtok`, `cache_write_per_mtok` |

Fields absent from a window's `pricing` keep their base price, and outside every listed range the base prices apply. A consumer that ignores `time_windows` sees the peak rate and so never understates cost.

```json
"pricing": {
  "currency": "USD",
  "input_per_mtok": 0.44,
  "output_per_mtok": 1.32,
  "cached_input_per_mtok": 0.014,
  "time_windows": [
    {
      "label": "off_peak",
      "utc_ranges": [
        { "start": "04:00", "end": "06:00" },
        { "start": "10:00", "end": "01:00" }
      ],
      "pricing": {
        "input_per_mtok": 0.22,
        "output_per_mtok": 0.66,
        "cached_input_per_mtok": 0.007
      }
    }
  ]
}
```

### Enums

**Modes** — operational types (array, at least one required):

`chat`, `completion`, `embedding`, `image_generation`, `image_edit`, `video_generation`, `video_edit`, `audio_speech`, `audio_transcription`, `rerank`, `moderation`, `ocr`, `search`, `responses`, `code_interpreter`

**Capabilities** (boolean, only include when `true`; absent = unsupported or unknown):

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
      "modes": ["chat"],
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

### `build_registry.py`

Fetches upstream source snapshots, resolves canonical models, applies rankings, and writes `models.json` plus `models.min.json`.

```bash
python scripts/build_registry.py
python scripts/build_registry.py --report-md tmp/build/report.md
```

Set `ARTIFICIAL_ANALYSIS_API_KEY` to include optional Artificial Analysis rankings during the build.

### `fetch_sources.py`

Internal helper used by `build_registry.py` to snapshot upstream sources before normalization and merge.

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

1. Add the model entry to `models` with at minimum `display_name` and `modes`
2. Add `provider_models` entries for each provider that serves it
3. Run `python scripts/validate.py` to verify

### Adding a new provider

1. Add the provider entry to `providers` with `display_name` and `api_type`
2. Add `provider_models` entries linking existing models to this provider
3. Run `python scripts/validate.py` to verify

### Updating registry data

Update the curated inputs or source adapters, run `python scripts/build_registry.py --report-md tmp/build/report.md`, then run `python scripts/validate.py --models models.json --schema schema.json`.

### Correcting a wrong upstream price

Prices come from ranked sources (`registry/curated/source_policies.json`), and the highest-ranked one wins — so when an upstream publishes a stale or wrong rate, the registry ships it until that upstream catches up. `registry/curated/pricing_overrides.json` is the reviewed correction layer, applied after resolution:

```json
{
  "overrides": [
    {
      "model": "gpt-5.6-luna",
      "providers": ["openai"],
      "pricing": { "input_per_mtok": 0.2, "output_per_mtok": 1.2 },
      "source_url": "https://developers.openai.com/api/docs/pricing",
      "verified_on": "2026-09-05",
      "reason": "Upstream feed publishes 1.00/6.00; the vendor's own table lists 0.20/1.20."
    }
  ]
}
```

| Field | Description |
|---|---|
| `model` | Canonical model key to correct |
| `providers` | Provider slugs whose `provider_models` entries share the correction. Opt-in — a vendor's published rate says nothing about what a reseller charges, so unlisted providers keep their own price |
| `pricing` | Only the pricing fields being corrected; everything else (`tiers`, `batch_*`, …) stays sourced from upstream |
| `source_url` | The vendor page the corrected numbers were read from; becomes the `pricing_source_url` on records it touches |
| `verified_on` | When a human last checked the numbers against `source_url` |
| `reason` | Why the entry exists — what upstream says, and what the vendor says |

Add an entry only when the vendor's own published page disagrees with the built registry, and keep it narrow. Each build reports every override under **Pricing Overrides** in `tmp/build/report.md`; an override that stops changing anything is listed as no longer needed and should be deleted, because a stale pin becomes the next wrong price the moment the vendor updates.
