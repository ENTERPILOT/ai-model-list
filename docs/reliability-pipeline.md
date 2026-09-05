# Reliability Pipeline

This registry is now built from a staged pipeline instead of directly mirroring a third-party source dump.
Every build refreshes the published snapshot bundle from `ENTERPILOT/ai-model-price-list`, plus direct Portkey pricing configs for curated providers, before resolving the registry.

## Build Flow

1. Load curated authority config from `registry/curated/`.
2. Load source snapshots from a fixed snapshot directory.
3. Normalize raw source payloads into shared evidence records.
4. Resolve canonical `models` and `provider_models`.
5. Apply reviewed price corrections from `registry/curated/pricing_overrides.json`.
6. Render `models.json` and `models.min.json`.
7. Emit audit artifacts:
   - `tmp/build/report.json`
   - `tmp/build/report.md`
   - `tmp/build/quarantine.json`

## Commands

Build the registry from the published snapshot bundle:

```bash
python scripts/build_registry.py --report-md tmp/build/report.md
```

Validate the generated registry:

```bash
python scripts/validate.py
```

## Review Checklist

- Verify new canonical models are backed by clean canonical IDs or reviewed aliases.
- Verify provider-specific IDs live in `provider_models`, not `models`.
- Verify `provider_models` only reference curated providers.
- Verify quarantine contains rejected artifacts or unresolved low-confidence records.
- Verify curated alias additions are narrowly scoped and reviewable.
- Verify each pricing override still disagrees with its sources; retire the ones the report lists as no longer needed.

## Notes

- `registry/curated/providers.json` is the curated provider catalog used to keep provider slugs stable.
- `registry/curated/canonical_aliases.json` is the reviewed alias map for canonical model promotion.
- `registry/curated/rejections.json` blocks known garbage IDs and source artifacts before resolution.
- `registry/curated/pricing_overrides.json` corrects prices an upstream publishes wrong, after resolution. Source ranking alone cannot fix a wrong number from the top-ranked source, so these are reviewed by hand, scoped to named fields and providers, and reported on every build so they get retired once upstream agrees.
- Source snapshots are fetched from `https://github.com/ENTERPILOT/ai-model-price-list`.
- Portkey pricing snapshots are fetched directly from `https://configs.portkey.ai/pricing/` so provider coverage and pricing components are not limited by the mirrored bundle.
- Validation now enforces canonical key shape, `owned_by` provider consistency, duplicate-like cluster detection, and orphan model detection.
