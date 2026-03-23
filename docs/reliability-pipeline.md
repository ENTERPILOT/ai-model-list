# Reliability Pipeline

This registry is now built from a staged pipeline instead of directly mirroring a third-party source dump.

## Build Flow

1. Load curated authority config from `registry/curated/`.
2. Load source snapshots from a fixed snapshot directory.
3. Normalize raw source payloads into shared evidence records.
4. Resolve canonical `models` and `provider_models`.
5. Render `models.json` and `models.min.json`.
6. Emit audit artifacts:
   - `tmp/build/report.json`
   - `tmp/build/report.md`
   - `tmp/build/quarantine.json`

## Commands

Build from an existing snapshot:

```bash
python scripts/build_registry.py --snapshot-dir ../ai-model-price-list/sources --report-md tmp/build/report.md
```

Fetch a fresh snapshot and build:

```bash
python scripts/build_registry.py --fetch --report-md tmp/build/report.md
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

## Notes

- `registry/curated/providers.json` is the curated provider catalog used to keep provider slugs stable.
- `registry/curated/canonical_aliases.json` is the reviewed alias map for canonical model promotion.
- `registry/curated/rejections.json` blocks known garbage IDs and source artifacts before resolution.
- Validation now enforces canonical key shape, `owned_by` provider consistency, duplicate-like cluster detection, and orphan model detection.
