#!/usr/bin/env python3
"""Validate models.json against schema.json.

Usage:
    python scripts/validate.py
    python scripts/validate.py --models path/to/models.json --schema path/to/schema.json
"""

import argparse
from collections import defaultdict
import json
import re
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.normalize import infer_owned_by
from pipeline.rules import is_canonical_model_key

try:
    import jsonschema
except ImportError:
    print("Error: jsonschema package required. Install with: pip install jsonschema", file=sys.stderr)
    sys.exit(1)


MAX_REPORTED_DUPLICATE_CLUSTERS = 20
MAX_REPORTED_ORPHAN_MODELS = 10
NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")


def validate(models_path: Path, schema_path: Path) -> list[str]:
    """Validate models.json against schema.json. Returns list of error messages."""
    with open(schema_path) as f:
        schema = json.load(f)
    with open(models_path) as f:
        data = json.load(f)

    # Support both old (< 4.0) and new jsonschema versions
    if hasattr(jsonschema, "Draft202012Validator"):
        validator = jsonschema.Draft202012Validator(schema)
    else:
        validator = jsonschema.Draft7Validator(schema)
    errors = []
    for error in sorted(validator.iter_errors(data), key=lambda e: list(e.path)):
        path = ".".join(str(p) for p in error.absolute_path) or "(root)"
        errors.append(f"  {path}: {error.message}")

    return errors


def check_referential_integrity(models_path: Path) -> list[str]:
    """Check that provider_models reference valid providers and models."""
    with open(models_path) as f:
        data = json.load(f)

    errors = []
    providers = set(data.get("providers", {}).keys())
    models = set(data.get("models", {}).keys())

    for pm_key, pm_value in data.get("provider_models", {}).items():
        # Check key format
        parts = pm_key.split("/", 1)
        if len(parts) != 2:
            errors.append(f"  {pm_key}: invalid key format (expected 'provider/model')")
            continue

        provider_slug, _ = parts

        # Check provider exists
        if provider_slug not in providers:
            errors.append(f"  {pm_key}: provider '{provider_slug}' not found in providers section")

        # Check model_ref exists
        model_ref = pm_value.get("model_ref")
        if model_ref and model_ref not in models:
            errors.append(f"  {pm_key}: model_ref '{model_ref}' not found in models section")

    return errors


def check_canonical_model_keys(data: dict) -> list[str]:
    errors = []
    for model_key in sorted(data.get("models", {})):
        if not is_canonical_model_key(model_key):
            errors.append(f"  {model_key}: provider-specific model key found in models")
    return errors


def check_owned_by_values(data: dict) -> list[str]:
    errors = []
    providers = set(data.get("providers", {}))

    for model_key, model_value in sorted(data.get("models", {}).items()):
        owned_by = model_value.get("owned_by")
        if owned_by is None:
            continue
        if owned_by not in providers:
            errors.append(f"  {model_key}: owned_by '{owned_by}' not found in providers section")
            continue

        expected_owner = infer_owned_by(model_key, model_value.get("display_name"))
        if expected_owner is not None and owned_by != expected_owner:
            errors.append(f"  {model_key}: owned_by '{owned_by}' disagrees with inferred family owner '{expected_owner}'")

    return errors


def check_provider_coverage(data: dict) -> list[str]:
    provider_models = data.get("provider_models", {})
    counts_by_provider: dict[str, int] = defaultdict(int)
    for provider_model_key in provider_models:
        provider_slug, _, _ = provider_model_key.partition("/")
        counts_by_provider[provider_slug] += 1

    errors = []
    for provider_slug in sorted(data.get("providers", {})):
        if counts_by_provider.get(provider_slug, 0) == 0:
            errors.append(f"  {provider_slug}: provider has zero provider_models")
    return errors


def _duplicate_like_token(model_key: str) -> str:
    return NON_ALNUM_PATTERN.sub("", model_key.lower())


def check_duplicate_like_clusters(data: dict) -> list[str]:
    clusters_by_token: dict[str, list[str]] = defaultdict(list)
    for model_key in data.get("models", {}):
        token = _duplicate_like_token(model_key)
        if token:
            clusters_by_token[token].append(model_key)

    clusters = [
        sorted(cluster)
        for cluster in clusters_by_token.values()
        if len(cluster) > 1
    ]
    clusters.sort(key=lambda cluster: (-len(cluster), cluster[0]))

    errors = [f"  duplicate-like cluster: {', '.join(cluster)}" for cluster in clusters[:MAX_REPORTED_DUPLICATE_CLUSTERS]]
    remaining_clusters = len(clusters) - len(errors)
    if remaining_clusters > 0:
        errors.append(f"  ... and {remaining_clusters} more duplicate-like clusters")
    return errors


def check_orphan_model_records(data: dict) -> list[str]:
    model_keys = sorted(data.get("models", {}))
    referenced_models = {
        provider_model.get("model_ref")
        for provider_model in data.get("provider_models", {}).values()
        if provider_model.get("model_ref")
    }
    orphan_models = [model_key for model_key in model_keys if model_key not in referenced_models]
    if not orphan_models:
        return []

    sample = ", ".join(orphan_models[:MAX_REPORTED_ORPHAN_MODELS])
    remaining_orphans = len(orphan_models) - min(len(orphan_models), MAX_REPORTED_ORPHAN_MODELS)
    suffix = f", and {remaining_orphans} more" if remaining_orphans > 0 else ""
    return [f"  orphan model records: {sample}{suffix}"]


def check_registry_quality(data: dict) -> list[str]:
    errors = []
    errors.extend(check_canonical_model_keys(data))
    errors.extend(check_owned_by_values(data))
    errors.extend(check_provider_coverage(data))
    errors.extend(check_duplicate_like_clusters(data))
    errors.extend(check_orphan_model_records(data))
    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate models.json against schema.json")
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--models", type=Path, default=repo_root / "models.json",
                        help="Path to models.json (default: repo root)")
    parser.add_argument("--schema", type=Path, default=repo_root / "schema.json",
                        help="Path to schema.json (default: repo root)")
    args = parser.parse_args()

    if not args.models.exists():
        print(f"Error: {args.models} not found", file=sys.stderr)
        sys.exit(1)
    if not args.schema.exists():
        print(f"Error: {args.schema} not found", file=sys.stderr)
        sys.exit(1)

    all_errors = []
    with open(args.models) as f:
        data = json.load(f)

    # Schema validation
    print(f"Validating {args.models} against {args.schema}...")
    schema_errors = validate(args.models, args.schema)
    if schema_errors:
        all_errors.extend(["Schema validation errors:"] + schema_errors)

    # Referential integrity
    print("Checking referential integrity...")
    ref_errors = check_referential_integrity(args.models)
    if ref_errors:
        all_errors.extend(["Referential integrity errors:"] + ref_errors)

    # Quality validation
    print("Checking registry quality...")
    quality_errors = check_registry_quality(data)
    if quality_errors:
        all_errors.extend(["Quality validation errors:"] + quality_errors)

    # Summary
    if all_errors:
        print()
        for line in all_errors:
            print(line, file=sys.stderr)
        print(
            f"\nFAILED: {len(schema_errors)} schema error(s), {len(ref_errors)} referential integrity error(s), "
            f"{len(quality_errors)} quality error(s)"
        )
        sys.exit(1)
    else:
        # Print summary stats
        n_providers = len(data.get("providers", {}))
        n_models = len(data.get("models", {}))
        n_pm = len(data.get("provider_models", {}))
        print(f"\nPASSED: {n_providers} providers, {n_models} models, {n_pm} provider_models")


if __name__ == "__main__":
    main()
