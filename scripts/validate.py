#!/usr/bin/env python3
"""Validate models.json against schema.json.

Usage:
    python scripts/validate.py
    python scripts/validate.py --models path/to/models.json --schema path/to/schema.json
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import jsonschema
except ImportError:
    print("Error: jsonschema package required. Install with: pip install jsonschema", file=sys.stderr)
    sys.exit(1)


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

    # Summary
    if all_errors:
        print()
        for line in all_errors:
            print(line, file=sys.stderr)
        print(f"\nFAILED: {len(schema_errors)} schema error(s), {len(ref_errors)} referential integrity error(s)")
        sys.exit(1)
    else:
        # Print summary stats
        with open(args.models) as f:
            data = json.load(f)
        n_providers = len(data.get("providers", {}))
        n_models = len(data.get("models", {}))
        n_pm = len(data.get("provider_models", {}))
        print(f"\nPASSED: {n_providers} providers, {n_models} models, {n_pm} provider_models")


if __name__ == "__main__":
    main()
