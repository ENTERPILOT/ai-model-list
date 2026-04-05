"""Detect whether generated registry changes are substantive enough for a PR."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

IGNORED_DATE_FIELDS = frozenset({"updated_at", "as_of"})
NO_MEANINGFUL_CHANGES_EXIT_CODE = 10


def strip_ignored_date_fields(value: Any, ignored_fields: frozenset[str] = IGNORED_DATE_FIELDS) -> Any:
    """Remove freshness-only fields from nested JSON-like structures."""
    if isinstance(value, dict):
        return {
            key: strip_ignored_date_fields(item, ignored_fields)
            for key, item in value.items()
            if key not in ignored_fields
        }
    if isinstance(value, list):
        return [strip_ignored_date_fields(item, ignored_fields) for item in value]
    return value


def has_meaningful_json_changes(
    base_document: Any,
    candidate_document: Any,
    ignored_fields: frozenset[str] = IGNORED_DATE_FIELDS,
) -> bool:
    """Return True when the documents differ after removing ignored date fields."""
    return strip_ignored_date_fields(base_document, ignored_fields) != strip_ignored_date_fields(
        candidate_document,
        ignored_fields,
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_object_exists(base_ref: str, path: Path) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{base_ref}:{path.as_posix()}"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _load_json_from_git(base_ref: str, path: Path) -> Any | None:
    if not _git_object_exists(base_ref, path):
        return None

    result = subprocess.run(
        ["git", "show", f"{base_ref}:{path.as_posix()}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to read {path} from {base_ref}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def has_meaningful_file_changes(
    base_ref: str,
    paths: list[Path],
    ignored_fields: frozenset[str] = IGNORED_DATE_FIELDS,
) -> bool:
    """Return True when any tracked JSON file changed beyond ignored date fields."""
    for path in paths:
        if not path.exists():
            return True

        base_document = _load_json_from_git(base_ref, path)
        if base_document is None:
            return True

        candidate_document = _load_json(path)
        if has_meaningful_json_changes(base_document, candidate_document, ignored_fields):
            return True

    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check whether registry JSON changes should trigger a PR.")
    parser.add_argument("paths", nargs="+", help="JSON files to compare against the base ref")
    parser.add_argument("--base-ref", default="HEAD", help="Git ref to compare against")
    args = parser.parse_args(argv)

    paths = [Path(path) for path in args.paths]
    if has_meaningful_file_changes(args.base_ref, paths):
        print("Meaningful registry changes detected.")
        return 0

    print("Only ignored date-related fields changed; skipping PR.")
    return NO_MEANINGFUL_CHANGES_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
