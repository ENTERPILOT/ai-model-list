#!/usr/bin/env python3
"""Build the registry artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.loaders import load_curated_config
from pipeline.render import render_registry
from pipeline.report import build_markdown_report, build_report
from scripts.fetch_sources import fetch_sources_to, snapshot_path_for_run


def build_registry(snapshot_dir: Path, curated_dir: Path) -> dict:
    load_curated_config(curated_dir)
    resolved = {
        "providers": {},
        "models": {},
        "provider_models": {},
    }
    return render_registry(resolved, updated_at="1970-01-01T00:00:00Z")


def _write_json(path: Path, payload: dict, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    else:
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(f"{text}\n", encoding="utf-8")


def _default_snapshot_dir() -> Path:
    return snapshot_path_for_run(Path.cwd() / "tmp", "latest")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build models.json and related artifacts")
    parser.add_argument("--snapshot-dir", type=Path, help="Path to a source snapshot directory")
    parser.add_argument("--fetch", action="store_true", help="Fetch a fresh snapshot before building")
    parser.add_argument("--report-md", type=Path, help="Path to write the Markdown report")
    args = parser.parse_args(argv)

    curated_dir = Path("registry/curated")
    snapshot_dir = args.snapshot_dir
    if args.fetch:
        snapshot_dir = fetch_sources_to(snapshot_dir or _default_snapshot_dir())
    if snapshot_dir is None:
        snapshot_dir = _default_snapshot_dir()

    registry = build_registry(snapshot_dir=snapshot_dir, curated_dir=curated_dir)
    report = build_report()

    _write_json(Path("models.json"), registry)
    _write_json(Path("models.min.json"), registry, compact=True)

    if args.report_md is not None:
        report_json_path = args.report_md.with_suffix(".json")
        _write_json(report_json_path, report)
        args.report_md.parent.mkdir(parents=True, exist_ok=True)
        args.report_md.write_text(build_markdown_report(report) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
