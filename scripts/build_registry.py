#!/usr/bin/env python3
"""Build the registry artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.loaders import load_curated_config, load_snapshot_payloads
from pipeline.normalize import NORMALIZER_BY_SOURCE
from pipeline.render import render_registry
from pipeline.report import build_markdown_report, build_report
from pipeline.resolve import resolve_registry
from scripts.fetch_sources import fetch_sources_to, snapshot_path_for_run

DUPLICATE_TOKEN_PATTERN = re.compile(r"[^a-z0-9]+")


def build_registry(snapshot_dir: Path, curated_dir: Path) -> dict:
    registry, _, _ = build_registry_artifacts(snapshot_dir=snapshot_dir, curated_dir=curated_dir)
    return registry


def _write_json(path: Path, payload: dict, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    else:
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(f"{text}\n", encoding="utf-8")


def _default_snapshot_dir() -> Path:
    return snapshot_path_for_run(Path.cwd() / "tmp", "latest")


def _normalize_snapshot_payloads(payloads: dict[str, Any], curated: dict[str, Any]) -> list[Any]:
    evidence = []
    rejection_policy = curated.get("rejections", {})
    official_providers = curated.get("source_policies", {}).get("official_sources", [])
    for source_name, payload in payloads.items():
        if source_name == "pydantic_genai" and "xai_models_official" in payloads:
            continue
        normalizer = NORMALIZER_BY_SOURCE.get(source_name)
        if normalizer is None:
            continue
        kwargs: dict[str, Any] = {"rejection_policy": rejection_policy}
        if source_name == "pydantic_genai":
            kwargs["allowed_providers"] = official_providers
        evidence.extend(normalizer(payload, **kwargs))
    return evidence


def _duplicate_like_clusters(model_keys: Iterable[str]) -> list[list[str]]:
    clusters_by_token: dict[str, list[str]] = {}
    for model_key in model_keys:
        token = DUPLICATE_TOKEN_PATTERN.sub("", model_key.lower())
        if not token:
            continue
        clusters_by_token.setdefault(token, []).append(model_key)

    return [
        sorted(cluster)
        for cluster in sorted(clusters_by_token.values(), key=lambda cluster: (-len(cluster), cluster[0]))
        if len(cluster) > 1
    ]


def _resolve_updated_at(snapshot_payloads: dict[str, Any]) -> str:
    if not snapshot_payloads:
        return "1970-01-01T00:00:00Z"

    fetch_metadata = snapshot_payloads.get("fetch_metadata", {})
    fetched_at = fetch_metadata.get("fetched_at") if isinstance(fetch_metadata, dict) else None
    if isinstance(fetched_at, str) and fetched_at:
        return fetched_at

    llm_prices_metadata = snapshot_payloads.get("llm_prices_metadata", {})
    llm_prices_updated_at = llm_prices_metadata.get("updated_at") if isinstance(llm_prices_metadata, dict) else None
    if isinstance(llm_prices_updated_at, str) and llm_prices_updated_at:
        return llm_prices_updated_at

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _source_freshness(snapshot_payloads: dict[str, Any]) -> dict[str, Any]:
    freshness: dict[str, Any] = {}
    fetch_metadata = snapshot_payloads.get("fetch_metadata", {})
    fetched_at = fetch_metadata.get("fetched_at") if isinstance(fetch_metadata, dict) else None
    sources = fetch_metadata.get("sources", {}) if isinstance(fetch_metadata, dict) else {}

    if isinstance(fetched_at, str):
        for source_name in sorted(sources):
            freshness[source_name] = fetched_at

    llm_prices_metadata = snapshot_payloads.get("llm_prices_metadata", {})
    llm_prices_updated_at = llm_prices_metadata.get("updated_at") if isinstance(llm_prices_metadata, dict) else None
    if isinstance(llm_prices_updated_at, str):
        freshness.setdefault("llm_prices", llm_prices_updated_at)

    return freshness


def build_registry_artifacts(snapshot_dir: Path, curated_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    curated = load_curated_config(curated_dir)
    snapshot_payloads = load_snapshot_payloads(snapshot_dir)
    evidence = _normalize_snapshot_payloads(snapshot_payloads, curated)
    resolved_registry, resolve_report = resolve_registry(evidence, curated)
    allowed_providers = set(resolved_registry["providers"])
    resolved_registry["provider_models"] = {
        provider_model_key: provider_model
        for provider_model_key, provider_model in resolved_registry["provider_models"].items()
        if provider_model_key.split("/", 1)[0] in allowed_providers
    }
    updated_at = _resolve_updated_at(snapshot_payloads)
    registry = render_registry(resolved_registry, updated_at=updated_at)
    quarantine = list(resolve_report.get("quarantine", []))
    source_freshness = _source_freshness(snapshot_payloads)
    report = build_report(
        duplicate_clusters=_duplicate_like_clusters(registry["models"]),
        quarantine=quarantine,
        source_freshness=source_freshness or None,
    )
    return registry, report, quarantine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build models.json and related artifacts")
    parser.add_argument("--report-md", type=Path, help="Path to write the Markdown report")
    args = parser.parse_args(argv)

    curated_dir = Path("registry/curated")
    snapshot_dir = fetch_sources_to(_default_snapshot_dir())

    registry, report, quarantine = build_registry_artifacts(snapshot_dir=snapshot_dir, curated_dir=curated_dir)

    _write_json(Path("models.json"), registry)
    _write_json(Path("models.min.json"), registry, compact=True)

    if args.report_md is not None:
        report_json_path = args.report_md.with_suffix(".json")
        quarantine_json_path = args.report_md.with_name("quarantine.json")
        _write_json(report_json_path, report)
        _write_json(quarantine_json_path, {"quarantine": quarantine})
        args.report_md.parent.mkdir(parents=True, exist_ok=True)
        args.report_md.write_text(build_markdown_report(report) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
