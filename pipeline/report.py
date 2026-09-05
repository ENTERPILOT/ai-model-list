"""Build machine-readable and Markdown audit reports."""

from __future__ import annotations

from typing import Any, Iterable

MAX_MARKDOWN_SECTION_ITEMS = 10


def build_report(
    duplicate_clusters: Iterable[Iterable[str]] = (),
    quarantine: Iterable[dict[str, Any]] = (),
    new_models: Iterable[str] = (),
    resolved_duplicates: Iterable[Iterable[str]] | None = None,
    source_freshness: dict[str, Any] | None = None,
    pricing_overrides: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    duplicate_clusters_list = [list(cluster) for cluster in duplicate_clusters]
    resolved_duplicates_list = [list(cluster) for cluster in resolved_duplicates] if resolved_duplicates is not None else []
    quarantine_list = [dict(entry) for entry in quarantine]
    new_models_list = list(new_models)
    pricing_overrides_list = [dict(entry) for entry in pricing_overrides]
    stale_overrides = [entry for entry in pricing_overrides_list if entry.get("status") != "applied"]

    report: dict[str, Any] = {
        "summary": {
            "duplicate_clusters": len(duplicate_clusters_list),
            "quarantine_count": len(quarantine_list),
        },
        "duplicate_clusters": duplicate_clusters_list,
        "resolved_duplicates": resolved_duplicates_list,
        "quarantine": quarantine_list,
        "new_models": new_models_list,
    }
    # Keep the report shape unchanged for builds that carry no corrections.
    if pricing_overrides_list:
        report["summary"]["pricing_overrides_applied"] = len(pricing_overrides_list) - len(stale_overrides)
        report["summary"]["pricing_overrides_stale"] = len(stale_overrides)
        report["pricing_overrides"] = pricing_overrides_list
    if source_freshness is not None:
        report["source_freshness"] = dict(source_freshness)
    return report


def build_markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Registry Audit Report",
        "",
        f"- Duplicate clusters: {summary['duplicate_clusters']}",
        f"- Quarantine count: {summary['quarantine_count']}",
    ]
    if summary.get("pricing_overrides_applied") or summary.get("pricing_overrides_stale"):
        lines.append(f"- Pricing overrides applied: {summary.get('pricing_overrides_applied', 0)}")
        lines.append(f"- Pricing overrides stale: {summary.get('pricing_overrides_stale', 0)}")

    lines.extend(_pricing_override_lines(report.get("pricing_overrides", [])))

    new_models = report.get("new_models", [])
    if new_models:
        lines.extend(["", "## New Models"])
        for model_name in new_models[:MAX_MARKDOWN_SECTION_ITEMS]:
            lines.append(f"- {model_name}")
        if len(new_models) > MAX_MARKDOWN_SECTION_ITEMS:
            lines.append(f"- ... and {len(new_models) - MAX_MARKDOWN_SECTION_ITEMS} more new models")

    duplicate_clusters = report.get("resolved_duplicates") or report.get("duplicate_clusters", [])
    if duplicate_clusters:
        lines.extend(["", "## Duplicate Clusters"])
        for cluster in duplicate_clusters[:MAX_MARKDOWN_SECTION_ITEMS]:
            lines.append(f"- {', '.join(cluster)}")
        if len(duplicate_clusters) > MAX_MARKDOWN_SECTION_ITEMS:
            remaining = len(duplicate_clusters) - MAX_MARKDOWN_SECTION_ITEMS
            lines.append(f"- ... and {remaining} more duplicate clusters")

    quarantine = report.get("quarantine", [])
    if quarantine:
        lines.extend(["", "## Quarantine"])
        for entry in quarantine[:MAX_MARKDOWN_SECTION_ITEMS]:
            source_model_id = entry.get("source_model_id", "")
            reason = entry.get("reason", "")
            if reason:
                lines.append(f"- {source_model_id}: {reason}")
            else:
                lines.append(f"- {source_model_id}")
        if len(quarantine) > MAX_MARKDOWN_SECTION_ITEMS:
            remaining = len(quarantine) - MAX_MARKDOWN_SECTION_ITEMS
            lines.append(f"- ... and {remaining} more quarantine entries")

    source_freshness = report.get("source_freshness")
    if source_freshness:
        lines.extend(["", "## Source Freshness"])
        for source_name in sorted(source_freshness):
            lines.append(f"- {source_name}: {source_freshness[source_name]}")

    return "\n".join(lines)


def _pricing_override_lines(pricing_overrides: list[dict[str, Any]]) -> list[str]:
    """Render curated price corrections, flagging the ones that have gone stale.

    A ``noop`` override means every source now agrees with the curated value, and
    an ``unmatched`` one means its model key no longer resolves. Both are cues to
    retire the entry rather than let it silently pin a price forever.
    """
    if not pricing_overrides:
        return []

    lines = ["", "## Pricing Overrides"]
    for entry in pricing_overrides[:MAX_MARKDOWN_SECTION_ITEMS]:
        model = entry.get("model", "")
        status = entry.get("status", "")
        if status == "applied":
            fields = ", ".join(entry.get("changed_fields", [])) or "no fields"
            targets = ", ".join(entry.get("changed_targets", []))
            lines.append(f"- {model}: corrected {fields} on {targets}")
        elif status == "noop":
            lines.append(f"- {model}: no longer needed — sources already match the curated value")
        else:
            lines.append(f"- {model}: unmatched — no resolved model carries this key")
    if len(pricing_overrides) > MAX_MARKDOWN_SECTION_ITEMS:
        remaining = len(pricing_overrides) - MAX_MARKDOWN_SECTION_ITEMS
        lines.append(f"- ... and {remaining} more pricing overrides")
    return lines
