"""Build machine-readable and Markdown audit reports."""

from __future__ import annotations

from typing import Any, Iterable


def build_report(
    duplicate_clusters: Iterable[Iterable[str]] = (),
    quarantine: Iterable[dict[str, Any]] = (),
    new_models: Iterable[str] = (),
    resolved_duplicates: Iterable[Iterable[str]] | None = None,
    source_freshness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    duplicate_clusters_list = [list(cluster) for cluster in duplicate_clusters]
    resolved_duplicates_list = [list(cluster) for cluster in resolved_duplicates] if resolved_duplicates is not None else []
    quarantine_list = [dict(entry) for entry in quarantine]
    new_models_list = list(new_models)

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
    if source_freshness is not None:
        report["source_freshness"] = dict(source_freshness)
    return report


def build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Registry Audit Report",
        "",
        f"- Duplicate clusters: {report['summary']['duplicate_clusters']}",
        f"- Quarantine count: {report['summary']['quarantine_count']}",
    ]

    new_models = report.get("new_models", [])
    if new_models:
        lines.extend(["", "## New Models"])
        for model_name in new_models:
            lines.append(f"- {model_name}")

    duplicate_clusters = report.get("resolved_duplicates") or report.get("duplicate_clusters", [])
    if duplicate_clusters:
        lines.extend(["", "## Duplicate Clusters"])
        for cluster in duplicate_clusters:
            lines.append(f"- {', '.join(cluster)}")

    quarantine = report.get("quarantine", [])
    if quarantine:
        lines.extend(["", "## Quarantine"])
        for entry in quarantine:
            source_model_id = entry.get("source_model_id", "")
            reason = entry.get("reason", "")
            if reason:
                lines.append(f"- {source_model_id}: {reason}")
            else:
                lines.append(f"- {source_model_id}")

    source_freshness = report.get("source_freshness")
    if source_freshness:
        lines.extend(["", "## Source Freshness"])
        for source_name in sorted(source_freshness):
            lines.append(f"- {source_name}: {source_freshness[source_name]}")

    return "\n".join(lines)
