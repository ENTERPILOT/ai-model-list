from pipeline.render import render_registry
from pipeline.report import build_markdown_report, build_report


def test_render_registry_outputs_sparse_sorted_sections_and_strips_nulls() -> None:
    resolved = {
        "providers": {
            "zeta": {"display_name": "Zeta", "description": None},
            "alpha": {"display_name": "Alpha"},
        },
        "models": {
            "grok-4": {
                "display_name": "Grok 4",
                "description": None,
                "metadata": {"source": "official", "notes": None},
                "aliases": [None, "grok-four"],
            },
            "claude-4": {"display_name": "Claude 4"},
        },
        "provider_models": {
            "zeta/grok-4": {"model_ref": "grok-4", "enabled": True, "notes": None},
            "alpha/claude-4": {"model_ref": "claude-4", "enabled": True},
        },
    }

    rendered = render_registry(resolved, updated_at="2026-03-23T00:00:00Z")

    assert rendered["version"] == 1
    assert rendered["updated_at"] == "2026-03-23T00:00:00Z"
    assert list(rendered["providers"]) == ["alpha", "zeta"]
    assert list(rendered["models"]) == ["claude-4", "grok-4"]
    assert list(rendered["provider_models"]) == ["alpha/claude-4", "zeta/grok-4"]
    assert rendered["providers"]["zeta"] == {"display_name": "Zeta"}
    assert rendered["models"]["grok-4"] == {
        "aliases": ["grok-four"],
        "display_name": "Grok 4",
        "metadata": {"source": "official"},
    }
    assert rendered["provider_models"]["zeta/grok-4"] == {
        "enabled": True,
        "model_ref": "grok-4",
    }


def test_build_report_counts_duplicate_clusters_and_quarantine_entries() -> None:
    report = build_report(
        duplicate_clusters=[["claude-opus-41", "claude-opus-4.1"], ["grok-4", "grok-four"]],
        quarantine=[
            {"source_model_id": "sample_spec", "reason": "rejected"},
            {"source_model_id": "claude-opus-41", "reason": "unapproved_alias_or_low_confidence_only"},
        ],
    )

    assert report["summary"] == {"duplicate_clusters": 2, "quarantine_count": 2}
    assert report["duplicate_clusters"] == [
        ["claude-opus-41", "claude-opus-4.1"],
        ["grok-4", "grok-four"],
    ]
    assert report["quarantine"][0]["source_model_id"] == "sample_spec"


def test_build_markdown_report_lists_duplicate_clusters_and_quarantine_entries() -> None:
    markdown = build_markdown_report(
        build_report(
            duplicate_clusters=[["claude-opus-41", "claude-opus-4.1"]],
            quarantine=[{"source_model_id": "sample_spec", "reason": "rejected"}],
        )
    )

    assert "# Registry Audit Report" in markdown
    assert "Duplicate clusters: 1" in markdown
    assert "Quarantine count: 1" in markdown
    assert "claude-opus-41" in markdown
    assert "sample_spec" in markdown
