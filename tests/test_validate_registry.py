from pathlib import Path

from scripts.build_registry import build_registry


def test_build_registry_returns_top_level_sections(tmp_path: Path) -> None:
    result = build_registry(snapshot_dir=tmp_path, curated_dir=tmp_path)
    assert set(result.keys()) == {"version", "updated_at", "providers", "models", "provider_models"}
