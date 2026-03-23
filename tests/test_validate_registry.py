from pathlib import Path

from pipeline.loaders import load_curated_config
from scripts.build_registry import build_registry


def test_build_registry_returns_top_level_sections(tmp_path: Path) -> None:
    result = build_registry(snapshot_dir=tmp_path, curated_dir=tmp_path)
    assert set(result.keys()) == {"version", "updated_at", "providers", "models", "provider_models"}


def test_load_curated_config_reads_authority_files() -> None:
    config = load_curated_config(Path(__file__).resolve().parents[1] / "registry/curated")
    assert "providers" in config
    assert "source_policies" in config
    assert "canonical_aliases" in config
    assert "rejections" in config
