from pathlib import Path


def build_registry(snapshot_dir: Path, curated_dir: Path) -> dict:
    return {
        "version": 1,
        "updated_at": "1970-01-01T00:00:00Z",
        "providers": {},
        "models": {},
        "provider_models": {},
    }
