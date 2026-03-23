"""Pipeline loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_curated_config(curated_dir: Path) -> dict[str, Any]:
    return {
        "providers": _read_json(curated_dir / "providers.json"),
        "source_policies": _read_json(curated_dir / "source_policies.json"),
        "canonical_aliases": _read_json(curated_dir / "canonical_aliases.json"),
        "rejections": _read_json(curated_dir / "rejections.json"),
    }
