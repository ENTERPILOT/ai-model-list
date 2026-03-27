"""Ranking source ingestion and merge helpers."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import re
from typing import Any


ARENA_CATALOG_BASE_URL = "https://raw.githubusercontent.com/lmarena/arena-catalog/main/data"
ARENA_CATALOG_DIRNAME = "arena_catalog"
ARENA_CATALOG_METADATA_FILENAME = "fetch_metadata.json"
ARENA_LEADERBOARD_FILENAMES = ("leaderboard-text.json", "leaderboard-vision.json")
ARENA_SOURCE_URLS = {
    filename: f"{ARENA_CATALOG_BASE_URL}/{filename}"
    for filename in ARENA_LEADERBOARD_FILENAMES
}

# Mapping: (source_file, source_category) -> target ranking key
DEFAULT_ARENA_CATEGORIES = {
    ("leaderboard-text.json", "full"): "chatbot_arena",
    ("leaderboard-text.json", "coding"): "chatbot_arena_coding",
    ("leaderboard-text.json", "math"): "chatbot_arena_math",
    ("leaderboard-text.json", "creative_writing"): "chatbot_arena_creative_writing",
    ("leaderboard-vision.json", "full"): "chatbot_arena_vision",
}

LEADERBOARD_MODEL_OVERRIDES = {
    "claude-3-7-sonnet-20250219-thinking-32k": "claude-3-7-sonnet-20250219",
    "claude-opus-4-20250514-thinking-16k": "claude-opus-4",
    "claude-opus-4-1-20250805-thinking-16k": "claude-opus-4-1",
    "claude-opus-4-5-20251101-thinking-32k": "claude-opus-4-5",
    "claude-sonnet-4-20250514-thinking-32k": "claude-sonnet-4",
    "claude-sonnet-4-5-20250929-thinking-32k": "claude-sonnet-4-5",
    "gemini-3-pro": "gemini-3-pro-preview",
}

_VARIANT_SUFFIX_PATTERNS = (
    re.compile(r"-thinking-\d+k$"),
    re.compile(r"-(?:no-)?thinking$"),
    re.compile(r"-(?:high|medium|low)$"),
    re.compile(r"-(20\d{6})$"),
    re.compile(r"-\d{2}-\d{2}$"),
)


def normalize_leaderboard_model_name(name: str) -> str:
    """Collapse safe leaderboard suffix variants to improve model matching."""
    normalized = name.casefold().strip()
    changed = True
    while changed:
        changed = False
        for pattern in _VARIANT_SUFFIX_PATTERNS:
            candidate = pattern.sub("", normalized)
            if candidate != normalized:
                normalized = candidate
                changed = True
    return normalized


def infer_rankings_as_of(metadata: dict[str, Any] | None) -> str:
    """Resolve an ISO date for ranking entries from snapshot metadata."""
    fetched_at = metadata.get("fetched_at") if isinstance(metadata, dict) else None
    if isinstance(fetched_at, str) and fetched_at:
        return fetched_at[:10]
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def build_model_lookup(models_data: dict) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Build exact and normalized lookups from canonical keys and aliases."""
    exact_lookup: dict[str, set[str]] = defaultdict(set)
    normalized_lookup: dict[str, set[str]] = defaultdict(set)

    for key, model in models_data.get("models", {}).items():
        candidates = [key, *((model or {}).get("aliases") or [])]
        for candidate in candidates:
            if not isinstance(candidate, str) or not candidate:
                continue
            lowered = candidate.casefold()
            exact_lookup[lowered].add(key)
            normalized_lookup[normalize_leaderboard_model_name(lowered)].add(key)

    return exact_lookup, normalized_lookup


def match_model_keys(
    leaderboard_name: str,
    exact_lookup: dict[str, set[str]],
    normalized_lookup: dict[str, set[str]],
) -> list[str]:
    """Match a leaderboard model name to canonical model keys."""
    lowered = leaderboard_name.casefold()
    override = LEADERBOARD_MODEL_OVERRIDES.get(lowered)
    if override is not None:
        return [override]

    if lowered in exact_lookup:
        return sorted(exact_lookup[lowered])

    normalized = normalize_leaderboard_model_name(lowered)
    normalized_matches = normalized_lookup.get(normalized, set())
    if len(normalized_matches) == 1:
        return sorted(normalized_matches)

    return []


def import_arena_rankings(
    leaderboards: dict[str, dict],
    models_data: dict,
    categories: dict[tuple[str, str], str],
    merge: bool = True,
    as_of: str | None = None,
) -> tuple[int, int, list[str]]:
    """Import arena rankings into models_data.

    Returns (updated_model_count, skipped_count, unmatched_arena_names).
    """
    if as_of is None:
        as_of = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    exact_lookup, normalized_lookup = build_model_lookup(models_data)
    updated_keys: set[str] = set()
    all_unmatched: set[str] = set()

    for (file_name, src_category), target_key in categories.items():
        leaderboard = leaderboards.get(file_name)
        if leaderboard is None:
            continue

        category_data = leaderboard.get(src_category)
        if category_data is None:
            continue

        sorted_models = sorted(
            category_data.items(),
            key=lambda item: item[1].get("rating", 0),
            reverse=True,
        )

        for rank, (arena_name, arena_data) in enumerate(sorted_models, start=1):
            rating = arena_data.get("rating")
            if rating is None:
                continue

            matched_keys = match_model_keys(arena_name, exact_lookup, normalized_lookup)
            if not matched_keys:
                all_unmatched.add(arena_name)
                continue

            ranking_entry = {
                "elo": round(rating),
                "rank": rank,
                "as_of": as_of,
            }

            for model_key in matched_keys:
                model = models_data["models"][model_key]
                if model.get("rankings") is None:
                    model["rankings"] = {}

                if merge and target_key in model["rankings"]:
                    existing = model["rankings"][target_key]
                    existing_date = existing.get("as_of", "")
                    if as_of >= existing_date:
                        if existing.get("elo") == ranking_entry["elo"] and existing.get("rank") == rank:
                            continue
                        model["rankings"][target_key] = ranking_entry
                else:
                    model["rankings"][target_key] = ranking_entry

                updated_keys.add(model_key)

    return len(updated_keys), len(all_unmatched), sorted(all_unmatched)


def seed_existing_rankings(models_data: dict[str, Any], existing_registry: dict[str, Any] | None) -> None:
    """Copy previously published rankings into the freshly resolved registry."""
    existing_models = existing_registry.get("models") if isinstance(existing_registry, dict) else None
    if not isinstance(existing_models, dict):
        return

    for model_key, model in models_data.get("models", {}).items():
        if not isinstance(model, dict):
            continue
        existing_model = existing_models.get(model_key)
        if not isinstance(existing_model, dict):
            continue
        existing_rankings = existing_model.get("rankings")
        if not isinstance(existing_rankings, dict) or not existing_rankings:
            continue

        current_rankings = model.get("rankings")
        if current_rankings is None:
            model["rankings"] = deepcopy(existing_rankings)
            continue
        if not isinstance(current_rankings, dict):
            continue

        for ranking_key, ranking_value in existing_rankings.items():
            if ranking_key not in current_rankings:
                current_rankings[ranking_key] = deepcopy(ranking_value)


def apply_snapshot_rankings(models_data: dict, snapshot_payloads: dict[str, Any]) -> dict[str, Any]:
    """Merge ranking snapshots into the in-memory registry and return a summary."""
    summary = {"sources": {}}

    arena_catalog = snapshot_payloads.get("arena_catalog")
    if isinstance(arena_catalog, dict) and arena_catalog:
        as_of = infer_rankings_as_of(snapshot_payloads.get("arena_catalog_metadata"))
        updated, skipped, unmatched = import_arena_rankings(
            arena_catalog,
            models_data,
            categories=DEFAULT_ARENA_CATEGORIES,
            merge=True,
            as_of=as_of,
        )
        summary["sources"]["arena_catalog"] = {
            "updated_models": updated,
            "unmatched_names": skipped,
            "sample_unmatched": unmatched[:10],
            "as_of": as_of,
        }

    return summary


def discover_all_categories(leaderboards: dict[str, dict]) -> dict[tuple[str, str], str]:
    """Build category mapping for all categories found in loaded leaderboards."""
    categories: dict[tuple[str, str], str] = {}
    for file_name, leaderboard in leaderboards.items():
        prefix = file_name.replace("leaderboard-", "").replace(".json", "")
        for src_category in leaderboard:
            if prefix == "text" and src_category == "full":
                target_key = "chatbot_arena"
            elif prefix == "text":
                target_key = f"chatbot_arena_{src_category}"
            else:
                if src_category == "full":
                    target_key = f"chatbot_arena_{prefix}"
                else:
                    target_key = f"chatbot_arena_{prefix}_{src_category}"
            categories[(file_name, src_category)] = target_key
    return categories
