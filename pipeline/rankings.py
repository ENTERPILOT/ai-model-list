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

LIVEBENCH_DIRNAME = "livebench"
LIVEBENCH_BASE_URL = "https://livebench.ai"
LIVEBENCH_TABLE_FILENAME = "table.json"
LIVEBENCH_CATEGORIES_FILENAME = "categories.json"
LIVEBENCH_METADATA_FILENAME = "fetch_metadata.json"
LIVEBENCH_RELEASES = (
    "2024-06-24",
    "2024-07-26",
    "2024-08-31",
    "2024-11-25",
    "2025-04-02",
    "2025-04-25",
    "2025-05-30",
    "2025-11-25",
    "2025-12-23",
    "2026-01-08",
)
LIVEBENCH_RANKING_KEYS = {
    "global_average": "livebench",
    "Reasoning": "livebench_reasoning",
    "Coding": "livebench_coding",
    "Agentic Coding": "livebench_agentic_coding",
    "Data Analysis": "livebench_data_analysis",
    "Language": "livebench_language",
    "IF": "livebench_instruction_following",
    "Mathematics": "livebench_math",
}
LIVEBENCH_GLOBAL_AVERAGE_OVERRIDES = {
    "grok-3": 58.0,
    "grok-3-thinking": 72.0,
}
LIVEBENCH_MODEL_OVERRIDES = {
    "chatgpt-4o-latest-2025-01-29": "gpt-4o",
    "chatgpt-4o-latest-2025-01-30": "gpt-4o",
    "chatgpt-4o-latest-2025-03-27": "gpt-4o",
    "claude-3-7-sonnet-20250219-base": "claude-3-7-sonnet-20250219",
    "claude-3-7-sonnet-20250219-thinking-25k": "claude-3-7-sonnet-20250219",
    "claude-3-7-sonnet-20250219-thinking-64k": "claude-3-7-sonnet-20250219",
    "command-a-03-2025": "command-a",
    "gpt-4.5-preview-2025-02-27": "gpt-4.5-preview",
    "grok-3-mini-reasoning-beta": "grok-3-mini",
    "o1-2024-12-17-high": "o1-2024-12-17",
    "o1-2024-12-17-low": "o1-2024-12-17",
    "o1-2024-12-17-medium": "o1-2024-12-17",
    "o3-mini-2025-01-31-high": "o3-mini-2025-01-31",
    "o3-mini-2025-01-31-low": "o3-mini-2025-01-31",
    "o3-mini-2025-01-31-medium": "o3-mini-2025-01-31",
}

ARTIFICIAL_ANALYSIS_BASE_URL = "https://artificialanalysis.ai"
ARTIFICIAL_ANALYSIS_API_KEY_ENV = "ARTIFICIAL_ANALYSIS_API_KEY"
ARTIFICIAL_ANALYSIS_MODELS_API_URL = f"{ARTIFICIAL_ANALYSIS_BASE_URL}/api/v2/data/llms/models"
ARTIFICIAL_ANALYSIS_DIRNAME = "artificial_analysis"
ARTIFICIAL_ANALYSIS_MODELS_FILENAME = "llms_models.json"
ARTIFICIAL_ANALYSIS_METADATA_FILENAME = "fetch_metadata.json"
ARTIFICIAL_ANALYSIS_RANKING_KEYS = {
    "artificial_analysis_intelligence_index": "artificial_analysis_intelligence_index",
    "artificial_analysis_coding_index": "artificial_analysis_coding_index",
    "artificial_analysis_math_index": "artificial_analysis_math_index",
    "mmlu_pro": "mmlu_pro",
    "gpqa": "gpqa",
    "hle": "hle",
    "livecodebench": "livecodebench",
    "scicode": "scicode",
    "math_500": "math_500",
    "aime": "aime",
}

_VARIANT_SUFFIX_PATTERNS = (
    re.compile(r"-thinking-\d+k$"),
    re.compile(r"-(?:no-)?thinking$"),
    re.compile(r"-(?:high|medium|low)$"),
    re.compile(r"-(20\d{6})$"),
    re.compile(r"-(20\d{2}-\d{2}-\d{2})$"),
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
        display_name = (model or {}).get("display_name")
        if isinstance(display_name, str) and display_name:
            candidates.append(display_name)
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
    *,
    overrides: dict[str, str] | None = None,
) -> list[str]:
    """Match a leaderboard model name to canonical model keys."""
    lowered = leaderboard_name.casefold()
    override = overrides.get(lowered) if overrides is not None else None
    if override is None:
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


def match_model_keys_for_candidates(
    candidates: list[str | None],
    exact_lookup: dict[str, set[str]],
    normalized_lookup: dict[str, set[str]],
    *,
    overrides: dict[str, str] | None = None,
) -> list[str]:
    """Return the first successful match across candidate model identifiers."""
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate:
            continue
        matched_keys = match_model_keys(
            candidate,
            exact_lookup,
            normalized_lookup,
            overrides=overrides,
        )
        if matched_keys:
            return matched_keys
    return []


def _merge_ranking_entry(
    model: dict[str, Any],
    target_key: str,
    ranking_entry: dict[str, Any],
    *,
    merge: bool = True,
) -> bool:
    """Merge a ranking entry into a model and return whether it changed."""
    if model.get("rankings") is None:
        model["rankings"] = {}

    if merge and target_key in model["rankings"]:
        existing = model["rankings"][target_key]
        existing_date = existing.get("as_of", "")
        new_date = ranking_entry.get("as_of", "")
        if new_date >= existing_date:
            existing_without_as_of = {key: value for key, value in existing.items() if key != "as_of"}
            ranking_without_as_of = {key: value for key, value in ranking_entry.items() if key != "as_of"}
            if existing_without_as_of == ranking_without_as_of:
                return False
            model["rankings"][target_key] = ranking_entry
            return True
        return False

    if model["rankings"].get(target_key) == ranking_entry:
        return False

    model["rankings"][target_key] = ranking_entry
    return True


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
                if _merge_ranking_entry(model, target_key, ranking_entry, merge=merge):
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


def import_artificial_analysis_rankings(
    payload: dict[str, Any] | list[dict[str, Any]],
    models_data: dict,
    *,
    merge: bool = True,
    as_of: str | None = None,
) -> tuple[int, int, list[str]]:
    """Import score-based rankings from Artificial Analysis into models_data."""
    if as_of is None:
        as_of = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    entries = payload.get("data", []) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        return 0, 0, []

    exact_lookup, normalized_lookup = build_model_lookup(models_data)
    updated_keys: set[str] = set()
    unmatched_models: set[str] = set()

    for source_metric_key, target_key in ARTIFICIAL_ANALYSIS_RANKING_KEYS.items():
        scored_entries: list[tuple[float, dict[str, Any]]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            evaluations = entry.get("evaluations")
            if not isinstance(evaluations, dict):
                continue
            score = evaluations.get(source_metric_key)
            if isinstance(score, (int, float)):
                scored_entries.append((float(score), entry))

        scored_entries.sort(key=lambda item: item[0], reverse=True)

        for rank, (score, entry) in enumerate(scored_entries, start=1):
            slug = entry.get("slug")
            name = entry.get("name")
            creator = entry.get("model_creator") or {}
            creator_slug = creator.get("slug") if isinstance(creator, dict) else None
            candidates = [
                f"{creator_slug}/{slug}" if isinstance(creator_slug, str) and isinstance(slug, str) else None,
                slug if isinstance(slug, str) else None,
                name if isinstance(name, str) else None,
            ]
            matched_keys = match_model_keys_for_candidates(candidates, exact_lookup, normalized_lookup)
            if not matched_keys:
                unmatched_models.add(slug or name or str(entry.get("id", "unknown")))
                continue

            ranking_entry = {
                "score": score,
                "rank": rank,
                "as_of": as_of,
            }

            for model_key in matched_keys:
                model = models_data["models"][model_key]
                if _merge_ranking_entry(model, target_key, ranking_entry, merge=merge):
                    updated_keys.add(model_key)

    return len(updated_keys), len(unmatched_models), sorted(unmatched_models)


def infer_livebench_release(metadata: dict[str, Any] | None) -> str | None:
    """Return the resolved LiveBench leaderboard release from snapshot metadata."""
    if not isinstance(metadata, dict):
        return None
    release = metadata.get("release")
    if isinstance(release, str) and release:
        return release
    return None


def import_livebench_rankings(
    payload: dict[str, Any],
    models_data: dict,
    *,
    merge: bool = True,
    as_of: str | None = None,
    release: str | None = None,
) -> tuple[int, int, list[str]]:
    """Import LiveBench category and overall scores into models_data."""
    if as_of is None:
        as_of = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    table_rows = payload.get("table", []) if isinstance(payload, dict) else []
    categories = payload.get("categories", {}) if isinstance(payload, dict) else {}
    if not isinstance(table_rows, list) or not isinstance(categories, dict):
        return 0, 0, []

    if release is None:
        return 0, 0, []

    score_sets: dict[str, dict[str, float]] = defaultdict(dict)
    for row in table_rows:
        if not isinstance(row, dict):
            continue
        model_name = row.get("model")
        if not isinstance(model_name, str) or not model_name:
            continue

        category_scores: dict[str, float] = {}
        for category_name, tasks in categories.items():
            if not isinstance(category_name, str) or not isinstance(tasks, list):
                continue
            task_scores: list[float] = []
            for task in tasks:
                if not isinstance(task, str):
                    continue
                raw_value = row.get(task)
                try:
                    score = float(raw_value)
                except (TypeError, ValueError):
                    continue
                task_scores.append(score)
            if task_scores:
                category_scores[category_name] = sum(task_scores) / len(task_scores)
                score_sets[category_name][model_name.casefold().strip()] = category_scores[category_name]

        global_average = LIVEBENCH_GLOBAL_AVERAGE_OVERRIDES.get(model_name.casefold().strip())
        if global_average is None and category_scores:
            global_average = sum(category_scores.values()) / len(category_scores)
        if global_average is not None:
            score_sets["global_average"][model_name.casefold().strip()] = global_average

    if not score_sets:
        return 0, 0, []

    exact_lookup, normalized_lookup = build_model_lookup(models_data)
    updated_keys: set[str] = set()
    unmatched_models: set[str] = set()

    for score_key, scores_by_model in score_sets.items():
        target_key = LIVEBENCH_RANKING_KEYS.get(score_key)
        if target_key is None:
            continue

        ranked_models = sorted(
            scores_by_model.items(),
            key=lambda item: (-item[1], item[0]),
        )
        for rank, (source_model_name, score) in enumerate(ranked_models, start=1):
            matched_keys = match_model_keys(
                source_model_name,
                exact_lookup,
                normalized_lookup,
                overrides=LIVEBENCH_MODEL_OVERRIDES,
            )
            if not matched_keys:
                unmatched_models.add(source_model_name)
                continue

            ranking_entry = {
                "score": round(score, 1),
                "rank": rank,
                "as_of": as_of,
            }
            for model_key in matched_keys:
                model = models_data["models"][model_key]
                if _merge_ranking_entry(model, target_key, ranking_entry, merge=merge):
                    updated_keys.add(model_key)

    return len(updated_keys), len(unmatched_models), sorted(unmatched_models)


def apply_snapshot_rankings(models_data: dict, snapshot_payloads: dict[str, Any]) -> dict[str, Any]:
    """Merge ranking snapshots into the in-memory registry and return a summary."""
    summary = {"sources": {}}

    livebench = snapshot_payloads.get("livebench")
    if isinstance(livebench, dict) and livebench:
        as_of = infer_rankings_as_of(snapshot_payloads.get("livebench_metadata"))
        release = infer_livebench_release(snapshot_payloads.get("livebench_metadata"))
        updated, skipped, unmatched = import_livebench_rankings(
            livebench,
            models_data,
            merge=True,
            as_of=as_of,
            release=release,
        )
        summary["sources"]["livebench"] = {
            "updated_models": updated,
            "unmatched_models": skipped,
            "sample_unmatched": unmatched[:10],
            "as_of": as_of,
            "release": release,
        }

    artificial_analysis = snapshot_payloads.get("artificial_analysis")
    if isinstance(artificial_analysis, (dict, list)) and artificial_analysis:
        as_of = infer_rankings_as_of(snapshot_payloads.get("artificial_analysis_metadata"))
        updated, skipped, unmatched = import_artificial_analysis_rankings(
            artificial_analysis,
            models_data,
            merge=True,
            as_of=as_of,
        )
        summary["sources"]["artificial_analysis"] = {
            "updated_models": updated,
            "unmatched_models": skipped,
            "sample_unmatched": unmatched[:10],
            "as_of": as_of,
        }

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
